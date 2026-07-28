#!/usr/bin/env python3
"""
build_raw_dataset.py

Builds the canonical raw-events dataset for P2 by flattening each session's
nested `events` array and stacking every session's rows into one long table —
the P2 equivalent of P1's build_raw_dataset.py, which only had to concatenate
already-flat events.csv files. P2 has no such file; this script IS the
flattening step P1 never needed (see repo_map.md / setup guide, and memory
note: P2 has no browser-side windowing or auth_windows.csv equivalent —
canonical raw events is the only foundation everything downstream builds on).

Design principles:
- session.json is the source of truth. Nothing here recomputes or corrects
  values the way P1's typing-reaction-time fix did — that's a later concern,
  once there's a reason to.
- Permissive inclusion: only sessions that fail *structural* parsing are
  excluded (can't trust the shape of the data at all). Sessions with
  behavioural issues (e.g. incomplete tasks) are still included in the raw
  layer — filtering on quality happens downstream, same "log everything,
  decide later" philosophy P1 used for its behavioural-candidate filtering.
- Every row carries session-identifying columns (sessionId, participantId,
  sessionIndex, plus device/app context) so sessions stay fully separable
  after stacking — you can always `df[df.sessionId == X]` back out a single
  session, or group by participantId across sessions.
- Payload fields vary by event `kind` (a touchstart payload shares almost
  nothing with a deviceorientation payload). Rather than force one narrow
  schema, every distinct payload key across all 44+ kinds becomes its own
  sparse `payload_<key>` column — populated only for the kinds that actually
  carry it, NaN elsewhere. This is the auto-discovery pattern already used in
  01_raw_behavioural_explorer.ipynb, applied here as canonical output rather
  than ad hoc notebook exploration. The two payload fields that are
  themselves nested (task_end.evidence, session_complete.qualityWarnings) are
  preserved as JSON strings rather than exploded further.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

EXPECTED_SCHEMA_NAME = "continuous_auth_behavioural_biometrics_app"
MIN_SCHEMA_VERSION = (1, 2, 0)

COMMON_EVENT_FIELDS = [
    "kind", "tRelMs", "timestampIso", "taskId", "taskIndex",
    "screenId", "componentId", "activeArea", "instructionArea",
]

def _ctx_int(session: dict, key: str) -> Optional[int]:
    """fatigue and focusLevel come off HTML range sliders, so session.context holds
    them as string digits (e.g. "3"). Cast to int here rather than downstream, so
    every consumer of the raw dataset gets a numeric column instead of an object
    column of numeral strings that happens to sort correctly by luck."""
    raw = (session.get("context") or {}).get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# The context questionnaire (app.js renderContext / collectContextAnswers) writes
# every answer into session.context under these exact keys. Segment-chip and
# select fields are slugify()'d strings (e.g. "morning", "one_handed_right");
# fatigue and focusLevel are range-slider values and need int casting (see
# _ctx_int above) rather than being left as numeral strings.
CONTEXT_STRING_FIELDS = [
    "timeOfDay", "inputDevice", "movement", "environmentNoise",
    "privacy", "alcohol", "caffeine", "posture", "handUse",
]

SESSION_LEVEL_FIELDS = {
    "sessionId": lambda s: s.get("sessionId"),
    "participantId": lambda s: s.get("participantId"),
    "sessionIndex": lambda s: s.get("sessionIndex"),
    "identitySource": lambda s: s.get("identitySource"),
    "schemaVersion": lambda s: s.get("schemaVersion"),
    "appVersion": lambda s: s.get("appVersion"),
    "deviceFamily": lambda s: (s.get("device") or {}).get("deviceFamily"),
    "devicePlatform": lambda s: (s.get("context") or {}).get("devicePlatform"),
    "deviceModel": lambda s: (s.get("context") or {}).get("deviceModel"),
    "consentVersion": lambda s: (s.get("context") or {}).get("consentVersion"),
    "completedNormally": lambda s: s.get("completedNormally"),
    "usableForSignalExtraction": lambda s: (s.get("qualitySummary") or {}).get("usableForSignalExtraction"),
    "sessionDurationMs": lambda s: s.get("sessionDurationMs"),
    # --- context questionnaire (previously collected by the app but not extracted here) ---
    "ctxFatigue": lambda s: _ctx_int(s, "fatigue"),
    "ctxFocusLevel": lambda s: _ctx_int(s, "focusLevel"),
    **{f"ctx{key[0].upper()}{key[1:]}": (lambda s, key=key: (s.get("context") or {}).get(key))
       for key in CONTEXT_STRING_FIELDS},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical raw P2 events dataset from validated session.json files.")
    p.add_argument("--raw-sessions-dir", type=str, default="data/raw/sessions")
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--out-parquet", type=str, default="raw_events.parquet")
    p.add_argument("--out-csv", type=str, default="raw_events.csv")
    p.add_argument("--out-schema", type=str, default="raw_events_schema.json")
    p.add_argument("--out-manifest", type=str, default="raw_events_session_manifest.csv")
    p.add_argument("--skip-csv", action="store_true", help="Skip writing the CSV copy (can be large/slow).")
    return p.parse_args()


def _version_tuple(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(v).split("."))
    except (ValueError, AttributeError):
        return (0,)


def load_session_structural(session_dir: Path) -> Tuple[Optional[dict], List[str]]:
    """Lightweight structural gate — permissive on quality, strict on parseability.
    Mirrors validate_raw_sessions.py's structural checks but does not duplicate its
    behavioural checks, since the raw layer intentionally keeps low-quality sessions."""
    sid = session_dir.name
    issues: List[str] = []

    session_path = session_dir / "session.json"
    if not session_path.exists():
        return None, ["missing session.json"]

    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, [f"session.json is not valid JSON: {e}"]

    if session.get("schemaName") != EXPECTED_SCHEMA_NAME:
        issues.append(f"schemaName mismatch: got '{session.get('schemaName')}'")

    schema_version = session.get("schemaVersion")
    if schema_version is None or _version_tuple(schema_version) < MIN_SCHEMA_VERSION:
        issues.append(f"schemaVersion below minimum: got '{schema_version}'")

    if str(session.get("sessionId", "")) != sid:
        issues.append("sessionId differs from folder name")

    events = session.get("events")
    if not isinstance(events, list) or len(events) == 0:
        issues.append("events missing, not a list, or empty")

    if issues:
        return None, issues

    return session, []


def flatten_session_events(session: dict) -> List[dict]:
    """One row per raw event, with session-identifying columns attached to every row
    so sessions remain fully separable after all sessions are stacked together."""
    session_meta = {col: getter(session) for col, getter in SESSION_LEVEL_FIELDS.items()}
    events = session.get("events") or []

    rows: List[dict] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue

        row: Dict[str, object] = dict(session_meta)
        row["eventIndex"] = event_index
        row["rowId"] = f"{session_meta['sessionId']}__{event_index}"

        for field in COMMON_EVENT_FIELDS:
            row[field] = event.get(field)

        payload = event.get("payload")
        if isinstance(payload, dict):
            for key, value in payload.items():
                col = f"payload_{key}"
                if isinstance(value, (dict, list)):
                    row[col] = json.dumps(value)
                else:
                    row[col] = value

        rows.append(row)

    return rows


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_sessions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()]) if raw_dir.exists() else []

    all_rows: List[dict] = []
    manifest_rows: List[dict] = []

    for sdir in session_dirs:
        session, issues = load_session_structural(sdir)

        if session is None:
            manifest_rows.append({
                "sessionId": sdir.name,
                "included": False,
                "eventCount": 0,
                "participantId": None,
                "sessionIndex": None,
                "issues": "; ".join(issues),
            })
            print(f"EXCLUDED {sdir.name}: {'; '.join(issues)}")
            continue

        rows = flatten_session_events(session)
        all_rows.extend(rows)

        manifest_rows.append({
            "sessionId": sdir.name,
            "included": True,
            "eventCount": len(rows),
            "participantId": session.get("participantId"),
            "sessionIndex": session.get("sessionIndex"),
            "issues": "",
        })
        print(f"INCLUDED {sdir.name}: {len(rows)} events")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / args.out_manifest
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path} ({len(manifest_df)} sessions)")

    if not all_rows:
        print("No included sessions — no canonical raw events written.")
        return 1

    # Column order: session-identifying columns first, then row/event identity,
    # then common event fields, then payload_* columns alphabetically — so the
    # same sessionId/participantId/sessionIndex triple is always the leftmost,
    # easiest-to-scan block regardless of which event kinds happen to be present.
    df = pd.DataFrame(all_rows)
    session_cols = list(SESSION_LEVEL_FIELDS.keys())
    id_cols = ["rowId", "eventIndex"]
    common_cols = COMMON_EVENT_FIELDS
    payload_cols = sorted(c for c in df.columns if c.startswith("payload_"))
    ordered_cols = session_cols + id_cols + common_cols + payload_cols
    df = df[[c for c in ordered_cols if c in df.columns]]

    df = df.sort_values(["sessionId", "eventIndex"]).reset_index(drop=True)

    parquet_path = out_dir / args.out_parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df)} rows, {len(df.columns)} columns)")

    if not args.skip_csv:
        csv_path = out_dir / args.out_csv
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    schema = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sessions_included": int(manifest_df["included"].sum()),
        "sessions_excluded": int((~manifest_df["included"]).sum()),
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "non_null_pct": round(float(df[col].notna().mean()) * 100, 2),
            }
            for col in df.columns
        },
    }
    schema_path = out_dir / args.out_schema
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {schema_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
