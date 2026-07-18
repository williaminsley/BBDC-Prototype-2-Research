#!/usr/bin/env python3
"""
validate_raw_sessions.py

Validates P2 raw session export bundles (data/raw/sessions/{sessionId}/session.json
+ firestore_meta.json) before canonical raw-dataset construction. Mirrors the
structural/behavioural/consistency-warning split and PASS/WARN/FAIL report style
of BBDC-Prototype-1-Research's validate_raw_sessions.py, adapted for P2's single
nested-JSON-per-session format instead of P1's events.csv + auth_windows.csv.

Confirmed finding this script deliberately encodes (see repo_map.md / setup guide):
session.json's own `uploadStatus` field is written to the JSON blob BEFORE
firebase.js flips it to "success" — it will read "attempting" for every session,
forever, regardless of real upload outcome. This script does NOT treat that field
as meaningful. The presence of a firestore_meta.json (synced only when a Firestore
sessions_p2 doc exists) is treated as the real proof of successful upload, since
addDoc() in firebase.js only runs after uploadBytes() succeeds.

firebase.js was subsequently patched to also write uploadStatus: "success" and
uploadCompletedAtIso directly onto the Firestore doc (safe there, since that write
only happens after the Storage upload has already succeeded). Sessions synced
before that app fix won't have this field in firestore_meta.json at all — this
script treats its absence as fine (not a fail), and only warns if it's present
and NOT "success".
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

EXPECTED_SCHEMA_NAME = "continuous_auth_behavioural_biometrics_app"
MIN_SCHEMA_VERSION = (1, 2, 0)

REQUIRED_TOP_LEVEL_KEYS = {
    "schemaName",
    "schemaVersion",
    "appVersion",
    "sessionId",
    "participantId",
    "identitySource",
    "sessionIndex",
    "startedAtIso",
    "completedAtIso",
    "completedNormally",
    "events",
    "taskSummary",
    "qualitySummary",
}

COMMON_EVENT_FIELDS = {"kind", "tRelMs"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate P2 raw session export bundles before canonical build.")
    p.add_argument("--raw-sessions-dir", type=str, default="data/raw/sessions")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--out-json", type=str, default="session_export_validation_p2.json")
    p.add_argument("--out-md", type=str, default="session_export_validation_p2.md")
    p.add_argument("--min-sessions", type=int, default=1)
    p.add_argument("--min-participants", type=int, default=1)
    p.add_argument("--min-events-per-session", type=int, default=500)
    p.add_argument("--require-firestore-meta", action="store_true", default=True)
    return p.parse_args()


def _version_tuple(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(v).split("."))
    except (ValueError, AttributeError):
        return (0,)


def _empty_stats() -> Dict[str, object]:
    return {
        "eventCount": 0,
        "taskCount": 0,
        "expectedTaskCount": None,
        "completedNormally": None,
        "usableForSignalExtraction": None,
        "deviceFamily": None,
        "hasMotion": None,
        "hasOrientation": None,
        "sessionDurationMs": None,
    }


def check_session(session_dir: Path, args: argparse.Namespace) -> Tuple[Dict[str, object], str]:
    sid = session_dir.name

    structural_issues: List[str] = []
    behavioural_issues: List[str] = []
    consistency_warnings: List[str] = []
    warnings: List[str] = []

    session_path = session_dir / "session.json"
    meta_path = session_dir / "firestore_meta.json"

    if not session_path.exists():
        structural_issues.append("missing session.json")
        return {
            **_empty_stats(),
            "structural_issues": structural_issues,
            "behavioural_issues": behavioural_issues,
            "consistency_warnings": consistency_warnings,
            "warnings": warnings,
            "context": {},
        }, ""

    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        structural_issues.append(f"session.json is not valid JSON: {e}")
        return {
            **_empty_stats(),
            "structural_issues": structural_issues,
            "behavioural_issues": behavioural_issues,
            "consistency_warnings": consistency_warnings,
            "warnings": warnings,
            "context": {},
        }, ""

    # --- structural: required keys ---
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - session.keys()
    if missing_keys:
        structural_issues.append(f"missing top-level keys: {sorted(missing_keys)}")

    # --- structural: schema identity ---
    if session.get("schemaName") != EXPECTED_SCHEMA_NAME:
        structural_issues.append(
            f"schemaName mismatch: expected '{EXPECTED_SCHEMA_NAME}', got '{session.get('schemaName')}'"
        )

    schema_version = session.get("schemaVersion")
    if schema_version is None or _version_tuple(schema_version) < MIN_SCHEMA_VERSION:
        structural_issues.append(
            f"schemaVersion must be >= {'.'.join(map(str, MIN_SCHEMA_VERSION))}, got '{schema_version}'"
        )

    # --- structural: folder name / sessionId agreement ---
    if str(session.get("sessionId", "")) != sid:
        structural_issues.append("session.json sessionId differs from folder name")

    # --- structural: participantId present ---
    pid = str(session.get("participantId") or "")
    if not pid:
        structural_issues.append("participantId missing or empty")

    # --- structural: events shape ---
    events = session.get("events")
    if not isinstance(events, list) or len(events) == 0:
        structural_issues.append("events is missing, not a list, or empty")
        events = []
    else:
        bad_events = [
            i for i, e in enumerate(events)
            if not isinstance(e, dict) or not COMMON_EVENT_FIELDS.issubset(e.keys())
        ]
        if bad_events:
            sample = bad_events[:5]
            structural_issues.append(
                f"{len(bad_events)} event(s) missing required fields {sorted(COMMON_EVENT_FIELDS)}; "
                f"example indices: {sample}"
            )

    # --- firestore_meta.json: the real upload-success signal, not session.uploadStatus ---
    firestore_meta = None
    if not meta_path.exists():
        if args.require_firestore_meta:
            structural_issues.append(
                "missing firestore_meta.json — no Firestore sessions_p2 doc found for this session "
                "(addDoc only runs after uploadBytes succeeds in firebase.js, so this is the real "
                "upload-success signal; session.json's own uploadStatus field always reads "
                "'attempting' and should not be used)"
            )
        else:
            warnings.append("missing firestore_meta.json — upload success cannot be confirmed")
    else:
        try:
            firestore_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            structural_issues.append(f"firestore_meta.json is not valid JSON: {e}")

    if firestore_meta is not None:
        if str(firestore_meta.get("sessionId", "")) != sid:
            structural_issues.append("firestore_meta.json sessionId differs from folder name")

        # Cross-check Firestore's cached metadata against the actual session.json content —
        # these are written at different times (Firestore doc write happens after the Storage
        # upload) and can drift if the app changes what it writes into either one.
        fs_event_count = firestore_meta.get("eventCount")
        real_event_count = len(events)
        if fs_event_count is not None and fs_event_count != real_event_count:
            consistency_warnings.append(
                f"firestore eventCount ({fs_event_count}) disagrees with session.json events length ({real_event_count})"
            )

        fs_usable = firestore_meta.get("usableForSignalExtraction")
        real_usable = session.get("qualitySummary", {}).get("usableForSignalExtraction")
        if fs_usable is not None and real_usable is not None and fs_usable != real_usable:
            consistency_warnings.append(
                f"firestore usableForSignalExtraction ({fs_usable}) disagrees with "
                f"session.json qualitySummary.usableForSignalExtraction ({real_usable})"
            )

        # firebase.js was updated to write uploadStatus/uploadCompletedAtIso onto the Firestore
        # doc itself (after uploadBytes succeeds, so it's safe to say "success" there). Sessions
        # synced before that fix won't have this field at all — the None guard keeps those from
        # being wrongly flagged as a problem.
        fs_upload_status = firestore_meta.get("uploadStatus")
        if fs_upload_status is not None and fs_upload_status != "success":
            consistency_warnings.append(
                f"firestore uploadStatus is '{fs_upload_status}', not 'success'"
            )

    # --- behavioural: was the session actually a good one? ---
    quality = session.get("qualitySummary", {}) if isinstance(session.get("qualitySummary"), dict) else {}
    completed_normally = session.get("completedNormally")
    usable = quality.get("usableForSignalExtraction")
    expected_task_count = quality.get("expectedTaskCount")
    completed_task_count = quality.get("completedTaskCount")
    missing_task_ids = quality.get("missingTaskIds") or []
    quality_warnings = quality.get("warnings") or []

    if completed_normally is not True:
        behavioural_issues.append(f"completedNormally is not True (got {completed_normally!r})")

    if usable is not True:
        behavioural_issues.append(f"qualitySummary.usableForSignalExtraction is not True (got {usable!r})")

    if missing_task_ids:
        behavioural_issues.append(f"missingTaskIds is non-empty: {missing_task_ids}")

    if expected_task_count is not None and completed_task_count is not None and completed_task_count < expected_task_count:
        behavioural_issues.append(
            f"completedTaskCount ({completed_task_count}) < expectedTaskCount ({expected_task_count})"
        )

    if quality_warnings:
        # qualitySummary warnings (e.g. missing_motion_events) are informational, not necessarily
        # fatal — usableForSignalExtraction already excludes motion/orientation from its own gate.
        warnings.append(f"qualitySummary.warnings present: {quality_warnings}")

    if len(events) < args.min_events_per_session:
        behavioural_issues.append(f"too few events: {len(events)} < {args.min_events_per_session}")

    # --- context for the report ---
    context = {
        "deviceFamily": session.get("device", {}).get("deviceFamily") if isinstance(session.get("device"), dict) else None,
        "devicePlatform": session.get("context", {}).get("devicePlatform") if isinstance(session.get("context"), dict) else None,
        "appVersion": session.get("appVersion"),
        "schemaVersion": schema_version,
        "identitySource": session.get("identitySource"),
    }

    result = {
        "eventCount": len(events),
        "taskCount": len(session.get("taskSummary") or []),
        "expectedTaskCount": expected_task_count,
        "completedNormally": completed_normally,
        "usableForSignalExtraction": usable,
        "deviceFamily": context["deviceFamily"],
        "hasMotion": quality.get("hasMotion"),
        "hasOrientation": quality.get("hasOrientation"),
        "sessionDurationMs": session.get("sessionDurationMs"),
        "context": context,
        "structural_issues": structural_issues,
        "behavioural_issues": behavioural_issues,
        "consistency_warnings": consistency_warnings,
        "warnings": warnings,
    }
    return result, pid


def render_md(summary: dict) -> str:
    lines = [
        "# P2 Session Export Validation",
        "",
        f"- **Generated:** {summary['generated_at_utc']}",
        f"- **Verdict:** **{summary['verdict']}**",
        f"- **Sessions scanned:** {summary['sessions_scanned']}",
        f"- **Participants found:** {summary['participants_found']}",
        f"- **Pass sessions:** {summary['pass_sessions']}",
        f"- **Warn-only sessions:** {summary['warn_sessions']}",
        f"- **Fail sessions:** {summary['fail_sessions']}",
        "",
        "## Global checks",
    ]
    for c in summary["global_checks"]:
        lines.append(f"- {c}")

    lines += [
        "",
        "## Notes",
        "- This script validates raw P2 session export bundles (session.json + firestore_meta.json)",
        "  before canonical raw/window/candidate dataset construction.",
        "- session.json's own `uploadStatus` field is NOT used as a success signal — it is written",
        "  before firebase.js flips it to 'success', so it always reads 'attempting' regardless of",
        "  real outcome. Presence of firestore_meta.json is the real success signal instead.",
        "- Structural issues are hard export/data-contract problems.",
        "- Behavioural issues indicate an incomplete or low-quality session.",
        "- Consistency warnings come from firestore_meta.json vs session.json reconciliation and do",
        "  not by themselves define modelling truth.",
        "",
        "## Session checks",
    ]

    for sid, info in summary["sessions"].items():
        status = "PASS"
        if info["structural_issues"] or info["behavioural_issues"]:
            status = "FAIL"
        elif info["consistency_warnings"] or info["warnings"]:
            status = "WARN"

        lines.append(f"- `{sid}`: {status}")
        ctx = info.get("context", {})
        if ctx:
            ctx_items = [f"{k}={v}" for k, v in ctx.items() if v not in (None, "", "nan")]
            if ctx_items:
                lines.append(f"  - CONTEXT: {', '.join(ctx_items)}")

        lines.append(
            f"  - STATS: events={info['eventCount']}, tasks={info['taskCount']}/{info['expectedTaskCount']}, "
            f"completedNormally={info['completedNormally']}, usableForSignalExtraction={info['usableForSignalExtraction']}, "
            f"deviceFamily={info['deviceFamily']}, durationMs={info['sessionDurationMs']}"
        )

        for issue in info["structural_issues"]:
            lines.append(f"  - STRUCTURAL ISSUE: {issue}")
        for issue in info["behavioural_issues"]:
            lines.append(f"  - BEHAVIOURAL ISSUE: {issue}")
        for warning in info["consistency_warnings"]:
            lines.append(f"  - CONSISTENCY WARNING: {warning}")
        for warning in info["warnings"]:
            lines.append(f"  - WARNING: {warning}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    raw = Path(args.raw_sessions_dir)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([p for p in raw.iterdir() if p.is_dir()]) if raw.exists() else []

    session_summary: Dict[str, dict] = {}
    participants = set()
    global_checks: List[str] = []
    has_fail = False
    pass_sessions = 0
    warn_sessions = 0
    fail_sessions = 0

    for sdir in session_dirs:
        info, pid = check_session(sdir, args)
        if pid:
            participants.add(pid)

        session_summary[sdir.name] = info

        session_has_fail = bool(info["structural_issues"] or info["behavioural_issues"])
        session_has_warn = bool(info["consistency_warnings"] or info["warnings"])

        if session_has_fail:
            fail_sessions += 1
            has_fail = True
        elif session_has_warn:
            warn_sessions += 1
        else:
            pass_sessions += 1

    sessions_count = len(session_dirs)
    participants_count = len(participants)

    if sessions_count < args.min_sessions:
        global_checks.append(f"FAIL: sessions {sessions_count} < {args.min_sessions}")
        has_fail = True
    else:
        global_checks.append(f"PASS: sessions {sessions_count} >= {args.min_sessions}")

    if participants_count < args.min_participants:
        global_checks.append(f"FAIL: participants {participants_count} < {args.min_participants}")
        has_fail = True
    else:
        global_checks.append(f"PASS: participants {participants_count} >= {args.min_participants}")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "FAIL" if has_fail else ("WARN" if warn_sessions > 0 else "PASS"),
        "sessions_scanned": sessions_count,
        "participants_found": participants_count,
        "pass_sessions": pass_sessions,
        "warn_sessions": warn_sessions,
        "fail_sessions": fail_sessions,
        "global_checks": global_checks,
        "sessions": session_summary,
    }

    out_json = reports / args.out_json
    out_md = reports / args.out_md

    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    out_md.write_text(render_md(summary), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Verdict: {summary['verdict']}")

    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
