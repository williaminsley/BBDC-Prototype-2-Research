#!/usr/bin/env python3
"""
build_windows_dataset.py

Builds the P2 windows dataset from canonical raw events. This is the genuinely new
pipeline stage P1 never needed — P1's version only concatenated windows the browser
had already computed. P2 has no browser-side windowing at all (confirmed: memory note
+ repo_map.md), so window slicing AND feature computation both happen here for the
first time.

Design decisions this script implements (settled through discussion, not guessed):
- Continuous sliding windows (7.5s window, 2.5s step, from schema.js's
  recommendedWindowMs/recommendedStepMs), matching P1's precedent of pure continuous
  sliding with no task-boundary respecting — a deployed CA system never gets clean
  task boundaries either, so training on task-pure windows would be a train/deploy
  mismatch.
- Whole-session span (first event to last), not anchored to task_start — preserves
  the pre-task consent/permission-negotiation phase, where devicemotion/deviceorientation
  are already flowing (confirmed in real data).
- Task purity as a continuous soft metric per window (dominant task's share of events),
  not a hard windowing constraint — feeds confidence-weighted fusion later rather than
  discarding or force-labelling straddling windows.
- deviceorientation.alpha is NEVER averaged as raw degrees — always via sin/cos
  components (confirmed necessary: 194 genuine wraps found across 3 real sessions).
- devicemotion/deviceorientation dropout is treated as "modality inactive this window",
  the same mechanism as task-driven inactivity (hasTyping/hasTapping-style flags) —
  confirmed real and substantial in practice (16-29% of session time across all 3
  real sessions), so every window gets an explicit coverage percentage, not a silent
  computation over mostly-missing data.
- Feature families and exact payload field names below were confirmed directly against
  real raw_events data, not assumed from schema.js alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WINDOW_MS = 7_500
STEP_MS = 2_500
DROPOUT_GAP_MS = 250  # 5x the 50ms motion/orientation throttle — matches the audit notebook's threshold

# Base task ID -> coarse activity type, taken directly from tasks.js's own task list.
# Confirmed against real data (§10 of 00_data_audit.ipynb): task-driven modality
# availability is real and substantial, and activity-segmented comparison surfaces
# signal that whole-session comparison dilutes.
BASE_TASK_TYPE = {
    "unlock_code": "tap", "home_balance_check": "tap", "home_explore_cards": "scroll",
    "activity_search": "type", "activity_filter_review": "tap", "activity_scroll_select": "scroll",
    "transaction_category": "tap", "transaction_note": "type", "pots_drag_amount": "drag_swipe",
    "pots_transfer": "tap", "insights_swipe_cards": "drag_swipe", "insights_review": "tap",
    "secure_approval": "drag_swipe", "secure_reply": "type", "finish_feeling": "tap",
}

SESSION_LEVEL_COLS = [
    "sessionId", "participantId", "sessionIndex", "identitySource", "schemaVersion",
    "appVersion", "deviceFamily", "devicePlatform", "deviceModel", "consentVersion",
    "completedNormally", "usableForSignalExtraction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build P2 windows dataset (slicing + feature computation) from canonical raw events.")
    p.add_argument("--in-parquet", type=str, default="data/processed/raw_events.parquet")
    p.add_argument("--in-csv", type=str, default="data/processed/raw_events.csv")
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--out-parquet", type=str, default="windows.parquet")
    p.add_argument("--out-csv", type=str, default="windows.csv")
    p.add_argument("--out-schema", type=str, default="windows_schema.json")
    p.add_argument("--window-ms", type=int, default=WINDOW_MS)
    p.add_argument("--step-ms", type=int, default=STEP_MS)
    p.add_argument("--skip-csv", action="store_true")
    return p.parse_args()


def load_raw(in_parquet: Path, in_csv: Path) -> pd.DataFrame:
    if in_parquet.exists():
        return pd.read_parquet(in_parquet)
    if in_csv.exists():
        return pd.read_csv(in_csv, low_memory=False)
    raise FileNotFoundError(f"Could not find raw events at {in_parquet} or {in_csv}")


def base_task_id(task_id) -> Optional[str]:
    if pd.isna(task_id):
        return None
    parts = str(task_id).rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id


def generate_windows(start_ms: float, end_ms: float, window_ms: int, step_ms: int) -> list[dict]:
    """Continuous sliding windows, whole-session span. Deliberately does NOT respect
    task boundaries — see module docstring for why."""
    windows = []
    idx = 0
    t = start_ms
    while t + window_ms <= end_ms:
        windows.append({"windowIndex": idx, "windowStartMs": float(t), "windowEndMs": float(t + window_ms)})
        idx += 1
        t += step_ms
    return windows


def circular_mean_deg(alpha_deg: pd.Series) -> float:
    if alpha_deg.empty:
        return np.nan
    rad = np.radians(alpha_deg.astype(float))
    return float((np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360) % 360)


def safe_stats(series: pd.Series, prefix: str) -> dict:
    """mean/std/median/iqr for a numeric series, NaN-safe, empty-safe."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_median": np.nan, f"{prefix}_iqr": np.nan}
    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std()) if len(s) > 1 else np.nan,
        f"{prefix}_median": float(s.median()),
        f"{prefix}_iqr": float(s.quantile(0.75) - s.quantile(0.25)),
    }


def mean_abs_jerk(t_ms: pd.Series, magnitude: pd.Series) -> float:
    """Mean absolute rate of change of a magnitude signal — same concept as P1's
    mean_abs_jerk for pointer trajectories, applied here to motion magnitude."""
    if len(magnitude) < 2:
        return np.nan
    dt_s = t_ms.diff() / 1000.0
    dmag = magnitude.diff()
    jerk = (dmag / dt_s).replace([np.inf, -np.inf], np.nan).abs()
    return float(jerk.mean()) if jerk.notna().any() else np.nan


def coverage_pct(t_ms: pd.Series, window_start: float, window_end: float, gap_ms: int = DROPOUT_GAP_MS) -> float:
    """Fraction of the window's duration NOT lost to dropout gaps. An empty modality
    in this window gets 0.0, not NaN, so downstream fusion can treat it as zero-weight
    rather than propagate missingness."""
    window_dur = window_end - window_start
    if window_dur <= 0:
        return 0.0
    if len(t_ms) == 0:
        return 0.0
    t = t_ms.sort_values().values
    lost = 0.0
    prev = window_start
    for ts in t:
        gap = ts - prev
        if gap > gap_ms:
            lost += gap
        prev = ts
    tail_gap = window_end - prev
    if tail_gap > gap_ms:
        lost += tail_gap
    covered = max(0.0, window_dur - lost)
    return round(covered / window_dur, 4)


def typing_features(win_events: pd.DataFrame) -> dict:
    kd = win_events[win_events["kind"] == "keydown"].sort_values("tRelMs")
    ku = win_events[win_events["kind"] == "keyup"].sort_values("tRelMs")
    out = {"typing_n_keydown": int(len(kd)), "typing_n_keyup": int(len(ku))}
    out.update(safe_stats(kd["tRelMs"].diff(), "typing_press_to_press_ms"))

    # Pair each keyup with the nearest preceding unmatched keydown for dwell time.
    dwell_vals = []
    kd_times = kd["tRelMs"].tolist()
    used = [False] * len(kd_times)
    for ku_t in ku["tRelMs"].tolist():
        best_i, best_dt = None, None
        for i, kd_t in enumerate(kd_times):
            if used[i] or kd_t > ku_t:
                continue
            dt = ku_t - kd_t
            if best_dt is None or dt < best_dt:
                best_dt, best_i = dt, i
        if best_i is not None:
            used[best_i] = True
            dwell_vals.append(best_dt)
    out.update(safe_stats(pd.Series(dwell_vals), "typing_dwell_ms"))
    return out


def touch_pointer_features(win_events: pd.DataFrame) -> dict:
    out = {}
    touchmove = win_events[win_events["kind"] == "touchmove"].sort_values("tRelMs")
    pointermove = win_events[win_events["kind"] == "pointermove"].sort_values("tRelMs")

    out["touch_n_touchstart"] = int((win_events["kind"] == "touchstart").sum())
    out["touch_n_touchmove"] = int(len(touchmove))
    out["pointer_n_pointerdown"] = int((win_events["kind"] == "pointerdown").sum())
    out["pointer_n_pointermove"] = int(len(pointermove))

    def speed_stats(move_df: pd.DataFrame, prefix: str) -> dict:
        if len(move_df) < 2:
            return safe_stats(pd.Series(dtype=float), f"{prefix}_speed")
        dx = move_df["payload_x"].astype(float).diff()
        dy = move_df["payload_y"].astype(float).diff()
        dt = (move_df["tRelMs"].diff() / 1000.0).replace(0, np.nan)
        dist = np.sqrt(dx**2 + dy**2)
        speed = dist / dt
        return safe_stats(speed.replace([np.inf, -np.inf], np.nan), f"{prefix}_speed")

    if {"payload_x", "payload_y"}.issubset(touchmove.columns):
        out.update(speed_stats(touchmove, "touch"))
    if {"payload_x", "payload_y"}.issubset(pointermove.columns):
        out.update(speed_stats(pointermove, "pointer"))

    if "payload_pressure" in pointermove.columns:
        out.update(safe_stats(pointermove["payload_pressure"], "pointer_pressure"))
    if "payload_force" in win_events.columns:
        # Kept for completeness only — P1 explicitly excluded force-derived features
        # from its baseline behavioural set (tap_force_ / tap_touch_ prefixes in
        # build_behavioural_dataset.py's DROP_PREFIXES); not expected to be useful.
        out.update(safe_stats(win_events.loc[win_events["kind"] == "touchstart", "payload_force"], "touch_force"))

    return out


def scroll_features(win_events: pd.DataFrame) -> dict:
    scroll = win_events[win_events["kind"] == "scroll"].sort_values("tRelMs")
    out = {"scroll_n_events": int(len(scroll))}
    if len(scroll) >= 2 and "payload_scrollTop" in scroll.columns:
        dt = (scroll["tRelMs"].diff() / 1000.0).replace(0, np.nan)
        dtop = scroll["payload_scrollTop"].astype(float).diff()
        velocity = (dtop / dt).replace([np.inf, -np.inf], np.nan)
        out.update(safe_stats(velocity, "scroll_velocity"))
    else:
        out.update(safe_stats(pd.Series(dtype=float), "scroll_velocity"))
    return out


def motion_features(win_events: pd.DataFrame, window_start: float, window_end: float) -> dict:
    motion = win_events[win_events["kind"] == "devicemotion"].sort_values("tRelMs")
    out = {"motion_n_events": int(len(motion))}
    out["motion_coverage_pct"] = coverage_pct(motion["tRelMs"], window_start, window_end)

    axes = ["payload_ax", "payload_ay", "payload_az", "payload_agx", "payload_agy", "payload_agz",
            "payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"]
    for axis in axes:
        col_name = axis.replace("payload_", "motion_")
        if axis in motion.columns:
            out.update(safe_stats(motion[axis], col_name))

    if {"payload_ax", "payload_ay", "payload_az"}.issubset(motion.columns):
        mag = np.sqrt(motion["payload_ax"].astype(float) ** 2 + motion["payload_ay"].astype(float) ** 2 + motion["payload_az"].astype(float) ** 2)
        out.update(safe_stats(mag, "motion_magnitude"))
        out["motion_mean_abs_jerk"] = mean_abs_jerk(motion["tRelMs"], mag)
    return out


def orientation_features(win_events: pd.DataFrame, window_start: float, window_end: float) -> dict:
    orient = win_events[win_events["kind"] == "deviceorientation"].sort_values("tRelMs")
    out = {"orientation_n_events": int(len(orient))}
    out["orientation_coverage_pct"] = coverage_pct(orient["tRelMs"], window_start, window_end)

    for axis in ["payload_beta", "payload_gamma"]:
        col_name = axis.replace("payload_", "orientation_")
        if axis in orient.columns:
            out.update(safe_stats(orient[axis], col_name))

    # alpha: NEVER averaged as raw degrees — sin/cos components instead, confirmed
    # necessary against real data (wraps are frequent and genuine, not gap artifacts).
    if "payload_alpha" in orient.columns and len(orient) > 0:
        alpha = orient["payload_alpha"].astype(float)
        rad = np.radians(alpha)
        sin_a, cos_a = np.sin(rad), np.cos(rad)
        out["orientation_alpha_circular_mean_deg"] = circular_mean_deg(alpha)
        out.update(safe_stats(sin_a, "orientation_alpha_sin"))
        out.update(safe_stats(cos_a, "orientation_alpha_cos"))
    return out


def gesture_features(win_events: pd.DataFrame) -> dict:
    """Drag/swipe/pot-transfer events are sparse and discrete (a handful per session)
    — window-level presence/count plus raw stats when present, not a full stats suite."""
    out = {}
    for kind, prefix in [
        ("gesture_drag_end", "drag"), ("card_swipe_summary", "swipe"),
        ("approval_swipe_release", "approval_swipe"), ("pot_drag_release", "pot_drag"),
        ("pot_transfer_confirmed", "pot_transfer"),
    ]:
        sub = win_events[win_events["kind"] == kind]
        out[f"{prefix}_n_events"] = int(len(sub))
        if "payload_distancePx" in sub.columns and len(sub):
            out[f"{prefix}_distance_mean"] = float(pd.to_numeric(sub["payload_distancePx"], errors="coerce").mean())
        if "payload_durationMs" in sub.columns and len(sub):
            out[f"{prefix}_duration_mean"] = float(pd.to_numeric(sub["payload_durationMs"], errors="coerce").mean())
    return out


def task_purity_features(win_events: pd.DataFrame) -> dict:
    task_events = win_events[win_events["taskId"].notna()]
    if task_events.empty:
        return {"dominantTaskId": None, "dominantBaseTaskId": None, "activityType": None,
                "taskPurity": np.nan, "n_distinct_tasks_in_window": 0}
    counts = task_events["taskId"].value_counts()
    dominant_task = counts.index[0]
    purity = float(counts.iloc[0] / len(task_events))
    base = base_task_id(dominant_task)
    return {
        "dominantTaskId": dominant_task,
        "dominantBaseTaskId": base,
        "activityType": BASE_TASK_TYPE.get(base, "other"),
        "taskPurity": round(purity, 4),
        "n_distinct_tasks_in_window": int(task_events["taskId"].nunique()),
    }


def build_session_windows(session_df: pd.DataFrame, window_ms: int, step_ms: int) -> list[dict]:
    session_df = session_df.sort_values("eventIndex")
    t_min, t_max = session_df["tRelMs"].min(), session_df["tRelMs"].max()
    if pd.isna(t_min) or pd.isna(t_max) or (t_max - t_min) < window_ms:
        return []

    session_meta = {col: session_df[col].iloc[0] for col in SESSION_LEVEL_COLS if col in session_df.columns}
    windows = generate_windows(float(t_min), float(t_max), window_ms, step_ms)

    rows = []
    for w in windows:
        mask = (session_df["tRelMs"] >= w["windowStartMs"]) & (session_df["tRelMs"] < w["windowEndMs"])
        win_events = session_df[mask]

        row = dict(session_meta)
        row.update(w)
        row["n_events_in_window"] = int(len(win_events))
        row.update(task_purity_features(win_events))
        row.update(typing_features(win_events))
        row.update(touch_pointer_features(win_events))
        row.update(scroll_features(win_events))
        row.update(motion_features(win_events, w["windowStartMs"], w["windowEndMs"]))
        row.update(orientation_features(win_events, w["windowStartMs"], w["windowEndMs"]))
        row.update(gesture_features(win_events))

        row["hasTyping"] = row["typing_n_keydown"] > 0
        row["hasTouch"] = row["touch_n_touchstart"] > 0 or row["touch_n_touchmove"] > 0
        row["hasPointer"] = row["pointer_n_pointerdown"] > 0 or row["pointer_n_pointermove"] > 0
        row["hasScroll"] = row["scroll_n_events"] > 0
        row["hasMotion"] = row["motion_coverage_pct"] > 0
        row["hasOrientation"] = row["orientation_coverage_pct"] > 0
        row["hasGesture"] = any(row.get(f"{p}_n_events", 0) > 0 for p in ["drag", "swipe", "approval_swipe", "pot_drag", "pot_transfer"])

        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    raw = load_raw(Path(args.in_parquet), Path(args.in_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sid, session_df in raw.groupby("sessionId", sort=False):
        rows = build_session_windows(session_df, args.window_ms, args.step_ms)
        print(f"{sid}: {len(rows)} windows")
        all_rows.extend(rows)

    if not all_rows:
        print("No windows produced — sessions may be shorter than one window.")
        return 1

    windows_df = pd.DataFrame(all_rows)
    windows_df = windows_df.sort_values(["sessionId", "windowIndex"]).reset_index(drop=True)

    id_cols = ["sessionId", "participantId", "sessionIndex", "windowIndex", "windowStartMs", "windowEndMs"]
    other_cols = [c for c in windows_df.columns if c not in id_cols]
    windows_df = windows_df[id_cols + other_cols]

    out_parquet = out_dir / args.out_parquet
    windows_df.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({len(windows_df)} rows, {len(windows_df.columns)} columns)")

    if not args.skip_csv:
        out_csv = out_dir / args.out_csv
        windows_df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    schema = {
        "window_ms": args.window_ms,
        "step_ms": args.step_ms,
        "row_count": int(len(windows_df)),
        "column_count": int(len(windows_df.columns)),
        "sessions": int(windows_df["sessionId"].nunique()),
        "columns": {
            col: {
                "dtype": str(windows_df[col].dtype),
                "non_null_pct": round(float(windows_df[col].notna().mean()) * 100, 2),
            }
            for col in windows_df.columns
        },
    }
    out_schema = out_dir / args.out_schema
    out_schema.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_schema}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
