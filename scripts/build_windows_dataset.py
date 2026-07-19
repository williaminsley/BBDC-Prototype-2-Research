#!/usr/bin/env python3
"""
build_windows_dataset.py

Builds the P2 windows dataset from canonical raw events — window slicing AND feature
computation, since P2 has no browser-side windowing to port from P1.

v2: deliberately maximalist. Per project decision, compute as many features as are
cheaply derivable now; filtering/curation happens downstream in
build_behavioural_dataset.py, not here. "Log everything, decide later" applied to
feature computation, not just raw event logging.

Statistic suite per metric (ported from P1's build_windows_dataset.py, with three
deliberate adaptations noted below):
    mean, std, median, iqr, p95, max, min, n,
    cv, burstiness, local_inconsistency, early_late_diff, slope

Adaptations vs. P1's version:
  1. std uses ddof=0 (population std, matches P1's np.std(arr, ddof=0) exactly) —
     P1's own summary_stats() was checked directly for this, not assumed.
  2. cv and burstiness both divide by mean, and are set to NaN when the metric is
     zero-centered (mean ~ 0) rather than returning a numerically unstable ratio.
     P1 never needed this guard because dwell/press-to-press/speed are all strictly
     positive; P2's motion axes (ax/ay/az) and alpha_sin/alpha_cos are not.
  3. slope regresses against real elapsed time (tRelMs) for continuous/throttled
     signals (motion, orientation, scroll), not sample index — P1 only had discrete
     typing/tap events where index-based slope and time-based slope barely differ;
     P2 mixes discrete and near-continuous streams in the same pipeline, and a window
     with a dropout gap would give a misleading index-based slope.

Design decisions carried from earlier discussion (see repo_map.md / memory):
- Continuous sliding windows, whole-session span, no task-boundary respecting.
- deviceorientation.alpha only ever handled via sin/cos, never raw degrees.
- devicemotion/deviceorientation dropout tracked as an explicit per-window coverage
  percentage, not silently computed over mostly-missing data.
- Gesture-kind feature columns are only generated for payload fields that kind
  actually carries (fixes a real bug from the previous version, where distance/duration
  columns were generated uniformly for all 5 gesture kinds regardless of whether the
  underlying payload had those fields — confirmed against real data which kinds carry what).
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
DROPOUT_GAP_MS = 250

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

GESTURE_KIND_FIELDS = {
    "gesture_drag_end": {"numeric": ["payload_distancePx", "payload_durationMs"], "bool": []},
    "pot_drag_release": {"numeric": ["payload_durationMs"], "bool": ["payload_correctRelease"]},
    "approval_swipe_release": {"numeric": ["payload_durationMs", "payload_swipeRatio"], "bool": ["payload_approved"]},
    "card_swipe_summary": {"numeric": [], "bool": ["payload_swiped", "payload_targetCardSelected"]},
    "pot_transfer_confirmed": {"numeric": [], "bool": []},
}
GESTURE_PREFIX = {
    "gesture_drag_end": "drag", "pot_drag_release": "pot_drag", "approval_swipe_release": "approval_swipe",
    "card_swipe_summary": "swipe", "pot_transfer_confirmed": "pot_transfer",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build P2 windows dataset (maximalist feature set).")
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


def _valid_arrays(values: pd.Series, times: Optional[pd.Series]):
    v = pd.to_numeric(values, errors="coerce")
    mask = v.notna() & np.isfinite(v)
    if times is not None:
        t = pd.to_numeric(times, errors="coerce")
        mask = mask & t.notna()
        return v[mask].to_numpy(dtype=float), t[mask].to_numpy(dtype=float)
    return v[mask].to_numpy(dtype=float), None


def full_stats(values: pd.Series, prefix: str, times: Optional[pd.Series] = None, zero_centered: bool = False) -> dict:
    """The full P1-style stat suite for one metric: shape stats + within-window dynamics."""
    arr, t = _valid_arrays(values, times)
    n = arr.size

    if n == 0:
        keys = ["mean", "std", "median", "iqr", "p95", "max", "min", "n",
                "cv", "burstiness", "local_inconsistency", "early_late_diff", "slope"]
        return {f"{prefix}_{k}": (0 if k == "n" else np.nan) for k in keys}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    out = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if n >= 2 else np.nan,
        f"{prefix}_p95": float(np.percentile(arr, 95)) if n >= 2 else np.nan,
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_n": int(n),
    }

    if zero_centered or mean == 0 or not np.isfinite(mean):
        out[f"{prefix}_cv"] = np.nan
        out[f"{prefix}_burstiness"] = np.nan
    else:
        out[f"{prefix}_cv"] = std / mean
        denom = std + mean
        out[f"{prefix}_burstiness"] = (std - mean) / denom if denom != 0 else np.nan

    if n >= 2:
        out[f"{prefix}_local_inconsistency"] = float(np.mean(np.abs(np.diff(arr))))
        mid = max(1, n // 2)
        early, late = arr[:mid], arr[mid:]
        out[f"{prefix}_early_late_diff"] = float(np.mean(late) - np.mean(early)) if len(early) and len(late) else np.nan

        if t is not None and n >= 2 and not np.allclose(t, t[0]):
            out[f"{prefix}_slope"] = float(np.polyfit(t, arr, 1)[0]) if not np.allclose(arr, arr[0]) else 0.0
        elif n >= 2:
            x = np.arange(n, dtype=float)
            out[f"{prefix}_slope"] = float(np.polyfit(x, arr, 1)[0]) if not np.allclose(arr, arr[0]) else 0.0
        else:
            out[f"{prefix}_slope"] = np.nan
    else:
        out[f"{prefix}_local_inconsistency"] = np.nan
        out[f"{prefix}_early_late_diff"] = np.nan
        out[f"{prefix}_slope"] = np.nan

    return out


def pct_true(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(100.0 * vals.mean()) if len(vals) else np.nan


def pct_match(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return np.nan
    vals = series.dropna().astype(str)
    return float(100.0 * (vals == value).mean()) if len(vals) else np.nan


def mean_abs_jerk(t_ms: np.ndarray, magnitude: np.ndarray) -> float:
    if len(magnitude) < 2:
        return np.nan
    dt_s = np.diff(t_ms) / 1000.0
    dmag = np.diff(magnitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        jerk = np.abs(dmag / dt_s)
    jerk = jerk[np.isfinite(jerk)]
    return float(np.mean(jerk)) if jerk.size else np.nan


def path_straightness(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    path_len = float(np.sum(seg))
    displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    if displacement == 0:
        return np.nan
    return path_len / displacement


def coverage_pct(t_ms: pd.Series, window_start: float, window_end: float, gap_ms: int = DROPOUT_GAP_MS) -> float:
    window_dur = window_end - window_start
    if window_dur <= 0 or len(t_ms) == 0:
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
    return round(max(0.0, window_dur - lost) / window_dur, 4)


def pair_by_nearest_preceding(end_events: pd.Series, start_events: pd.Series) -> list[float]:
    starts = sorted(start_events.tolist())
    used = [False] * len(starts)
    durations = []
    for end_t in sorted(end_events.tolist()):
        best_i, best_dt = None, None
        for i, s_t in enumerate(starts):
            if used[i] or s_t > end_t:
                continue
            dt = end_t - s_t
            if best_dt is None or dt < best_dt:
                best_dt, best_i = dt, i
        if best_i is not None:
            used[best_i] = True
            durations.append(best_dt)
    return durations


def typing_features(win_events: pd.DataFrame) -> dict:
    kd = win_events[win_events["kind"] == "keydown"].sort_values("tRelMs")
    ku = win_events[win_events["kind"] == "keyup"].sort_values("tRelMs")
    inp = win_events[win_events["kind"] == "input"]

    out = {"typing_n_keydown": int(len(kd)), "typing_n_keyup": int(len(ku)), "typing_n_input": int(len(inp))}
    out.update(full_stats(kd["tRelMs"].diff(), "typing_press_to_press_ms", times=kd["tRelMs"]))

    dwell_vals = pair_by_nearest_preceding(ku["tRelMs"], kd["tRelMs"])
    out.update(full_stats(pd.Series(dwell_vals), "typing_dwell_ms"))

    if "payload_keyClass" in kd.columns:
        out["typing_pct_letter"] = pct_match(kd["payload_keyClass"], "LETTER")
        out["typing_pct_backspace"] = pct_match(kd["payload_keyClass"], "BACKSPACE")
        out["typing_pct_space"] = pct_match(kd["payload_keyClass"], "SPACE")

    if "payload_valueLength" in inp.columns:
        out.update(full_stats(inp["payload_valueLength"], "typing_value_length"))
    if "payload_deltaLength" in inp.columns:
        out.update(full_stats(inp["payload_deltaLength"], "typing_delta_length"))

    return out


def touch_features(win_events: pd.DataFrame) -> dict:
    ts = win_events[win_events["kind"] == "touchstart"].sort_values("tRelMs")
    tm = win_events[win_events["kind"] == "touchmove"].sort_values("tRelMs")
    te = win_events[win_events["kind"] == "touchend"].sort_values("tRelMs")

    out = {"touch_n_touchstart": int(len(ts)), "touch_n_touchmove": int(len(tm)), "touch_n_touchend": int(len(te))}

    hold_vals = pair_by_nearest_preceding(te["tRelMs"], ts["tRelMs"])
    out.update(full_stats(pd.Series(hold_vals), "touch_hold_ms"))

    if len(tm) >= 2 and {"payload_x", "payload_y"}.issubset(tm.columns):
        x = pd.to_numeric(tm["payload_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(tm["payload_y"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(tm["tRelMs"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x, y, t = x[valid], y[valid], t[valid]
        if len(x) >= 2:
            dt = np.diff(t) / 1000.0
            dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = dist / dt
            out.update(full_stats(pd.Series(speed), "touch_speed", times=pd.Series(t[1:])))
            out["touch_mean_abs_jerk"] = mean_abs_jerk(t, np.sqrt(x**2 + y**2))
            out["touch_straightness_ratio"] = path_straightness(x, y)
        else:
            out.update(full_stats(pd.Series(dtype=float), "touch_speed"))
    else:
        out.update(full_stats(pd.Series(dtype=float), "touch_speed"))

    for col, prefix in [("payload_radiusX", "touch_radiusX"), ("payload_radiusY", "touch_radiusY"),
                         ("payload_force", "touch_force"), ("payload_touchesCount", "touch_touchesCount")]:
        if col in win_events.columns:
            src = ts if col == "payload_force" else tm
            if col in src.columns:
                out.update(full_stats(src[col], prefix))

    return out


def pointer_features(win_events: pd.DataFrame) -> dict:
    pd_ = win_events[win_events["kind"] == "pointerdown"].sort_values("tRelMs")
    pm = win_events[win_events["kind"] == "pointermove"].sort_values("tRelMs")
    pu = win_events[win_events["kind"] == "pointerup"].sort_values("tRelMs")

    out = {"pointer_n_pointerdown": int(len(pd_)), "pointer_n_pointermove": int(len(pm)), "pointer_n_pointerup": int(len(pu))}

    hold_vals = pair_by_nearest_preceding(pu["tRelMs"], pd_["tRelMs"])
    out.update(full_stats(pd.Series(hold_vals), "pointer_hold_ms"))

    if len(pm) >= 2 and {"payload_x", "payload_y"}.issubset(pm.columns):
        x = pd.to_numeric(pm["payload_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(pm["payload_y"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(pm["tRelMs"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x, y, t = x[valid], y[valid], t[valid]
        if len(x) >= 2:
            dt = np.diff(t) / 1000.0
            dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = dist / dt
            out.update(full_stats(pd.Series(speed), "pointer_speed", times=pd.Series(t[1:])))
            out["pointer_mean_abs_jerk"] = mean_abs_jerk(t, np.sqrt(x**2 + y**2))
            out["pointer_straightness_ratio"] = path_straightness(x, y)
        else:
            out.update(full_stats(pd.Series(dtype=float), "pointer_speed"))
    else:
        out.update(full_stats(pd.Series(dtype=float), "pointer_speed"))

    for col, prefix in [("payload_pressure", "pointer_pressure"), ("payload_width", "pointer_width"),
                         ("payload_height", "pointer_height"), ("payload_tiltX", "pointer_tiltX"),
                         ("payload_tiltY", "pointer_tiltY")]:
        if col in pm.columns:
            out.update(full_stats(pm[col], prefix))

    return out


def scroll_features(win_events: pd.DataFrame) -> dict:
    scroll = win_events[win_events["kind"] == "scroll"].sort_values("tRelMs")
    out = {"scroll_n_events": int(len(scroll))}
    if len(scroll) >= 2 and "payload_scrollTop" in scroll.columns:
        t = pd.to_numeric(scroll["tRelMs"], errors="coerce")
        top = pd.to_numeric(scroll["payload_scrollTop"], errors="coerce")
        dt = (t.diff() / 1000.0).replace(0, np.nan)
        dtop = top.diff()
        velocity = (dtop / dt).replace([np.inf, -np.inf], np.nan)
        out.update(full_stats(velocity, "scroll_velocity", times=t))
        direction = np.sign(dtop.dropna())
        out["scroll_direction_changes"] = int((direction.diff().fillna(0) != 0).sum())
    else:
        out.update(full_stats(pd.Series(dtype=float), "scroll_velocity"))
        out["scroll_direction_changes"] = 0
    return out


def motion_features(win_events: pd.DataFrame, window_start: float, window_end: float) -> dict:
    motion = win_events[win_events["kind"] == "devicemotion"].sort_values("tRelMs")
    out = {"motion_n_events": int(len(motion)), "motion_coverage_pct": coverage_pct(motion["tRelMs"], window_start, window_end)}

    zero_centered_axes = {"payload_ax", "payload_ay", "payload_az", "payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"}
    axes = ["payload_ax", "payload_ay", "payload_az", "payload_agx", "payload_agy", "payload_agz",
            "payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"]
    for axis in axes:
        if axis in motion.columns:
            col_name = axis.replace("payload_", "motion_")
            out.update(full_stats(motion[axis], col_name, times=motion["tRelMs"], zero_centered=axis in zero_centered_axes))

    if {"payload_ax", "payload_ay", "payload_az"}.issubset(motion.columns) and len(motion) >= 2:
        ax = pd.to_numeric(motion["payload_ax"], errors="coerce")
        ay = pd.to_numeric(motion["payload_ay"], errors="coerce")
        az = pd.to_numeric(motion["payload_az"], errors="coerce")
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        out.update(full_stats(mag, "motion_magnitude", times=motion["tRelMs"]))
        t = pd.to_numeric(motion["tRelMs"], errors="coerce").to_numpy(dtype=float)
        out["motion_mean_abs_jerk"] = mean_abs_jerk(t, mag.to_numpy(dtype=float))

    if {"payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"}.issubset(motion.columns) and len(motion) >= 1:
        ra = pd.to_numeric(motion["payload_rotAlpha"], errors="coerce")
        rb = pd.to_numeric(motion["payload_rotBeta"], errors="coerce")
        rg = pd.to_numeric(motion["payload_rotGamma"], errors="coerce")
        rot_mag = np.sqrt(ra**2 + rb**2 + rg**2)
        out.update(full_stats(rot_mag, "motion_rotation_magnitude", times=motion["tRelMs"]))

    return out


def orientation_features(win_events: pd.DataFrame, window_start: float, window_end: float) -> dict:
    orient = win_events[win_events["kind"] == "deviceorientation"].sort_values("tRelMs")
    out = {"orientation_n_events": int(len(orient)), "orientation_coverage_pct": coverage_pct(orient["tRelMs"], window_start, window_end)}

    for axis in ["payload_beta", "payload_gamma"]:
        if axis in orient.columns:
            col_name = axis.replace("payload_", "orientation_")
            out.update(full_stats(orient[axis], col_name, times=orient["tRelMs"]))

    if "payload_alpha" in orient.columns and len(orient) > 0:
        alpha = pd.to_numeric(orient["payload_alpha"], errors="coerce")
        rad = np.radians(alpha)
        out["orientation_alpha_circular_mean_deg"] = circular_mean_deg(alpha)
        out.update(full_stats(np.sin(rad), "orientation_alpha_sin", times=orient["tRelMs"], zero_centered=True))
        out.update(full_stats(np.cos(rad), "orientation_alpha_cos", times=orient["tRelMs"], zero_centered=True))
    return out


def gesture_features(win_events: pd.DataFrame) -> dict:
    out = {}
    for kind, fields in GESTURE_KIND_FIELDS.items():
        prefix = GESTURE_PREFIX[kind]
        sub = win_events[win_events["kind"] == kind]
        out[f"{prefix}_n_events"] = int(len(sub))
        for col in fields["numeric"]:
            if col in sub.columns:
                short = col.replace("payload_", "").replace("Px", "").replace("Ms", "")
                out.update(full_stats(sub[col], f"{prefix}_{short}"))
        for col in fields["bool"]:
            if col in sub.columns:
                short = col.replace("payload_", "")
                out[f"{prefix}_pct_{short}"] = pct_true(sub[col])
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
        row.update(touch_features(win_events))
        row.update(pointer_features(win_events))
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
        row["hasGesture"] = any(row.get(f"{GESTURE_PREFIX[k]}_n_events", 0) > 0 for k in GESTURE_KIND_FIELDS)

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
        "window_ms": args.window_ms, "step_ms": args.step_ms,
        "row_count": int(len(windows_df)), "column_count": int(len(windows_df.columns)),
        "sessions": int(windows_df["sessionId"].nunique()),
        "columns": {
            col: {"dtype": str(windows_df[col].dtype), "non_null_pct": round(float(windows_df[col].notna().mean()) * 100, 2)}
            for col in windows_df.columns
        },
    }
    out_schema = out_dir / args.out_schema
    out_schema.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
