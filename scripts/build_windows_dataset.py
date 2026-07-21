#!/usr/bin/env python3
"""
build_windows_dataset.py

Builds the P2 windows dataset from canonical raw events — window slicing AND feature
computation, since P2 has no browser-side windowing to port from P1.

v3: adds cross-axis correlation, coarse frequency-domain, and movement-direction feature
families targeted at continuous authentication specifically, plus two correctness fixes.
See "v3 changes" below for the full list and the reasoning behind each addition/fix.

v2: deliberately maximalist. Per project decision, compute as many features as are
cheaply derivable now; filtering/curation happens downstream in
build_behavioural_dataset.py, not here. "Log everything, decide later" applied to
feature computation, not just raw event logging.

Statistic suite per metric (ported from P1's build_windows_dataset.py, with three
deliberate adaptations noted below):
    mean, std, median, iqr, p95, max, min, n,
    cv, burstiness, local_inconsistency, early_late_diff, slope, slope_r2

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

v3 changes:
  1. NEW — slope_r2 added alongside every slope. A slope fit through 2 points and one
     fit through 50 looked identical before; slope_r2 makes fit reliability visible
     without needing to cross-reference the separate _n column.
  2. NEW — cross-axis correlations for motion (linear accel, gravity-included accel,
     rotation rate) and orientation (beta vs gamma). Per-axis stats can't see how axes
     move *together*, which is exactly what a person's characteristic hold-angle/
     tremor coupling would show up as.
  3. NEW — coarse frequency-domain features (spectral_features) on the two
     orientation-invariant magnitude signals (linear accel magnitude, rotation
     magnitude). Targets the ~4-12Hz physiological hand-tremor band documented in
     behavioural-biometrics literature; time-domain stats alone can't see this.
     Implemented as interpolate-to-uniform-grid + Hann window + rfft (numpy only, no
     new dependency) since events are throttled, not perfectly periodic. With a 7.5s
     window this is coarse (~0.13Hz resolution) — treat as exploratory, not a precise
     spectral estimate.
  4. NEW — movement bearing features (touch/pointer): mean direction and a circular
     "bearing_consistency" (0=random headings, 1=one consistent heading). Distinct
     question from the existing path_straightness (net displacement vs path length):
     a person can move in a very consistent *direction* on a path that isn't straight,
     or vice versa.
  5. NEW — typing class-transition timing (typing_ptp_*). The schema never logs actual
     key identity (privacy design — only a coarse keyClass), so true digraph-specific
     timing (e.g. "th" vs "er", as named in P1's feature-family registry) isn't
     derivable here and this does NOT attempt it. Instead: press-to-press interval
     conditioned on (previous class -> this class), e.g. timing into/out of a
     backspace as a correction-hesitation signal, or letter-to-space as a
     word-boundary signal — real behavioural structure obtainable without key
     identity.
  6. NEW — scroll_total_extent (volume-of-activity) and scroll_mean_abs_accel,
     bringing scroll up to the same feature depth already given to touch/pointer.
  7. FIX — touch/pointer "jerk" was computed as mean_abs_jerk(t, sqrt(x^2+y^2)), i.e.
     the rate of change of *distance from the screen corner* — not a jerk in any
     physical sense (x,y are raw position, the 0th derivative; that call never
     differentiated speed at all), and not translation-invariant (the same gesture
     performed in a different part of the screen would score differently). Renamed to
     *_mean_abs_accel and now computed from the already-differentiated speed signal,
     matching how motion_features correctly derives jerk from the accelerometer
     (which is already an acceleration reading, so one more derivative is genuine
     jerk). This is a breaking column rename from prior versions — deliberate, so the
     semantic change is visible rather than silent.

Design decisions carried from earlier discussion (see repo_map.md / memory):
- Continuous sliding windows, whole-session span, no task-boundary respecting.
- deviceorientation.alpha only ever handled via sin/cos (plus a circular mean/
  consistency summary), never raw degrees.
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


def circular_summary(deg_array: np.ndarray) -> tuple[float, float]:
    """Circular mean direction (degrees) and resultant length (0=uniformly scattered
    directions, 1=perfectly consistent direction) for an array of angles in degrees.
    Used both for compass heading (orientation.alpha) and movement bearing (touch/pointer)."""
    if deg_array.size == 0:
        return np.nan, np.nan
    rad = np.radians(deg_array.astype(float))
    c, s = np.mean(np.cos(rad)), np.mean(np.sin(rad))
    mean_deg = float((np.degrees(np.arctan2(s, c)) + 360) % 360)
    resultant_length = float(np.sqrt(c**2 + s**2))
    return mean_deg, resultant_length


def circular_mean_deg(alpha_deg: pd.Series) -> float:
    if alpha_deg.empty:
        return np.nan
    mean_deg, _ = circular_summary(pd.to_numeric(alpha_deg, errors="coerce").dropna().to_numpy(dtype=float))
    return mean_deg


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
                "cv", "burstiness", "local_inconsistency", "early_late_diff", "slope", "slope_r2"]
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

        x = t if (t is not None and not np.allclose(t, t[0])) else np.arange(n, dtype=float)
        if not np.allclose(arr, arr[0]):
            coeffs = np.polyfit(x, arr, 1)
            out[f"{prefix}_slope"] = float(coeffs[0])
            if n >= 3:
                fitted = np.polyval(coeffs, x)
                ss_res = float(np.sum((arr - fitted) ** 2))
                ss_tot = float(np.sum((arr - mean) ** 2))
                out[f"{prefix}_slope_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                # A 2-point "trend" is just a line through two points — not a supported fit.
                out[f"{prefix}_slope_r2"] = np.nan
        else:
            out[f"{prefix}_slope"] = 0.0
            out[f"{prefix}_slope_r2"] = np.nan
    else:
        out[f"{prefix}_local_inconsistency"] = np.nan
        out[f"{prefix}_early_late_diff"] = np.nan
        out[f"{prefix}_slope"] = np.nan
        out[f"{prefix}_slope_r2"] = np.nan

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


def pairwise_correlations(cols: dict[str, np.ndarray], prefix: str, min_n: int = 5) -> dict:
    """Pearson correlation between every pair of same-length axis arrays already aligned
    to a common set of valid rows (caller's responsibility). NaN if either axis is constant
    or there are too few points for a stable estimate."""
    out = {}
    names = list(cols.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = cols[a_name], cols[b_name]
            key = f"{prefix}_corr_{a_name}_{b_name}"
            if a.size < min_n or np.std(a) == 0 or np.std(b) == 0:
                out[key] = np.nan
                continue
            with np.errstate(invalid="ignore"):
                r = np.corrcoef(a, b)[0, 1]
            out[key] = float(r) if np.isfinite(r) else np.nan
    return out


def spectral_features(t_ms: np.ndarray, values: np.ndarray, prefix: str,
                       sample_rate_hz: float = 20.0, min_n: int = 16) -> dict:
    """Coarse frequency-domain summary for a near-continuous signal (motion magnitude).
    Resamples to a uniform grid (linear interpolation) since events are throttled, not
    perfectly periodic, then takes an FFT of the mean-removed, Hann-windowed signal.
    Targets the ~4-12Hz physiological hand-tremor band noted in behavioural-biometrics
    literature; with a 7.5s window this gives ~0.13Hz resolution — coarse, exploratory,
    not a precise spectral estimate. NaN below min_n or when coverage is too sparse to
    interpolate meaningfully."""
    keys = ["dominant_freq_hz", "dominant_power_frac", "tremor_band_energy_frac", "spectral_entropy"]
    empty = {f"{prefix}_{k}": np.nan for k in keys}

    if t_ms.size < min_n or values.size < min_n:
        return empty

    order = np.argsort(t_ms)
    t_sorted, v_sorted = t_ms[order], values[order]
    span_s = (t_sorted[-1] - t_sorted[0]) / 1000.0
    if span_s <= 0:
        return empty

    n_samples = max(min_n, int(span_s * sample_rate_hz))
    grid_t = np.linspace(t_sorted[0], t_sorted[-1], n_samples)
    grid_v = np.interp(grid_t, t_sorted, v_sorted)

    grid_v = grid_v - np.mean(grid_v)
    window = np.hanning(n_samples)
    spectrum = np.fft.rfft(grid_v * window)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    power = np.abs(spectrum) ** 2

    if power.size <= 1 or np.sum(power[1:]) <= 0:
        return empty

    power_ac = power[1:]   # drop the DC bin (index 0) — not a frequency, just the mean offset
    freqs_ac = freqs[1:]
    total_power = float(np.sum(power_ac))

    dominant_idx = int(np.argmax(power_ac))
    tremor_mask = (freqs_ac >= 3.0) & (freqs_ac <= 12.0)

    probs = power_ac / total_power
    spectral_entropy = float(-np.sum(probs * np.log2(probs + 1e-15)) / np.log2(len(probs)))

    return {
        f"{prefix}_dominant_freq_hz": float(freqs_ac[dominant_idx]),
        f"{prefix}_dominant_power_frac": float(power_ac[dominant_idx] / total_power),
        f"{prefix}_tremor_band_energy_frac": float(np.sum(power_ac[tremor_mask]) / total_power) if tremor_mask.any() else 0.0,
        f"{prefix}_spectral_entropy": spectral_entropy,
    }


def movement_bearing_features(x: np.ndarray, y: np.ndarray, prefix: str) -> dict:
    """Direction (bearing) of movement between consecutive points — complements
    path_straightness (net displacement vs path length) with a distinct question:
    does the movement keep a consistent heading, regardless of how direct the path is?"""
    if x.size < 2:
        return {f"{prefix}_bearing_mean_deg": np.nan, f"{prefix}_bearing_consistency": np.nan}
    bearings = (np.degrees(np.arctan2(np.diff(y), np.diff(x))) + 360) % 360
    mean_deg, resultant_length = circular_summary(bearings)
    return {f"{prefix}_bearing_mean_deg": mean_deg, f"{prefix}_bearing_consistency": resultant_length}


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

        # Class-transition timing: the schema never logs actual key identity (privacy
        # design — only a coarse class), so true digraph-specific timing (e.g. "th" vs
        # "er") isn't derivable here. This is the closest privacy-safe substitute:
        # press-to-press interval conditioned on the (previous class -> this class)
        # transition, which still captures behaviourally meaningful structure — e.g.
        # hesitation before/after a correction — without needing letter identity.
        classes = kd["payload_keyClass"].to_numpy()
        ptp = kd["tRelMs"].diff().to_numpy()
        transitions = {
            "letter_to_letter": (classes[:-1] == "LETTER") & (classes[1:] == "LETTER"),
            "to_backspace": classes[1:] == "BACKSPACE",
            "from_backspace": classes[:-1] == "BACKSPACE",
            "letter_to_space": (classes[:-1] == "LETTER") & (classes[1:] == "SPACE"),
        }
        for name, mask in transitions.items():
            vals = ptp[1:][mask]
            vals = vals[np.isfinite(vals)]
            out[f"typing_ptp_{name}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            out[f"typing_ptp_{name}_n"] = int(vals.size)

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
            # NOTE: this replaces a previous version that computed
            # mean_abs_jerk(t, sqrt(x^2+y^2)) — the rate of change of distance from the
            # screen corner, which is neither translation-invariant nor a true jerk
            # (position is the 0th derivative; that computation never differentiated
            # speed at all). Jerk of *movement* requires differentiating speed, matching
            # how motion_features derives jerk from the already-differentiated
            # accelerometer signal.
            out["touch_mean_abs_accel"] = mean_abs_jerk(t[1:], speed)
            out["touch_straightness_ratio"] = path_straightness(x, y)
            out.update(movement_bearing_features(x, y, "touch"))
        else:
            out.update(full_stats(pd.Series(dtype=float), "touch_speed"))
            out.update(movement_bearing_features(np.array([]), np.array([]), "touch"))
    else:
        out.update(full_stats(pd.Series(dtype=float), "touch_speed"))
        out.update(movement_bearing_features(np.array([]), np.array([]), "touch"))

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
            out["pointer_mean_abs_accel"] = mean_abs_jerk(t[1:], speed)
            out["pointer_straightness_ratio"] = path_straightness(x, y)
            out.update(movement_bearing_features(x, y, "pointer"))
        else:
            out.update(full_stats(pd.Series(dtype=float), "pointer_speed"))
            out.update(movement_bearing_features(np.array([]), np.array([]), "pointer"))
    else:
        out.update(full_stats(pd.Series(dtype=float), "pointer_speed"))
        out.update(movement_bearing_features(np.array([]), np.array([]), "pointer"))

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
        out["scroll_total_extent"] = float(dtop.abs().sum(skipna=True))
        vel_arr, vel_t = _valid_arrays(velocity, t)
        out["scroll_mean_abs_accel"] = mean_abs_jerk(vel_t, vel_arr) if vel_arr.size >= 2 else np.nan
        direction = np.sign(dtop.dropna())
        out["scroll_direction_changes"] = int((direction.diff().fillna(0) != 0).sum())
    else:
        out.update(full_stats(pd.Series(dtype=float), "scroll_velocity"))
        out["scroll_total_extent"] = 0.0
        out["scroll_mean_abs_accel"] = np.nan
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

    t_all = pd.to_numeric(motion["tRelMs"], errors="coerce").to_numpy(dtype=float)

    # Cross-axis correlations: per-axis stats can't see how axes move *together*, which is
    # exactly what a characteristic hold-angle/tremor coupling would show up as. Computed
    # on rows valid across all three axes of each group so the arrays are aligned.
    for group_name, cols in [
        ("motion_lin", ["payload_ax", "payload_ay", "payload_az"]),
        ("motion_grav", ["payload_agx", "payload_agy", "payload_agz"]),
        ("motion_rot", ["payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"]),
    ]:
        if set(cols).issubset(motion.columns):
            sub = motion[cols].apply(pd.to_numeric, errors="coerce")
            valid = sub.notna().all(axis=1) & np.isfinite(sub).all(axis=1)
            aligned = sub[valid]
            axis_arrays = {c.replace("payload_", ""): aligned[c].to_numpy(dtype=float) for c in cols}
            out.update(pairwise_correlations(axis_arrays, group_name))

    if {"payload_ax", "payload_ay", "payload_az"}.issubset(motion.columns) and len(motion) >= 2:
        ax = pd.to_numeric(motion["payload_ax"], errors="coerce")
        ay = pd.to_numeric(motion["payload_ay"], errors="coerce")
        az = pd.to_numeric(motion["payload_az"], errors="coerce")
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        out.update(full_stats(mag, "motion_magnitude", times=motion["tRelMs"]))
        t = pd.to_numeric(motion["tRelMs"], errors="coerce").to_numpy(dtype=float)
        out["motion_mean_abs_jerk"] = mean_abs_jerk(t, mag.to_numpy(dtype=float))
        out.update(spectral_features(t, mag.to_numpy(dtype=float), "motion_magnitude"))

    if {"payload_rotAlpha", "payload_rotBeta", "payload_rotGamma"}.issubset(motion.columns) and len(motion) >= 1:
        ra = pd.to_numeric(motion["payload_rotAlpha"], errors="coerce")
        rb = pd.to_numeric(motion["payload_rotBeta"], errors="coerce")
        rg = pd.to_numeric(motion["payload_rotGamma"], errors="coerce")
        rot_mag = np.sqrt(ra**2 + rb**2 + rg**2)
        out.update(full_stats(rot_mag, "motion_rotation_magnitude", times=motion["tRelMs"]))
        out.update(spectral_features(t_all, rot_mag.to_numpy(dtype=float), "motion_rotation_magnitude"))

    return out


def orientation_features(win_events: pd.DataFrame, window_start: float, window_end: float) -> dict:
    orient = win_events[win_events["kind"] == "deviceorientation"].sort_values("tRelMs")
    out = {"orientation_n_events": int(len(orient)), "orientation_coverage_pct": coverage_pct(orient["tRelMs"], window_start, window_end)}

    for axis in ["payload_beta", "payload_gamma"]:
        if axis in orient.columns:
            col_name = axis.replace("payload_", "orientation_")
            out.update(full_stats(orient[axis], col_name, times=orient["tRelMs"]))

    if {"payload_beta", "payload_gamma"}.issubset(orient.columns):
        sub = orient[["payload_beta", "payload_gamma"]].apply(pd.to_numeric, errors="coerce")
        valid = sub.notna().all(axis=1) & np.isfinite(sub).all(axis=1)
        aligned = sub[valid]
        out.update(pairwise_correlations(
            {"beta": aligned["payload_beta"].to_numpy(dtype=float), "gamma": aligned["payload_gamma"].to_numpy(dtype=float)},
            "orientation",
        ))

    if "payload_alpha" in orient.columns and len(orient) > 0:
        alpha = pd.to_numeric(orient["payload_alpha"], errors="coerce")
        rad = np.radians(alpha)
        mean_deg, resultant_length = circular_summary(alpha.dropna().to_numpy(dtype=float))
        out["orientation_alpha_circular_mean_deg"] = mean_deg
        out["orientation_alpha_consistency"] = resultant_length
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
