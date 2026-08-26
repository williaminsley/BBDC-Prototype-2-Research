"""
scripts/primitives/coupling.py

Coupling feature family: transient accelerometer/orientation response to a
discrete motor action (tap, keystroke, gesture). Three extraction paths:
  1. Accelerometer-magnitude coupling (05_1 tap-motion, 05_3 keystroke-
     motion) -- confirmed identical windowing constants across both
     notebooks, built as ONE shared function over two anchor event kinds.
  2. Gesture-motion coupling (05_2) -- phase-normalised, reuses gesture.py's
     already-validated gesture boundaries rather than 05_2's own flawed
     classifier (same double-counting bug as reliability_check.py).
  3. Orientation coupling (05_4) -- beta/gamma response, genuinely
     different signal from accelerometer, own short local baseline
     (distinct from corrections.py's 30s causal baseline).

CORRECTION TO EARLIER MAPPING (2026-08): an earlier design pass claimed
05_4's epoch extractors should be the canonical base for the WHOLE coupling
family. Wrong -- they're hardcoded to payload_beta/payload_gamma and
cannot substitute for accelerometer-magnitude extraction (different kind:
'devicemotion' vs 'deviceorientation'). Corrected: accel and orientation
coupling are separate paths, each from its own correct source.

UNIFORM SUFFIX REBUILD (2026-08): applies the full 9-statistic suffix set
(mean/std/median/iqr/p95/max/n/cv/slope, via _stats.py) to every per-event
coupling metric, rather than the mean/median-only pairs an earlier version
used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))  # ensures _stats resolves whether this
                                                    # module is run standalone or imported
                                                    # as part of the primitives package
from _stats import compute_summary_stats, summary_stat_columns

TAP_DOWN_KIND = "touchstart"
KEY_DOWN_KIND = "keydown"
MOTION_KIND = "devicemotion"
ORIENT_KIND = "deviceorientation"
ACC_COLS = ["payload_ax", "payload_ay", "payload_az"]

PRE_MS, POST_MS = 300.0, 500.0
MIN_PRE, MIN_POST = 2, 3

PRE_FRAC = 0.3
N_PHASE_BINS = 24
PHASE_BINS = np.linspace(-PRE_FRAC, 1.5, N_PHASE_BINS)
MID_PHASE_IDX = int(np.argmin(np.abs(PHASE_BINS - 0.5)))

ORIENT_PRE_MS, ORIENT_POST_MS = 300, 500
ORIENT_MIN_PRE, ORIENT_MIN_POST = 2, 3

ACCEL_METRICS = ["peak", "impulse_auc", "latency_ms", "decay_ratio", "z_share", "snr"]
ORIENT_METRICS = ["beta_peak_dev", "beta_peak_latency_ms", "gamma_peak_dev", "gamma_peak_latency_ms"]
GESTURE_COUPLING_METRICS = ["mid_phase_energy", "peak_phase_energy"]


def _trapz_compat(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def _motion_arrays(sess: pd.DataFrame):
    mo = sess.loc[sess["kind"] == MOTION_KIND].sort_values("tRelMs")
    t = mo["tRelMs"].to_numpy(dtype=float)
    a = [pd.to_numeric(mo[c], errors="coerce").to_numpy(dtype=float) for c in ACC_COLS]
    mag = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    ok = np.isfinite(mag) & np.isfinite(t)
    return t[ok], a[0][ok], a[1][ok], a[2][ok], mag[ok]


# =============================================================================
# 1. Accelerometer-magnitude coupling -- shared for taps and keystrokes
# =============================================================================

def accel_event_coupling_features(raw_df: pd.DataFrame, event_kind: str) -> pd.DataFrame:
    rows = []
    for sid, sess in raw_df.groupby("sessionId"):
        t_m, ax_, ay_, az_, mag = _motion_arrays(sess)
        if len(t_m) < 10:
            continue
        events = sess.loc[sess["kind"] == event_kind, ["participantId", "tRelMs"]]
        if events.empty:
            continue
        pid = events["participantId"].iloc[0]
        for t0 in events["tRelMs"].to_numpy(dtype=float):
            sel = (t_m >= t0 - PRE_MS) & (t_m < t0 + POST_MS)
            if sel.sum() < 5:
                continue
            rel = t_m[sel] - t0
            v = mag[sel]
            pre_m = rel < 0
            if pre_m.sum() < MIN_PRE:
                continue
            base = v[pre_m].mean()
            noise = v[pre_m].std() + 1e-9
            post_m = rel >= 0
            if post_m.sum() < MIN_POST:
                continue
            vp = v[post_m] - base
            rp = rel[post_m]
            pk_i = int(np.argmax(vp))
            peak = float(vp[pk_i])
            lat = float(rp[pk_i])
            auc = float(_trapz_compat(np.clip(vp, 0, None), rp)) if len(rp) > 1 else np.nan
            tail = vp[rp >= lat + 150]
            decay = float(np.mean(tail) / peak) if (len(tail) and peak > 0) else np.nan
            zc = az_[sel][post_m] - az_[sel][pre_m].mean()
            xc = ax_[sel][post_m] - ax_[sel][pre_m].mean()
            yc = ay_[sel][post_m] - ay_[sel][pre_m].mean()
            tot = np.abs(xc).sum() + np.abs(yc).sum() + np.abs(zc).sum()
            zsh = float(np.abs(zc).sum() / tot) if tot > 0 else np.nan
            rows.append({
                "sessionId": sid, "participantId": pid, "t_s": t0 / 1000.0,
                "peak": peak, "impulse_auc": auc, "latency_ms": lat,
                "decay_ratio": decay, "z_share": zsh, "snr": peak / noise,
            })
    cols = ["sessionId", "participantId", "t_s"] + ACCEL_METRICS
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# =============================================================================
# 2. Gesture-motion coupling -- phase-normalised
# =============================================================================

def phase_profile(t0: float, t1: float, t_m: np.ndarray, mag: np.ndarray):
    dur = t1 - t0
    if dur <= 0:
        return None
    pre_lo = t0 - PRE_FRAC * dur
    sel = (t_m >= pre_lo) & (t_m < t0 + 1.5 * dur)
    if sel.sum() < 5:
        return None
    tt, vv = t_m[sel], mag[sel]
    phase = (tt - t0) / dur
    pre_m = phase < 0
    if pre_m.sum() < 2:
        return None
    base = vv[pre_m].mean()
    return np.interp(PHASE_BINS, phase, vv - base, left=np.nan, right=np.nan)


def gesture_coupling_features(raw_df: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sess in raw_df.groupby("sessionId"):
        t_m, _, _, _, mag = _motion_arrays(sess)
        if len(t_m) < 20:
            continue
        int_sess = interactions.loc[interactions["sessionId"] == sid]
        for r in int_sess.itertuples():
            t0_ms = r.t_s * 1000.0
            t1_ms = t0_ms + r.hold_ms
            profile = phase_profile(t0_ms, t1_ms, t_m, mag)
            if profile is None:
                continue
            mid_val = profile[MID_PHASE_IDX]
            rows.append({
                "sessionId": sid, "participantId": r.participantId, "t_s": r.t_s,
                "gesture": r.gesture,
                "mid_phase_energy": float(mid_val) if np.isfinite(mid_val) else np.nan,
                "peak_phase_energy": float(np.nanmax(profile)) if np.isfinite(profile).any() else np.nan,
            })
    cols = ["sessionId", "participantId", "t_s", "gesture"] + GESTURE_COUPLING_METRICS
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# =============================================================================
# 3. Orientation coupling -- beta/gamma response
# =============================================================================

def orient_event_coupling_features(raw_df: pd.DataFrame, event_kind: str) -> pd.DataFrame:
    rows = []
    for sid, sess in raw_df.groupby("sessionId"):
        o = sess.loc[sess["kind"] == ORIENT_KIND].sort_values("tRelMs")
        if len(o) < 10:
            continue
        o_t = o["tRelMs"].to_numpy(dtype=float)
        o_beta = pd.to_numeric(o["payload_beta"], errors="coerce").to_numpy(dtype=float)
        o_gamma = pd.to_numeric(o["payload_gamma"], errors="coerce").to_numpy(dtype=float)

        events = sess.loc[sess["kind"] == event_kind, ["participantId", "tRelMs"]]
        if events.empty:
            continue
        pid = events["participantId"].iloc[0]

        for t0 in events["tRelMs"].to_numpy(dtype=float):
            pre_mask = (o_t >= t0 - ORIENT_PRE_MS) & (o_t < t0)
            post_mask = (o_t >= t0) & (o_t <= t0 + ORIENT_POST_MS)
            if pre_mask.sum() < ORIENT_MIN_PRE or post_mask.sum() < ORIENT_MIN_POST:
                continue

            row = {"sessionId": sid, "participantId": pid, "t_s": t0 / 1000.0}
            populated = False
            for label, series in [("beta", o_beta), ("gamma", o_gamma)]:
                base = series[pre_mask]
                base = base[np.isfinite(base)]
                if len(base) < ORIENT_MIN_PRE:
                    continue
                baseline = base.mean()
                post_vals = series[post_mask] - baseline
                post_rel_t = o_t[post_mask] - t0
                ok = np.isfinite(post_vals)
                if ok.sum() < ORIENT_MIN_POST:
                    continue
                post_vals, post_rel_t = post_vals[ok], post_rel_t[ok]
                pk_i = int(np.argmax(np.abs(post_vals)))
                row[f"{label}_peak_dev"] = float(post_vals[pk_i])
                row[f"{label}_peak_latency_ms"] = float(post_rel_t[pk_i])
                populated = True
            if populated:
                rows.append(row)

    cols = ["sessionId", "participantId", "t_s"] + ORIENT_METRICS
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# =============================================================================
# Window-level aggregation
# =============================================================================

def _events_in_window(event_t_s: np.ndarray, w_start: float, w_end: float) -> np.ndarray:
    return (event_t_s >= w_start) & (event_t_s < w_end)


def _agg_block(windows, eligible_mask, events_df, value_cols, prefix):
    if events_df.empty:
        return windows
    for sid, sess_windows in windows.loc[eligible_mask].groupby("sessionId"):
        ev_sess = events_df.loc[events_df["sessionId"] == sid]
        if ev_sess.empty:
            continue
        ev_t = ev_sess["t_s"].to_numpy()
        for idx, w in sess_windows.iterrows():
            mask = _events_in_window(ev_t, w["window_start_s"], w["window_end_s"])
            ev_win = ev_sess.loc[mask]
            if ev_win.empty:
                continue
            rel_t = ev_win["t_s"].to_numpy() - w["window_start_s"]
            for col in value_cols:
                for suf, val in compute_summary_stats(ev_win[col].to_numpy(), rel_t).items():
                    windows.at[idx, f"{prefix}_{col}_{suf}"] = val
    return windows


def _build_feature_column_list() -> list[str]:
    cols = []
    for prefix in ["coupling_tap_accel", "coupling_key_accel"]:
        for m in ACCEL_METRICS:
            cols += summary_stat_columns(f"{prefix}_{m}")
    for prefix in ["coupling_tap_orient", "coupling_key_orient"]:
        for m in ORIENT_METRICS:
            cols += summary_stat_columns(f"{prefix}_{m}")
    for m in GESTURE_COUPLING_METRICS:
        cols += summary_stat_columns(f"coupling_gesture_{m}")
    return cols


def aggregate_coupling_features(windows: pd.DataFrame, raw_df: pd.DataFrame,
                                 gesture_interactions: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _build_feature_column_list()
    windows = pd.concat([windows, pd.DataFrame(np.nan, index=windows.index, columns=feature_cols)], axis=1)

    tap_eligible = (
        (~windows["tapping_straddle_conflict"].fillna(False))
        & (windows["tapping_afforded"] == True)  # noqa: E712
    )
    key_eligible = (
        (~windows["typing_straddle_conflict"].fillna(False))
        & (windows["typing_afforded"] == True)  # noqa: E712
    )

    tap_accel = accel_event_coupling_features(raw_df, TAP_DOWN_KIND)
    windows = _agg_block(windows, tap_eligible, tap_accel, ACCEL_METRICS, "coupling_tap_accel")

    key_accel = accel_event_coupling_features(raw_df, KEY_DOWN_KIND)
    windows = _agg_block(windows, key_eligible, key_accel, ACCEL_METRICS, "coupling_key_accel")

    tap_orient = orient_event_coupling_features(raw_df, TAP_DOWN_KIND)
    windows = _agg_block(windows, tap_eligible, tap_orient, ORIENT_METRICS, "coupling_tap_orient")

    key_orient = orient_event_coupling_features(raw_df, KEY_DOWN_KIND)
    windows = _agg_block(windows, key_eligible, key_orient, ORIENT_METRICS, "coupling_key_orient")

    gest_coupling = gesture_coupling_features(raw_df, gesture_interactions)
    windows = _agg_block(windows, tap_eligible, gest_coupling, GESTURE_COUPLING_METRICS, "coupling_gesture")

    return windows


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from build_windows_dataset import build_skeleton  # noqa: E402
    from gesture import build_classified_interactions  # noqa: E402

    RAW_CANDIDATES = [
        "data/processed/raw_events.parquet", "raw_events.parquet",
        "/mnt/user-data/uploads/raw_events.parquet",
    ]
    raw_path = next((p for p in RAW_CANDIDATES if Path(p).exists()), None)
    if raw_path is None:
        print("No raw_events.parquet found -- skipping.")
        sys.exit(0)

    raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions.\n")

    interactions = build_classified_interactions(raw)

    windows = build_skeleton(raw)
    feature_cols = _build_feature_column_list()
    print(f"coupling.py now generates {len(feature_cols)} candidate feature columns (44 before rebuild).\n")

    windows = aggregate_coupling_features(windows, raw, interactions)

    print("Non-null rate, top 10 and bottom 10 columns:")
    rates = windows[feature_cols].notna().mean().sort_values(ascending=False)
    print(rates.head(10).round(3).to_string())
    print("...")
    print(rates.tail(10).round(3).to_string())
