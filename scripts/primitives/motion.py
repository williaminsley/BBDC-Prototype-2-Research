"""
scripts/primitives/motion.py

Motion/orientation feature family: the always-on continuous background
signal (distinct from the discrete foreground modalities in typing.py/
tap.py/gesture.py -- this family is computed for every window
unconditionally, gated only on having enough raw motion/orientation
samples, not on task affordance). Ported from 08_1_posture_and_motion_v3
(idle_mask, idle_runs, motion_arrays) and 08_2_behavioural_motion_derivations
(steadiness_and_bands, cross_axis_coupling).

DELIBERATELY EXCLUDED (flagged provisional in earlier project review, not
re-litigated here):
  - normalised_jerk: needs low-pass filtering before it's trustworthy --
    currently dominated by sensor noise. Not ported.
  - orienting_response (task-transition motion): current window design is
    contaminated by the previous task's closing tap, needs redesign, not
    just more data. Not ported.

UNIFORM SUFFIX DISCIPLINE (applied from the start this time, not as a
second pass): every genuine per-SAMPLE metric -- idle accelerometer
magnitude, active accelerometer magnitude, cross-axis rolling correlation
-- gets the full 9-statistic suffix set via _stats.py. Single per-window
derived quantities that are not a distribution over multiple raw
observations (idle_frac, sway/tremor band power, each inherently one
computed number per window) stay as single columns, same rule already
applied in tap.py/gesture.py/coupling.py.

WINDOW-LEVEL VS SESSION-LEVEL: 08_2's steadiness_and_bands() and
cross_axis_coupling() were built as SESSION-level computations (one FFT
over the whole session's idle runs, one rolling correlation over the whole
session). Recomputed here at WINDOW level instead -- each 15s window's own
idle run(s) and rolling correlation values are used, not the whole
session's. This is a genuine adaptation, not a straight port: window-level
idle runs are shorter (up to ~15s vs a whole session), so FFT-based band
features will have less support and should be treated as more provisional
than the session-level notebook version. Flagged via idle_run_max_len_s
riding along so downstream filtering can see how much support each
window's spectral features actually had.
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

MOTION_KIND = "devicemotion"
ORIENT_KIND = "deviceorientation"
ACCEL_COLS = ["payload_ax", "payload_ay", "payload_az"]

INTERACTION_KINDS = ["keydown", "keyup", "input", "touchstart", "touchmove", "touchend",
                     "touchcancel", "scroll", "window_scroll", "carousel_scroll",
                     "pointerdown", "pointerup", "pointercancel", "click"]
                     # touchcancel/pointercancel added 2026-09: these are
                     # genuine physical interaction moments (the person WAS
                     # touching the device even though the browser aborted
                     # the touch) that were previously missing from this
                     # list, so motion samples right around a cancel event
                     # could be wrongly classified as "idle" -- same class
                     # of oversight as the gesture_hold_ms touchcancel bug,
                     # concentrated on the same high-cancel-rate device.
IDLE_GUARD_MS = 1000.0   # ported from 08_1 -- a motion sample counts as "idle" only if
                          # it's at least this far from any interaction event on either side
MIN_IDLE_RUN = 60         # samples (~3s at the observed ~20Hz rate) -- ported from 08_1/08_2

SAMPLE_HZ = 20.0          # ported from 08_2
N_FFT = 64                # ported from 08_2 -- FFT window for sway/tremor bands
ROLL_WINDOW_S = 5.0       # ported from 08_2 -- rolling window for beta/gamma cross-axis correlation

MOTION_METRICS = ["idle_accel_mag", "active_accel_mag"]
CROSS_AXIS_METRIC = "cross_axis_corr"


# =============================================================================
# Event-level extractors
# =============================================================================

def motion_arrays(sess: pd.DataFrame):
    """Ported from 08_1/08_2 (identical in both). Returns t_ms, mag for one session."""
    m = sess.loc[sess["kind"] == MOTION_KIND].sort_values("tRelMs")
    t = m["tRelMs"].to_numpy(dtype=float)
    mag = np.sqrt(sum(pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float) ** 2
                      for c in ACCEL_COLS))
    ok = np.isfinite(mag)
    return t[ok], mag[ok]


def idle_mask(sess: pd.DataFrame, motion_t_ms: np.ndarray) -> np.ndarray:
    """Ported unchanged from 08_1. True where a motion sample is far enough
    from any interaction event on both sides to count as idle."""
    inter = np.sort(sess.loc[sess["kind"].isin(INTERACTION_KINDS), "tRelMs"].to_numpy(dtype=float))
    if len(inter) == 0:
        return np.ones(len(motion_t_ms), dtype=bool)
    idx = np.searchsorted(inter, motion_t_ms)
    prev_gap = np.where(idx > 0, motion_t_ms - inter[np.clip(idx - 1, 0, len(inter) - 1)], np.inf)
    next_gap = np.where(idx < len(inter), inter[np.clip(idx, 0, len(inter) - 1)] - motion_t_ms, np.inf)
    return (prev_gap >= IDLE_GUARD_MS) & (next_gap >= IDLE_GUARD_MS)


def orientation_series_pair(sess: pd.DataFrame):
    """
    Beta AND gamma extracted together from the SAME orientation event rows,
    keeping both positionally aligned to the same timestamps.

    FIX (2026-09): the previous approach called orientation_series() twice,
    once per column, each independently dropping rows where THAT column's
    value was non-finite. If beta and gamma ever had different NaN
    positions (plausible if either sensor axis briefly failed to report
    within an otherwise-populated event), the two returned arrays would
    silently misalign once truncated to a common length in
    _cross_axis_corr_series -- position i in the beta array would not
    necessarily correspond to the same timestamp as position i in the
    gamma array. Not empirically confirmed as having caused wrong output on
    this cohort's data, but a real risk with no guard against it. Fixed by
    filtering both columns from one shared frame with a single "both
    finite" mask, so alignment is guaranteed by construction rather than by
    assumption."""
    o = sess.loc[sess["kind"] == ORIENT_KIND].sort_values("tRelMs")
    t = o["tRelMs"].to_numpy(dtype=float) / 1000.0
    b = pd.to_numeric(o["payload_beta"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(o["payload_gamma"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(b) & np.isfinite(g)
    return t[ok], b[ok], g[ok]


def orientation_series(sess: pd.DataFrame, col: str = "payload_beta"):
    """Single-column beta OR gamma series. No longer called within this
    file (see orientation_series_pair() above, used by
    _cross_axis_corr_series instead) -- kept as it may still be used from
    notebooks or other scripts referencing this module directly.

    Uses the already-unwrapped beta if corrections.py has run
    (payload_beta_residual/_baseline present); falls back to raw
    payload_beta/gamma otherwise, since cross-axis coupling is about the
    RATE OF CHANGE (diff), which is far less sensitive to a 0/360-style
    wrap than an absolute mean would be -- a single wrap event would show
    up as one large diff outlier, not a systematic bias, so this family
    does not require the same circular-encoding treatment corrections.py
    applies for absolute-value aggregation."""
    o = sess.loc[sess["kind"] == ORIENT_KIND].sort_values("tRelMs")
    t = o["tRelMs"].to_numpy(dtype=float) / 1000.0
    v = pd.to_numeric(o[col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    return t[ok], v[ok]


# =============================================================================
# Window-level computation (adapted from session-level 08_2 functions --
# see module docstring)
# =============================================================================

def _idle_active_split(t_ms: np.ndarray, mag: np.ndarray, im: np.ndarray,
                        w_start_s: float, w_end_s: float):
    """Restrict motion arrays + idle mask to one window's span, split idle/active."""
    sel = (t_ms >= w_start_s * 1000.0) & (t_ms < w_end_s * 1000.0)
    t_w, mag_w, im_w = t_ms[sel], mag[sel], im[sel]
    return mag_w[im_w], mag_w[~im_w], t_w[im_w], t_w[~im_w]


def _longest_idle_run_len(im_w: np.ndarray) -> int:
    if not im_w.any():
        return 0
    change = np.where(np.diff(im_w.astype(int)) != 0)[0] + 1
    runs = np.split(np.arange(len(im_w)), change)
    lens = [len(r) for r in runs if len(r) and im_w[r[0]]]
    return max(lens) if lens else 0


def _sway_tremor_power(mag_idle: np.ndarray) -> tuple:
    """FFT-based sway (<=2Hz) and tremor (8-9.5Hz, truncated at Nyquist for
    the ~20Hz sample rate -- ported from 08_2's documented caveat) band
    power, computed only if enough contiguous idle samples exist."""
    if len(mag_idle) < N_FFT:
        return np.nan, np.nan
    x = mag_idle[:N_FFT] - mag_idle[:N_FFT].mean()
    pw = np.abs(np.fft.rfft(x)) ** 2
    tot = pw[1:].sum()
    if tot <= 0:
        return np.nan, np.nan
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_HZ)
    sway = float((pw / tot)[(freqs > 0) & (freqs <= 2)].sum())
    tremor = float((pw / tot)[(freqs >= 8) & (freqs <= 9.5)].sum())
    return sway, tremor


def _cross_axis_corr_series(sess: pd.DataFrame, w_start_s: float, w_end_s: float) -> np.ndarray:
    """Rolling correlation between beta-diff and gamma-diff, restricted to
    one window's span. Ported/adapted from 08_2's cross_axis_coupling.
    Uses orientation_series_pair() (not two separate orientation_series()
    calls) so beta and gamma stay positionally aligned to the same
    timestamps by construction -- see that function's docstring."""
    t, b, g = orientation_series_pair(sess)
    n = len(t)
    if n < 20:
        return np.array([]), np.array([])
    db, dg = np.diff(b), np.diff(g)
    t_mid = t[1:]
    win = max(5, int(round(ROLL_WINDOW_S * (n / max(t[-1] - t[0], 1e-6)))))
    corr = pd.Series(db).rolling(win, min_periods=max(3, win // 2)).corr(pd.Series(dg)).to_numpy()
    sel = (t_mid >= w_start_s) & (t_mid < w_end_s)
    return t_mid[sel], corr[sel]


# =============================================================================
# Window-level aggregation
# =============================================================================

def _build_feature_column_list() -> list[str]:
    cols = ["motion_idle_frac", "motion_idle_run_max_len_s",
            "motion_idle_sway_power", "motion_idle_tremor_power"]
    for m in MOTION_METRICS:
        cols += summary_stat_columns(f"motion_{m}")
    cols += summary_stat_columns(f"motion_{CROSS_AXIS_METRIC}")
    return cols


def aggregate_motion_features(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds motion/orientation feature columns to every window unconditionally
    -- this family has no task-affordance gating (motion/orientation is
    always-on background, per the continuous/discrete architecture split
    established early in this project), only a minimum-sample gate applied
    implicitly (windows with too little motion data simply get NaN, same
    as any other insufficient-support case).
    """
    feature_cols = _build_feature_column_list()
    windows = pd.concat([windows, pd.DataFrame(np.nan, index=windows.index, columns=feature_cols)], axis=1)

    for sid, sess_windows in windows.groupby("sessionId"):
        sess = raw_df.loc[raw_df["sessionId"] == sid]
        t_ms, mag = motion_arrays(sess)
        if len(t_ms) < 20:
            continue
        im = idle_mask(sess, t_ms)

        for idx, w in sess_windows.iterrows():
            w_start, w_end = w["window_start_s"], w["window_end_s"]

            mag_idle, mag_active, t_idle, t_active = _idle_active_split(t_ms, mag, im, w_start, w_end)
            n_total = len(mag_idle) + len(mag_active)
            if n_total == 0:
                continue

            windows.at[idx, "motion_idle_frac"] = len(mag_idle) / n_total

            for suf, val in compute_summary_stats(mag_idle, (t_idle / 1000.0) - w_start).items():
                windows.at[idx, f"motion_idle_accel_mag_{suf}"] = val
            for suf, val in compute_summary_stats(mag_active, (t_active / 1000.0) - w_start).items():
                windows.at[idx, f"motion_active_accel_mag_{suf}"] = val

            sel_w = (t_ms >= w_start * 1000.0) & (t_ms < w_end * 1000.0)
            im_w = im[sel_w]
            run_len = _longest_idle_run_len(im_w)
            windows.at[idx, "motion_idle_run_max_len_s"] = run_len / SAMPLE_HZ

            if len(mag_idle) >= MIN_IDLE_RUN:
                sway, tremor = _sway_tremor_power(mag_idle)
                windows.at[idx, "motion_idle_sway_power"] = sway
                windows.at[idx, "motion_idle_tremor_power"] = tremor

            corr_t, corr_v = _cross_axis_corr_series(sess, w_start, w_end)
            if len(corr_v):
                for suf, val in compute_summary_stats(corr_v, corr_t - w_start).items():
                    windows.at[idx, f"motion_cross_axis_corr_{suf}"] = val

    return windows


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from build_windows_dataset import build_skeleton  # noqa: E402

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

    windows = build_skeleton(raw)
    feature_cols = _build_feature_column_list()
    print(f"motion.py generates {len(feature_cols)} candidate feature columns.\n")

    windows = aggregate_motion_features(windows, raw)

    has_motion = windows["motion_idle_frac"].notna()
    print(f"{has_motion.sum()} of {len(windows)} windows have any motion data at all "
          f"(this family is unconditional -- no task-affordance gating).\n")

    print("Non-null rate among windows WITH motion data, top 10 and bottom 10:")
    elig = windows.loc[has_motion]
    rates = elig[feature_cols].notna().mean().sort_values(ascending=False)
    print(rates.head(10).round(3).to_string())
    print("...")
    print(rates.tail(10).round(3).to_string())

    print(f"\nmedian idle_run_max_len_s (support for spectral features): "
          f"{elig['motion_idle_run_max_len_s'].median():.2f}s")
    print(f"windows with sway/tremor power actually computed: "
          f"{elig['motion_idle_sway_power'].notna().sum()} "
          f"({elig['motion_idle_sway_power'].notna().mean():.1%})")
