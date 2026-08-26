"""
corrections.py

Pre-windowing correction pass for the P2 canonical raw event stream.

This runs ONCE, on the whole raw stream, BEFORE build_windows_dataset.py cuts
anything into windows. Two corrections are applied here rather than inside
the windower, because both need context that a fixed-length window does not
have:

  1. Circular encoding for alpha/beta (orientation angles wrap at 0/360 or
     +-180; a window-level mean/std computed on raw degrees across a wrap is
     meaningless).
  2. Causal (backward-looking) baseline subtraction for beta/gamma, which
     needs up to 30s of history BEFORE a window starts. Building this inside
     the windower would mean either recomputing an overlapping baseline
     redundantly for every 7.5s stride, or having no valid baseline at all
     for the first windows of a session.

Both pieces are ported from 08_1_posture_and_motion_v3.ipynb, where they were
already built and validated (wrap counts confirmed, residual collapse
confirmed) -- this module is a de-plotted, de-notebooked version of that
logic, not a reimplementation.

Design decisions this module encodes (confirmed in conversation, 2026-08):
  - alpha: sin/cos circular encoding, applied unconditionally to every
    session (194 genuine wraps confirmed in the cohort; not session-specific).
  - beta: sin/cos circular encoding as well, applied unconditionally, even
    though only one session in the current cohort has been observed to wrap.
    Sin/cos costs nothing for sessions that don't wrap (a smooth reversible
    transform), so there is no upside to conditional per-session logic.
  - gamma: left as raw value. No wrap risk has been observed in any notebook
    for gamma (typically bounded -90..90), so no encoding is applied.
  - Baseline window: 30s trailing rolling median (BASE_WINDOW_S), decoupled
    from the 15s/7.5s window length/stride used downstream by the windower.
    These are answering different questions (stable "resting" orientation
    vs. fine-grained behavioural resolution) and should not be coupled.
  - Early-session coverage: windows whose baseline has fewer than WARMUP_S
    seconds of prior history still get a value (not dropped), but the row
    carries a baseline_coverage_frac flag so downstream filtering can decide
    whether to trust it. This follows the project's "log everything, decide
    later" principle (P1 Section 10.1) rather than discarding data at the
    windowing stage.

Public entry point: apply_corrections(raw_df) -> corrected_df
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORIENT_KIND = "deviceorientation"

# Baseline parameters -- ported unchanged from 08_1 (BASE_WINDOW_S, WARMUP_S).
# Deliberately decoupled from the windower's 15s/7.5s window length/stride.
BASE_WINDOW_S = 30.0   # memory length for the rolling-median baseline
WARMUP_S = 5.0          # minimum trailing history before a baseline counts as "full coverage"

# Orientation sample rate assumption used only to convert the second-based
# parameters above into a sample count for pandas' .rolling(). This is a
# fallback for sessions where the per-session rate can't be measured
# directly; per-session rate is computed and used when available (see
# _rolling_window_samples).
FALLBACK_ORIENT_HZ = 60.0


# =============================================================================
# Circular encoding
# =============================================================================

def circular_encode(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace a wrapping angle series (degrees) with (sin, cos) components.

    This removes the 0/360 (or +-180) discontinuity entirely rather than
    patching it after the fact -- any window-level mean/std computed on the
    sin/cos pair and converted back via atan2 is the correct circular mean,
    with no special-casing needed downstream.
    """
    v = np.asarray(series, dtype=float)
    rad = np.radians(v)
    return np.sin(rad), np.cos(rad)


def circular_mean_deg(sin_vals: np.ndarray, cos_vals: np.ndarray) -> float:
    """Circular mean in degrees [0, 360) from sin/cos components, ignoring NaNs."""
    s = np.nanmean(sin_vals)
    c = np.nanmean(cos_vals)
    if not (np.isfinite(s) and np.isfinite(c)):
        return np.nan
    return float((np.degrees(np.arctan2(s, c)) + 360.0) % 360.0)


def unwrap_deg(v: np.ndarray) -> np.ndarray:
    """
    Remove +-180 wrap discontinuities from a bounded circular series via
    np.unwrap. Ported unchanged from 08_1. Used only as a diagnostic /
    cross-check alongside circular_encode, not as the primary correction
    (sin/cos is preferred downstream because it composes safely with
    rolling mean/std without special-casing).
    """
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    out = v.copy()
    if ok.sum() > 1:
        out[ok] = np.degrees(np.unwrap(np.radians(v[ok])))
    return out


# =============================================================================
# Causal (backward-looking) baseline subtraction
# =============================================================================

def _rolling_window_samples(t_s: np.ndarray, base_window_s: float = BASE_WINDOW_S) -> int:
    """
    Estimate a sample count for pandas .rolling() from this session's actual
    median sample interval, falling back to FALLBACK_ORIENT_HZ if the
    session doesn't have enough samples to estimate a rate.
    """
    if len(t_s) < 3:
        hz = FALLBACK_ORIENT_HZ
    else:
        dt = np.median(np.diff(t_s))
        hz = 1.0 / dt if dt > 0 else FALLBACK_ORIENT_HZ
    return max(20, int(round(base_window_s * hz)))


def _warmup_samples(t_s: np.ndarray, warmup_s: float = WARMUP_S) -> int:
    if len(t_s) < 3:
        hz = FALLBACK_ORIENT_HZ
    else:
        dt = np.median(np.diff(t_s))
        hz = 1.0 / dt if dt > 0 else FALLBACK_ORIENT_HZ
    return max(10, int(round(warmup_s * hz)))


def causal_baseline(v: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """
    Trailing (backward-looking only) rolling-median baseline. Ported from
    08_1's causal_baselines(), 'rolling' method only -- the notebook also
    explored 'expanding' and 'ewma' variants during development; 'rolling'
    is the one carried forward here since it's the one whose behaviour was
    validated in the residual-collapse check (absolute tilt spread of tens
    of degrees collapsing to near-uniform residual spread across sessions).

    IMPORTANT: this must only ever be called on ordered, causal data. The
    caller (apply_corrections) is responsible for ensuring events are sorted
    by time and never cross a session boundary -- crossing a session
    boundary here would leak a different session's history into this one's
    baseline, which is the leakage failure mode flagged in conversation.
    """
    s = pd.Series(np.asarray(v, dtype=float))
    n_roll = _rolling_window_samples(t_s)
    n_warm = _warmup_samples(t_s)
    baseline = s.rolling(n_roll, min_periods=n_warm).median().to_numpy()
    return baseline


def baseline_coverage_frac(t_s: np.ndarray) -> np.ndarray:
    """
    Per-sample fraction of the full BASE_WINDOW_S history actually available
    at that point in the session (0 at session start, ramping to 1.0 once
    BASE_WINDOW_S of history has accumulated, capped at 1.0 thereafter).

    This is the flag that lets early-session windows be kept (per the
    keep-with-flag decision) rather than dropped outright.
    """
    t_s = np.asarray(t_s, dtype=float)
    if len(t_s) == 0:
        return np.array([])
    t0 = t_s[0]
    elapsed = t_s - t0
    return np.clip(elapsed / BASE_WINDOW_S, 0.0, 1.0)


# =============================================================================
# Public entry point
# =============================================================================

def apply_corrections(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full pre-windowing correction pass to the canonical raw event
    stream and return a corrected copy with new columns added (originals are
    preserved, nothing is overwritten or dropped):

        payload_alpha_sin, payload_alpha_cos     -- circular encoding, always
        payload_beta_sin,  payload_beta_cos      -- circular encoding, always
        payload_beta_baseline,  payload_beta_residual   -- causal baseline + residual
        payload_gamma_baseline, payload_gamma_residual  -- causal baseline + residual
        orient_baseline_coverage_frac            -- 0..1 flag, see baseline_coverage_frac

    gamma is NOT circularly encoded (no wrap risk observed), but DOES get the
    causal baseline treatment, same as beta -- baseline subtraction and
    circular encoding are separate corrections addressing separate problems.

    Runs per-session, in time order, and never lets one session's history
    leak into another's baseline (see causal_baseline docstring).
    """
    out_frames = []

    for sid, sess in raw_df.groupby("sessionId", sort=False):
        sess = sess.sort_values("tRelMs").copy()
        orient_mask = sess["kind"] == ORIENT_KIND

        if orient_mask.sum() == 0:
            out_frames.append(sess)
            continue

        idx = sess.index[orient_mask]
        t_s = sess.loc[idx, "tRelMs"].to_numpy(dtype=float) / 1000.0

        alpha = pd.to_numeric(sess.loc[idx, "payload_alpha"], errors="coerce").to_numpy()
        beta = pd.to_numeric(sess.loc[idx, "payload_beta"], errors="coerce").to_numpy()
        gamma = pd.to_numeric(sess.loc[idx, "payload_gamma"], errors="coerce").to_numpy()

        # --- circular encoding (alpha always, beta always, gamma never) ---
        alpha_sin, alpha_cos = circular_encode(alpha)
        beta_sin, beta_cos = circular_encode(beta)

        # --- causal baseline + residual (beta and gamma; alpha has no
        #     meaningful "resting value" to baseline against, it's a
        #     compass heading that can legitimately point anywhere) ---
        beta_baseline = causal_baseline(beta, t_s)
        gamma_baseline = causal_baseline(gamma, t_s)
        beta_residual = beta - beta_baseline
        gamma_residual = gamma - gamma_baseline

        coverage = baseline_coverage_frac(t_s)

        for col, vals in [
            ("payload_alpha_sin", alpha_sin),
            ("payload_alpha_cos", alpha_cos),
            ("payload_beta_sin", beta_sin),
            ("payload_beta_cos", beta_cos),
            ("payload_beta_baseline", beta_baseline),
            ("payload_beta_residual", beta_residual),
            ("payload_gamma_baseline", gamma_baseline),
            ("payload_gamma_residual", gamma_residual),
            ("orient_baseline_coverage_frac", coverage),
        ]:
            if col not in sess.columns:
                sess[col] = np.nan
            sess.loc[idx, col] = vals

        out_frames.append(sess)

    corrected = pd.concat(out_frames, ignore_index=False).sort_index()
    return corrected


# =============================================================================
# Self-check: reproduce 08_1's own wrap-count finding as a smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    RAW_CANDIDATES = [
        "data/processed/raw_events.parquet",
        "raw_events.parquet",
        "/mnt/user-data/uploads/raw_events.parquet",
    ]
    raw_path = next((p for p in RAW_CANDIDATES if Path(p).exists()), None)
    if raw_path is None:
        print("No raw_events.parquet found for self-check -- skipping.")
        sys.exit(0)

    cols = ["sessionId", "participantId", "kind", "tRelMs",
            "payload_alpha", "payload_beta", "payload_gamma"]
    raw = pd.read_parquet(raw_path, columns=cols)

    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions.")
    corrected = apply_corrections(raw)

    orient = corrected.loc[corrected["kind"] == ORIENT_KIND]
    n_alpha_wraps = 0
    for sid, s in orient.groupby("sessionId"):
        a = pd.to_numeric(s["payload_alpha"], errors="coerce").dropna().to_numpy()
        if len(a) > 1:
            n_alpha_wraps += int((np.abs(np.diff(a)) > 180).sum())

    print(f"alpha wrap events detected (raw degree jumps > 180 across cohort): {n_alpha_wraps}")
    print("(08_1 reported 194 genuine wraps for the cohort at the time it was run; "
          "session count may differ here, so this is a sanity magnitude check, "
          "not an exact-match assertion.)")

    non_null_resid = corrected["payload_beta_residual"].notna().sum()
    print(f"payload_beta_residual populated for {non_null_resid:,} of "
          f"{orient_count if (orient_count := len(orient)) else 0:,} orientation events")

    cov = corrected["orient_baseline_coverage_frac"].dropna()
    print(f"baseline_coverage_frac: min={cov.min():.3f}, "
          f"pct at full coverage (==1.0)={100*(cov == 1.0).mean():.1f}%")
