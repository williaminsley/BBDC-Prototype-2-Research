"""
scripts/primitives/_stats.py

Shared summary-statistic helper implementing P1's naming convention:
feature name = family + metric + summary-statistic suffix, with a FIXED
set of suffixes applied uniformly to every raw per-event metric:

    _mean, _std, _median, _iqr, _p95, _max, _n, _cv, _slope, _tailmean

This was confirmed against an actual P1 slide (2026-08) after typing.py,
tap.py, gesture.py, and coupling.py were each found to apply an
inconsistent, ad hoc subset of these suffixes per metric (e.g. dwell time
got mean/median/std, transitions got only mean/median, tap position got
only median/iqr) rather than the same fixed set every time. This module
exists so every feature-family file computes stats the same way, instead
of each one reinventing which subset "feels right" per metric -- which is
exactly how the inconsistency happened in the first place.

WHEN TO USE THIS: only for a genuine per-event metric, i.e. one where a
single window can contain MULTIPLE raw values to summarise (dwell time of
several keystrokes in a window, several tap x-positions, several coupling
peak values). Do NOT use this for a value that is already a single number
per window by construction (a composition share like typing_share_letter,
a single coverage percentage like tap_coverage) -- those have no
within-window distribution to take a std/p95/slope of, and forcing this
helper onto them would produce meaningless columns (std of a single
number is always NaN or 0).

RELIABILITY THRESHOLDS (2026-09): previously each stat's minimum-n gate
was a bare literal (`n > 1`, `n >= 3`) scattered through the function body.
Pulled into named constants below so they're a single source of truth --
tunable in one place if a threshold turns out too permissive, and
self-documenting about *why* each stat needs the n it needs, rather than
a magic number with no explanation next to it.

_slope caveat: computed via ordinary least squares against each value's
within-window elapsed time. At 15s window length with as few as 2-5
events for a sparse metric, this is often close to noise -- generated per
the project's maximalist "generate everything, filter later" principle,
but should be treated as a provisional column, not a trusted one, until
checked against real variance at the filtering stage.

_tailmean rationale (2026-09): added after an empirical check on real
windowed data (gesture/motion/tap) found that _max is within 5% of _p95
for ~90% of windows in several sparse per-event families (e.g.
gesture_stroke_x0, coupling_tap_orient_*_latency_ms) -- meaning _max adds
almost nothing beyond _p95 there, because there simply aren't enough raw
points in a 15s window for "the extreme value" and "the 95th percentile"
to differ. _tailmean (mean of values >= that window's 90th percentile) is
only a genuinely DIFFERENT, more artifact-robust statistic than _max when
there are enough raw points for "the top 10%" to mean more than one
value -- so it's gated behind MIN_N_FOR_TAIL and left NaN otherwise,
rather than silently degenerating into a slower re-implementation of
_max. Coverage of this column (how often it's non-NaN per feature) is
itself informative at the diagnostics stage: a feature where _tailmean is
almost always NaN is a feature where _max was never redundant in the
first place, and vice versa.

NOT added: a symmetric _headmean (mean of the bottom decile). Checked and
rejected empirically, not just by symmetry-aesthetics: window-level
medians for duration/energy/magnitude-style metrics (gesture_hold_ms,
motion_idle_accel_mag, coupling_tap_accel_snr) are heavily right-skewed,
with the bulk of windows sitting near the low end and a long thin tail
stretching to rare highs. That means the bottom decile of such a metric
sits very close to where _mean/_median/_iqr already are -- a _headmean
column would be highly correlated with existing central-tendency columns
rather than describing new behaviour, adding multicollinearity/overfitting
risk without a new axis of signal. This may not hold for metrics that are
naturally two-sided rather than bounded-near-zero (e.g. gesture_stroke_x0/
y0, which spread in both directions around a screen-position center) --
if a two-sided metric is later found to have separately meaningful high
and low tails, add a metric-specific _headmean call there rather than
reintroducing it as a blanket 10th suffix here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUFFIXES = ["mean", "std", "median", "iqr", "p95", "max", "n", "cv", "slope", "tailmean"]

# Minimum raw-value count required before each stat is computed; below
# this the column is left NaN rather than reporting a number that isn't
# meaningfully what its name claims.
MIN_N_FOR_DISPERSION = 2   # std, iqr, p95, cv: need at least 2 points to define any spread
MIN_N_FOR_SLOPE = 3        # OLS needs >=3 points to be more than a 2-point line through everything
MIN_N_FOR_TAIL = 10        # tailmean: below this, "top 10%" rounds to <=1 value == _max already
MEAN_EPS = 1e-9            # guards cv's division when mean is ~0


def summary_stat_columns(prefix: str) -> list[str]:
    """The 10 column names this helper will produce for a given metric prefix."""
    return [f"{prefix}_{suf}" for suf in SUFFIXES]


def compute_summary_stats(values: np.ndarray, times: np.ndarray | None = None) -> dict:
    """
    Compute all summary statistics for one metric's raw values within one
    window. `values` are the raw per-event measurements; `times` (optional,
    same length) are each value's within-window elapsed time in seconds,
    used only for the slope statistic (ordinary least squares, value vs.
    time). If times is None, slope is NaN.

    Returns a dict with keys matching SUFFIXES -- caller prefixes these
    with the metric name.
    """
    v_raw = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(v_raw)
    v = v_raw[finite_mask]
    n = len(v)

    out = {suf: np.nan for suf in SUFFIXES}
    out["n"] = float(n)
    if n == 0:
        return out

    mean = float(np.mean(v))
    out["mean"] = mean
    out["median"] = float(np.median(v))
    out["max"] = float(np.max(v))

    if n >= MIN_N_FOR_DISPERSION:
        std = float(np.std(v, ddof=1))
        out["std"] = std
        out["iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        p95 = float(np.percentile(v, 95))
        out["p95"] = p95
        if abs(mean) > MEAN_EPS:
            out["cv"] = std / abs(mean)

    if n >= MIN_N_FOR_TAIL:
        p90 = np.percentile(v, 90)
        tail = v[v >= p90]
        if len(tail):
            out["tailmean"] = float(np.mean(tail))

    if times is not None and n >= MIN_N_FOR_SLOPE:
        t_raw = np.asarray(times, dtype=float)
        if len(t_raw) == len(v_raw):
            t = t_raw[finite_mask]
            if np.all(np.isfinite(t)) and np.ptp(t) > 0:
                try:
                    slope, _ = np.polyfit(t, v, 1)
                    out["slope"] = float(slope)
                except (np.linalg.LinAlgError, ValueError):
                    pass

    return out


def write_summary_stats(windows: pd.DataFrame, idx, prefix: str,
                          values: np.ndarray, times: np.ndarray | None = None) -> None:
    """Convenience wrapper: compute stats and write them directly into windows.at[idx, ...]."""
    stats = compute_summary_stats(values, times)
    for suf, val in stats.items():
        windows.at[idx, f"{prefix}_{suf}"] = val