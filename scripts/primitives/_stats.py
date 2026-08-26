"""
scripts/primitives/_stats.py

Shared summary-statistic helper implementing P1's naming convention:
feature name = family + metric + summary-statistic suffix, with a FIXED
set of suffixes applied uniformly to every raw per-event metric:

    _mean, _std, _median, _iqr, _p95, _max, _n, _cv, _slope

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

_slope caveat: computed via ordinary least squares against each value's
within-window elapsed time. At 15s window length with as few as 2-5
events for a sparse metric, this is often close to noise -- generated per
the project's maximalist "generate everything, filter later" principle,
but should be treated as a provisional column, not a trusted one, until
checked against real variance at the filtering stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUFFIXES = ["mean", "std", "median", "iqr", "p95", "max", "n", "cv", "slope"]


def summary_stat_columns(prefix: str) -> list[str]:
    """The 9 column names this helper will produce for a given metric prefix."""
    return [f"{prefix}_{suf}" for suf in SUFFIXES]


def compute_summary_stats(values: np.ndarray, times: np.ndarray | None = None) -> dict:
    """
    Compute all 9 summary statistics for one metric's raw values within one
    window. `values` are the raw per-event measurements; `times` (optional,
    same length) are each value's within-window elapsed time in seconds,
    used only for the slope statistic (ordinary least squares, value vs.
    time). If times is None, slope is NaN.

    Returns a dict with keys 'mean', 'std', 'median', 'iqr', 'p95', 'max',
    'n', 'cv', 'slope' -- caller prefixes these with the metric name.
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
    std = float(np.std(v)) if n > 1 else np.nan
    out["mean"] = mean
    out["std"] = std
    out["median"] = float(np.median(v))
    out["max"] = float(np.max(v))
    if n > 1:
        out["iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        out["p95"] = float(np.percentile(v, 95))
    if n > 1 and abs(mean) > 1e-9:
        out["cv"] = std / abs(mean)

    if times is not None and n >= 3:
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
