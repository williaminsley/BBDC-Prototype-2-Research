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

TRIED AND REJECTED (2026-09): a tenth suffix, _tailmean (mean of values >=
a window's 90th percentile), was added as a candidate replacement for
_max on the theory that _max is a single, artifact-exposed point while
_tailmean would be a steadier, more robust summary of the same "how
extreme does this get" question -- gated behind a minimum raw-value count
(n >= 10) so it wouldn't just silently reproduce _max for the many sparse
per-event metrics in this project (gesture strokes, per-gesture coupling
values) where there usually aren't enough points in a window for "top
10%" to mean more than one value.

Removed after checking real data, not just the synthetic i.i.d. test that
motivated it: of the 29 _tailmean columns that survived the n-gate and
the diagnostics filtering pass, EVERY one was 0.92-0.998 correlated with
a column already in the dataset (mostly _p95, a few with _std) --
including motion_idle_accel_mag and the tap-accelerometer coupling
metrics, the exact families predicted to benefit most. The synthetic
test used independent samples, where the top decile and the 95th
percentile genuinely can diverge; real within-window events aren't
independent like that -- when a person's pace or vigor shifts during a
window, it shifts the whole tail together, so _p95 and a top-decile mean
end up moving in near-lockstep even at n>=10. Recorded here, with the
reasoning, so this isn't re-proposed later without the evidence that
killed it the first time.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUFFIXES = ["mean", "std", "median", "iqr", "p95", "max", "n", "cv", "slope"]

# Minimum raw-value count required before each stat is computed; below
# this the column is left NaN rather than reporting a number that isn't
# meaningfully what its name claims.
MIN_N_FOR_DISPERSION = 2   # std, iqr, p95, cv: need at least 2 points to define any spread
MIN_N_FOR_SLOPE = 3        # OLS needs >=3 points to be more than a 2-point line through everything
MEAN_EPS = 1e-9            # guards cv's division when mean is ~0


def summary_stat_columns(prefix: str) -> list[str]:
    """The 9 column names this helper will produce for a given metric prefix."""
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


def pair_sequential_events(down_t: np.ndarray, up_t: np.ndarray,
                            up_is_cancel: np.ndarray | None = None):
    """
    Greedy sequential down/up pairing, WITH memory of already-claimed up
    events. Added 2026-09 after finding the same bug independently
    duplicated in three places (gesture.py's touchstart/touchend pairing,
    tap.py's tap_hold_durations, typing_features.py's paired_durations):
    each used a "search forward from this down for the next up" pattern
    with no tracking of which up events an earlier down had already
    claimed. When two downs occur close together (touch/key rollover)
    before either's up fires -- normal in fast typing or overlapping taps
    -- both downs would independently find and report the SAME up event,
    silently corrupting one or both durations. Confirmed on real cohort
    data: 298 of 7,219 typing dwell pairs (4.1%) and 47 of 11,454 tap hold
    pairs reused the same up timestamp across multiple downs before this
    fix.

    This is now the single shared implementation all three should call,
    rather than each maintaining its own copy -- that duplication is
    exactly how the bug went unfixed in two places after being found and
    fixed in the third.

    down_t : sorted array of down-event timestamps (touchstart / keydown).
    up_t : sorted array of up-like timestamps -- the genuine terminating
        event (touchend / keyup). If a cancel concept applies (touchcancel
        has no keyboard equivalent), MERGE cancel timestamps into this
        array (sorted) and pass up_is_cancel to flag them -- this lets the
        pairing pointer correctly stop at a cancel instead of overshooting
        past it to steal a later down's genuine up (the second bug found
        in the same code: a touchstart resolved by touchcancel, not
        touchend, would otherwise get paired with an unrelated later
        touch's touchend, producing an artificially inflated duration).
    up_is_cancel : boolean array, same length as up_t, True where that
        entry is a cancel rather than a genuine release. None if there is
        no cancel concept for this down/up pair (e.g. keydown/keyup).

    Returns (down_idx, up_val, is_cancel) as three same-length arrays:
      down_idx  -- indices into down_t for every down that found a match
                   (a trailing unmatched down at the end of a session is
                   omitted, same as before this refactor).
      up_val    -- the matched up timestamp for each.
      is_cancel -- whether that match was a cancel (all False if
                   up_is_cancel was None). Callers should typically
                   exclude is_cancel==True rows from any duration feature,
                   since a cancelled cycle is not a genuine completed
                   press-release -- but the pairing pointer still needed
                   to see it to stay correctly synchronised.
    """
    if up_is_cancel is None:
        up_is_cancel = np.zeros(len(up_t), dtype=bool)

    down_idx_out, up_val_out, is_cancel_out = [], [], []
    ui = 0
    for di, d in enumerate(down_t):
        while ui < len(up_t) and up_t[ui] < d:
            ui += 1
        if ui >= len(up_t):
            break
        down_idx_out.append(di)
        up_val_out.append(up_t[ui])
        is_cancel_out.append(bool(up_is_cancel[ui]))
        ui += 1  # consume this up -- the fix. A later down can never claim it again.

    return (np.array(down_idx_out, dtype=int),
            np.array(up_val_out, dtype=float),
            np.array(is_cancel_out, dtype=bool))
