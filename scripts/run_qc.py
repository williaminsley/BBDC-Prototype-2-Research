"""
scripts/run_qc.py

Dataset-level quality-control gate for the P2 windowed dataset
(data/processed/windows.parquet), run AFTER build_windows_dataset.py --
distinct from validate_raw_sessions.py's per-session structural checks,
same separation P1 used (P1 Section 4.5: "This gate distinguishes hard
failures, which halt the pipeline, from soft warnings, which are logged
but do not block downstream use... distinct from the per-session
validation").

Severity structure ported from P1's design (not its exact thresholds --
P2's schema and window design differ enough that the numeric cutoffs below
are P2-specific judgment calls, documented inline, not copied values):

  HARD FAIL: halts the pipeline. Something structurally wrong enough that
             no downstream use of the dataset should proceed.
  SOFT WARNING: logged, printed, does not block. Worth a human look but
                not fatal on its own.

This is the FIRST point in the P2 pipeline where the actual VALUES in
windows.parquet get checked, not just "does this column get populated" --
every check up to this point (in each primitives/*.py self-check) only
verified non-null rates, never verified numbers were sane in ANY sense.
This QC pass still doesn't do that (it's structural, not a values sanity
check), which is worth knowing as a limitation: passing this gate means
the dataset is structurally trustworthy, not that every number in it has
been eyeballed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Thresholds -- P2-specific, not ported numbers. Documented per threshold.
# =============================================================================

# A window counts as "low-activity" if none of the three discrete
# modalities produced any observed event at all -- i.e. genuinely nothing
# happened in that window (not just "afforded but idle", which is a real
# behavioural signal, but literally no typing/tapping/scrolling events).
MIN_TYPING_PRESENCE_FRAC = 0.05     # hard fail below this
MIN_TYPING_PRESENCE_WARN_FRAC = 0.20  # soft warning below this (typing is
                                        # known to be structurally sparser
                                        # than tapping/scrolling -- see
                                        # metadata.py's affordance table,
                                        # ~44% afforded vs ~97%+ for the
                                        # other two -- so its warning
                                        # threshold is set lower than a
                                        # uniform-across-modalities rule
                                        # would give)
MIN_TAPPING_PRESENCE_FRAC = 0.05
MIN_SCROLLING_PRESENCE_FRAC = 0.05

MAX_LOW_ACTIVITY_FRAC_HARD = 0.80    # hard fail if >80% of windows are low-activity
MAX_LOW_ACTIVITY_FRAC_WARN = 0.50    # soft warning if 50-80%

MIN_WINDOWS_WARN = 1000              # soft warning below this -- P2's 2,684-window
                                       # current dataset is itself close to this,
                                       # worth knowing the gate would already be
                                       # warning at a meaningfully larger cohort
                                       # size than P1's 462 windows if the ratio
                                       # doesn't hold up
MIN_PARTICIPANTS_WARN = 10           # soft warning below this

# "Core" columns checked for conditional missingness (missing AMONG windows
# where the modality is both afforded and not straddle-conflicted -- i.e.
# where the value SHOULD be computable if any events occurred). A high
# conditional-missing rate on these specific columns would indicate an
# extraction bug, since these are the highest-volume, least-sparse metric
# per family and should have low missingness whenever the family is
# eligible at all.
CORE_COLUMNS = {
    "typing_dwell_ms_n": ("typing_afforded", "typing_straddle_conflict"),
    "tap_count": ("tapping_afforded", "tapping_straddle_conflict"),
    "gesture_n_interactions": ("tapping_afforded", "tapping_straddle_conflict"),
}
MAX_CORE_MISSING_WARN = 0.50  # soft warning if a core column is missing in
                                # more than half of its eligible windows


# =============================================================================
# Check functions -- each returns (severity, message) or None if it passes
# =============================================================================

def check_nonempty(windows: pd.DataFrame) -> list:
    issues = []
    if len(windows) == 0:
        issues.append(("HARD FAIL", "zero windows in the dataset"))
    if windows["participantId"].nunique() == 0:
        issues.append(("HARD FAIL", "zero participants in the dataset"))
    return issues


def check_duplicate_windows(windows: pd.DataFrame) -> list:
    dupe_mask = windows.duplicated(subset=["sessionId", "window_index"], keep=False)
    n_dupes = int(dupe_mask.sum())
    if n_dupes > 0:
        return [("HARD FAIL", f"{n_dupes} duplicate window rows sharing the same "
                              f"sessionId + window_index")]
    return []


def check_modality_presence(windows: pd.DataFrame) -> list:
    issues = []
    n = len(windows)
    if n == 0:
        return issues

    checks = [
        ("typing", "typing_observed", MIN_TYPING_PRESENCE_FRAC, MIN_TYPING_PRESENCE_WARN_FRAC),
        ("tapping", "tapping_observed", MIN_TAPPING_PRESENCE_FRAC, MIN_TAPPING_PRESENCE_FRAC * 4),
        ("scrolling", "scrolling_observed", MIN_SCROLLING_PRESENCE_FRAC, MIN_SCROLLING_PRESENCE_FRAC * 4),
    ]
    for label, col, hard_thresh, warn_thresh in checks:
        frac = windows[col].fillna(False).mean()
        if frac < hard_thresh:
            issues.append(("HARD FAIL", f"{label} signal present in only {frac:.1%} of windows "
                                        f"(< {hard_thresh:.0%} hard-fail threshold)"))
        elif frac < warn_thresh:
            issues.append(("SOFT WARNING", f"{label} signal present in only {frac:.1%} of windows "
                                           f"(< {warn_thresh:.0%} warning threshold)"))
    return issues


def check_low_activity(windows: pd.DataFrame) -> list:
    low_activity = (
        ~windows["typing_observed"].fillna(False)
        & ~windows["tapping_observed"].fillna(False)
        & ~windows["scrolling_observed"].fillna(False)
    )
    frac = low_activity.mean()
    if frac > MAX_LOW_ACTIVITY_FRAC_HARD:
        return [("HARD FAIL", f"{frac:.1%} of windows have zero observed events in ALL "
                              f"three discrete modalities (> {MAX_LOW_ACTIVITY_FRAC_HARD:.0%})")]
    elif frac > MAX_LOW_ACTIVITY_FRAC_WARN:
        return [("SOFT WARNING", f"{frac:.1%} of windows have zero observed events in ALL "
                                 f"three discrete modalities (> {MAX_LOW_ACTIVITY_FRAC_WARN:.0%})")]
    return []


def check_sample_size(windows: pd.DataFrame) -> list:
    issues = []
    n_windows = len(windows)
    n_participants = windows["participantId"].nunique()
    if n_windows < MIN_WINDOWS_WARN:
        issues.append(("SOFT WARNING", f"only {n_windows:,} windows total "
                                       f"(< {MIN_WINDOWS_WARN:,} warning threshold)"))
    if n_participants < MIN_PARTICIPANTS_WARN:
        issues.append(("SOFT WARNING", f"only {n_participants} participants "
                                       f"(< {MIN_PARTICIPANTS_WARN} warning threshold)"))
    return issues


def check_core_column_missingness(windows: pd.DataFrame) -> list:
    issues = []
    for col, (afforded_col, straddle_col) in CORE_COLUMNS.items():
        if col not in windows.columns:
            issues.append(("SOFT WARNING", f"expected core column '{col}' not found in dataset -- "
                                           f"schema drift between this QC script and the pipeline?"))
            continue
        eligible = (
            (windows[afforded_col] == True)  # noqa: E712
            & (~windows[straddle_col].fillna(False))
        )
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            continue
        missing_frac = windows.loc[eligible, col].isna().mean()
        if missing_frac > MAX_CORE_MISSING_WARN:
            issues.append(("SOFT WARNING", f"core column '{col}' is missing in {missing_frac:.1%} "
                                           f"of its {n_eligible:,} eligible windows "
                                           f"(> {MAX_CORE_MISSING_WARN:.0%}) -- possible extraction gap"))
    return issues


def check_session_safe_readiness(windows: pd.DataFrame) -> list:
    """
    Not a pass/fail check -- reports session-count-per-participant distribution,
    since this determines which of P1's three train/enrol/probe assignment rules
    (Section 8.1: 1 session -> train only, 2 -> enrol/probe no train, 3+ ->
    train+enrol+probe) each participant will fall into once evaluation is built.
    Purely informational, always returns as SOFT WARNING-severity so it's visible
    in the report without implying anything is wrong.
    """
    sessions_per_p = windows.groupby("participantId")["sessionId"].nunique()
    counts = sessions_per_p.value_counts().sort_index()
    dist_str = ", ".join(f"{n} session(s): {c} participant(s)" for n, c in counts.items())
    single_session = int((sessions_per_p == 1).sum())
    msg = f"session-count distribution -- {dist_str}."
    if single_session:
        msg += (f" {single_session} participant(s) have only 1 session and will "
                f"contribute training-only data under P1's assignment rule (no "
                f"evaluation possible for them without a second session).")
    return [("INFO", msg)]


# =============================================================================
# Public entry point
# =============================================================================

def run_qc(windows: pd.DataFrame) -> dict:
    """
    Runs every check and returns {'hard_fails': [...], 'soft_warnings': [...],
    'info': [...]}. Does not raise -- caller decides what to do with a
    non-empty hard_fails list (run_pipeline.sh should halt on it, per P1's
    pattern of reading a JSON report rather than an exit code).
    """
    all_checks = [
        check_nonempty,
        check_duplicate_windows,
        check_modality_presence,
        check_low_activity,
        check_sample_size,
        check_core_column_missingness,
        check_session_safe_readiness,
    ]

    hard_fails, soft_warnings, info = [], [], []
    for check_fn in all_checks:
        try:
            results = check_fn(windows)
        except Exception as e:  # noqa: BLE001 -- a QC check itself failing is
                                  # worth surfacing as a hard fail, not crashing
                                  # the whole gate silently
            hard_fails.append(f"QC check '{check_fn.__name__}' raised an exception: {e}")
            continue
        for severity, msg in results:
            if severity == "HARD FAIL":
                hard_fails.append(msg)
            elif severity == "SOFT WARNING":
                soft_warnings.append(msg)
            elif severity == "INFO":
                info.append(msg)

    return {"hard_fails": hard_fails, "soft_warnings": soft_warnings, "info": info}


def print_qc_report(report: dict) -> None:
    print("=" * 70)
    print("P2 WINDOWED DATASET QC REPORT")
    print("=" * 70)

    if report["hard_fails"]:
        print(f"\nHARD FAILS ({len(report['hard_fails'])}) -- pipeline should halt:")
        for msg in report["hard_fails"]:
            print(f"  [FAIL] {msg}")
    else:
        print("\nNo hard fails.")

    if report["soft_warnings"]:
        print(f"\nSOFT WARNINGS ({len(report['soft_warnings'])}) -- logged, not blocking:")
        for msg in report["soft_warnings"]:
            print(f"  [WARN] {msg}")
    else:
        print("\nNo soft warnings.")

    if report["info"]:
        print(f"\nINFO:")
        for msg in report["info"]:
            print(f"  [INFO] {msg}")

    print("\n" + "=" * 70)
    verdict = "FAIL" if report["hard_fails"] else "PASS"
    print(f"VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    WINDOWS_CANDIDATES = [
        "data/processed/windows.parquet",
        "windows.parquet",
        "/mnt/user-data/uploads/windows.parquet",
    ]
    windows_path = next((p for p in WINDOWS_CANDIDATES if Path(p).exists()), None)
    if windows_path is None:
        print("No windows.parquet found -- run build_windows_dataset.py first.")
        sys.exit(1)

    windows = pd.read_parquet(windows_path)
    print(f"Loaded {windows_path}: {windows.shape[0]:,} rows x {windows.shape[1]:,} columns\n")

    report = run_qc(windows)
    print_qc_report(report)

    sys.exit(1 if report["hard_fails"] else 0)
