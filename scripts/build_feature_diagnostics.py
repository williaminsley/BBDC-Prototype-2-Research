"""
Per-feature diagnostic screening table over windows.parquet.

WHAT THIS IS FOR
-----------------
Not a power ranking, not an evaluation. This is a screening pass: for each of
the ~574 behavioural feature columns in windows.parquet, answer "is this a
usable candidate feature, or is it dead/artifact/confounded, and why?" One
row per feature, a handful of yes/no flags plus supporting numbers, and a
free-text note for anything already known from prior notebooks. The goal is
to be able to scan the table and immediately see which features are clean,
which are dead weight, and which need a decision -- the way tap_radiusX/
tap_radiusY/tap_force turned out to need one (device-locked on iOS, found by
hand in notebooks/windows-explorer -- see IS_DEVICE_LOCKED below for the
same check now applied to every feature, not just the one we happened to
look at).

Deliberately NOT included (by design decision, not oversight):
  - between/within-PARTICIPANT separation ratio -- that's an evaluation-
    power question, not a screening one. Dropped after discussion.
  - within-person replication across sessions -- useful, but needs a second
    per-session aggregation pass; deferred to a v2 addition, doesn't change
    anything else here if bolted on later.
  - a long (feature x participant) coverage table -- v1 output is wide-only,
    one row per feature. The wide table's min_participant_non_null_pct
    column tells you the worst case exists; drill into which participant via
    windows.parquet directly if that number looks bad.

OUTPUT
------
data/processed/feature_diagnostics.csv (+ .parquet), one row per feature,
indexed by feature name.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Config
# =============================================================================

FAMILY_PREFIXES = ("typing_", "tap_", "gesture_", "coupling_", "motion_")

# These share a family prefix by coincidence (typing_*) but are skeleton/
# metadata columns from build_windows_dataset.py's affordance layer, not
# behavioural features -- excluded explicitly rather than relying on the
# prefix match alone, which would otherwise sweep them in.
NON_FEATURE_COLS_SHARING_PREFIX = {
    "typing_afforded", "typing_observed", "typing_straddle_conflict",
}

# Which {modality}_afforded flag gates each family's non-nullness. Only
# typing and tap map cleanly onto a single flag -- gesture windows can be
# gated by either tapping_afforded or scrolling_afforded depending on the
# gesture's sub-type (scroll_fling/scroll_drag vs drag_swipe/tap/long_press),
# and coupling/motion aren't gated by an explicit afforded flag at all
# (coupling depends on which underlying event types co-occur; motion is a
# largely passive stream). Left as None rather than forcing a wrong 1:1.
GOVERNING_FLAG = {
    "typing": "typing_afforded",
    "tap": "tapping_afforded",
    "gesture": None,   # multiple possible gates, see docstring above
    "coupling": None,
    "motion": None,
}

CTX_COLS = [
    "ctxTimeOfDay", "ctxFatigue", "ctxFocusLevel", "ctxEnvironmentNoise",
    "ctxMovement", "ctxPosture", "ctxHandUse", "ctxInputDevice",
    "ctxCaffeine", "ctxAlcohol", "ctxPrivacy",
]

# Column-name patterns treated as raw counts/tallies -- P1 blocked these
# outright from the behavioural candidate set (Section 4.4 of the P1
# reference doc), and this project's own finding (44.3% of participant pairs
# separable on raw decision-event counts alone, zero behavioural signal)
# is exactly why. Same logic applied here, generalized across all families.
COUNT_LIKE_SUFFIXES = ("_n", "_count")
COUNT_LIKE_EXACT = {
    "tap_count", "tap_cells_touched", "typing_keydown_count",
    "gesture_n_interactions",
}

# Thresholds -- documented here so they're easy to find and tune, not
# buried inline. Deliberately conservative (a feature needs a fairly stark
# pattern to get flagged) since these flags feed a manual decision, not an
# automatic exclusion.
NEAR_EMPTY_THRESHOLD = 0.01          # <1% populated, matches nb12's audit
MIN_GROUP_N = 5                      # a device-model/ctx group needs this many
                                      # non-null obs before its std is trusted
MIN_INFORMATIVE_WINDOWS = 20         # a feature must vary in at least this many
                                      # windows before the grouped lock/assoc
                                      # test is meaningful. Added 2026-09 after
                                      # typing_autorepeat_share was flagged
                                      # device-locked with assoc=inf: it is
                                      # non-zero in only 6 of 2,962 windows, so
                                      # nearly every device-model group has
                                      # exactly zero internal variance BY
                                      # CONSTRUCTION and the between/within
                                      # ratio explodes. That is a rare event
                                      # being mistaken for a device
                                      # fingerprint -- the same sparsity trap
                                      # already fixed in the separation-ratio
                                      # and correlation checks, recurring here.
DEVICE_LOCK_ASSOC_THRESHOLD = 3.0    # between-group std / within-group std
DEVICE_LOCK_WITHIN_RATIO = 0.40      # AND median within-group std must be
                                      # < 40% of the feature's overall std.
                                      # Loosened from an initial 0.10 after
                                      # checking against tap_radiusX/tap_force,
                                      # which are confirmed device-locked (see
                                      # notebook Section 9 -- row-level
                                      # identical to each other on 90%+ of
                                      # iPhone rows) but have real, nonzero
                                      # within-device spread, not tap_radiusY's
                                      # exact-zero case. This is a screening
                                      # tool, not a final verdict -- erring
                                      # toward flagging borderline cases for a
                                      # manual look costs little; missing a
                                      # real lock costs more.
HIGH_CTX_ASSOC_THRESHOLD = 3.0       # same style of check, applied per ctx col
HIGH_EXPOSURE_CORR_THRESHOLD = 0.5   # |corr| with participant window count
MIN_PAIRWISE_N = 30                  # a correlation computed from fewer
                                      # non-null overlapping rows than this
                                      # is masked out before ranking -- see
                                      # compute_redundancy docstring for the
                                      # bug this specifically fixes

# Known issues already surfaced by hand in prior notebooks -- seeded here so
# they show up in the table instead of only living in markdown cells.
# Matched by prefix; first match wins.
KNOWN_ISSUES = {
    "tap_radiusX_": "Device-locked on iOS: near-constant within deviceModel, "
                     "sharp jumps between models. Confirmed by is_device_locked below.",
    "tap_radiusY_": "Exactly 0 for ~95% of newer-iPhone windows -- looks like "
                     "an unsupported WebKit touch-event API field, not a "
                     "behavioural absence.",
    "tap_force_": "On iPhone, row-level identical to tap_radiusX_* in 90%+ "
                   "of rows -- almost certainly a shared fallback proxy value, "
                   "not two independent measurements. Confirmed by is_device_locked below.",
    "tap_coverage": "Correlates r=1.0 with tap_cells_touched -- fully redundant, "
                     "whatever's decided for one should apply to both.",
    "tap_cells_touched": "Correlates r=1.0 with tap_coverage -- fully redundant, "
                          "and is itself a raw count (see is_count_like).",
    "typing_transition_space_space_": "Rare key-class transition (space "
                                       "followed by space) -- near-empty by "
                                       "construction given this task's typed content, not a bug.",
    "typing_transition_backspace_space_": "Rare key-class transition -- near-empty "
                                           "by construction, not a bug.",
    "typing_dwell_digit_": "Digits are rare in this task's typed content -- "
                            "near-empty by construction, not a bug.",
    "typing_dwell_punct_or_symbol_": "Punctuation/symbols are rare in this "
                                      "task's typed content -- near-empty by construction, not a bug.",
    "typing_dwell_enter_": "Enter keypresses are infrequent relative to window "
                            "length -- near-empty by construction, not a bug.",
}


def get_known_issue(feature: str) -> str | None:
    for prefix, note in KNOWN_ISSUES.items():
        if feature.startswith(prefix) or feature == prefix:
            return note
    return None


def get_family(feature: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if feature.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def parse_metric_and_suffix(feature: str, family: str) -> tuple[str, str]:
    """Strip the family prefix, then split off a trailing 9-suffix stat if
    present. Columns without a stat suffix (tap_coverage, tap_count, the
    afforded/observed flags) get stat_suffix='none'."""
    body = feature[len(family) + 1:]
    KNOWN_SUFFIXES = ("mean", "std", "median", "iqr", "p95", "max", "n", "cv", "slope", "tailmean")
    parts = body.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in KNOWN_SUFFIXES:
        return parts[0], parts[1]
    return body, "none"


def is_count_like(feature: str) -> bool:
    if feature in COUNT_LIKE_EXACT:
        return True
    return feature.endswith(COUNT_LIKE_SUFFIXES)


# =============================================================================
# Per-feature diagnostics
# =============================================================================

def compute_availability(df: pd.DataFrame, feature: str) -> dict:
    col = df[feature]
    non_null_pct_overall = float(col.notna().mean())
    per_participant_pct = df.groupby("participantId")[feature].apply(lambda s: s.notna().mean())
    n_participants_any_data = int((per_participant_pct > 0).sum())
    min_participant_non_null_pct = float(per_participant_pct.min()) if len(per_participant_pct) else np.nan
    return {
        "non_null_pct_overall": round(non_null_pct_overall * 100, 3),
        "is_fully_empty": non_null_pct_overall == 0,
        "is_near_empty": 0 < non_null_pct_overall < NEAR_EMPTY_THRESHOLD,
        "n_participants_any_data": n_participants_any_data,
        "min_participant_non_null_pct": round(min_participant_non_null_pct * 100, 3)
        if pd.notna(min_participant_non_null_pct) else np.nan,
    }


def compute_exposure_confound(df: pd.DataFrame, feature: str, n_windows_per_participant: pd.Series) -> dict:
    participant_means = df.groupby("participantId")[feature].mean()
    aligned = pd.concat([participant_means, n_windows_per_participant], axis=1).dropna()
    aligned.columns = ["feature_mean", "n_windows"]
    corr = np.nan
    if len(aligned) >= 5 and aligned["feature_mean"].nunique() > 1:
        corr = aligned["feature_mean"].corr(aligned["n_windows"])
    return {
        "corr_with_participant_n_windows": round(float(corr), 3) if pd.notna(corr) else np.nan,
        "high_exposure_corr": pd.notna(corr) and abs(corr) >= HIGH_EXPOSURE_CORR_THRESHOLD,
    }


def _grouped_lock_check(df: pd.DataFrame, feature: str, group_col: str) -> dict:
    """Shared logic for device-model-lock and ctx-association checks: is this
    feature near-constant WITHIN groups of group_col but sharply different
    BETWEEN groups? That pattern means the feature is fingerprinting the
    group variable (device model, fatigue level, etc.), not behaviour."""
    sub = df[[group_col, feature]].dropna()
    if sub[group_col].nunique() < 2:
        return {"assoc": np.nan, "locked": False, "assessable": False}

    # Guard against near-constant features (see MIN_INFORMATIVE_WINDOWS).
    # Reported as NOT ASSESSABLE rather than as "not locked": those are
    # different claims, and this layer only screens -- it should never
    # assert an answer it cannot support. The decision layer
    # (build_behavioural_dataset.py) treats unknown as its own case.
    vals = sub[feature]
    if len(vals):
        mode_vals = vals.mode()
        if len(mode_vals) and int((vals != mode_vals.iloc[0]).sum()) < MIN_INFORMATIVE_WINDOWS:
            return {"assoc": np.nan, "locked": False, "assessable": False}

    grouped = sub.groupby(group_col)[feature].agg(["mean", "std", "count"])
    well_supported = grouped[grouped["count"] >= MIN_GROUP_N]
    if len(well_supported) < 2:
        return {"assoc": np.nan, "locked": False, "assessable": False}

    between = well_supported["mean"].std()
    # Median, not mean, across group stds -- deliberately robust to a
    # minority of noisy groups. tap_radiusY_mean is the motivating case:
    # 6 of 8 deviceModel groups have exactly zero variance (iOS not
    # reporting the field) while 2 groups (a vague "older_iphone" bucket,
    # and the one Android device) have real internal variance. The mean
    # across all 8 stds gets pulled up enough to miss the lock; the median
    # correctly reflects that most of the cohort is locked.
    within = well_supported["std"].median()
    overall_std = sub[feature].std()

    if pd.isna(between) or pd.isna(overall_std) or overall_std <= 0:
        return {"assoc": np.nan, "locked": False, "assessable": False}

    if within <= 0:
        # Majority of groups have exactly zero internal variance (like
        # tap_radiusY_mean above). Can't form a between/within ratio with a
        # zero denominator -- if there's any between-group spread at all,
        # that alone is the strongest possible lock signal, so flag it
        # directly rather than dividing by zero or discarding the case.
        locked = bool(between > 0)
        return {"assoc": np.inf if locked else np.nan, "locked": locked, "assessable": True}

    assoc = between / within
    locked = bool(assoc >= DEVICE_LOCK_ASSOC_THRESHOLD and (within / overall_std) < DEVICE_LOCK_WITHIN_RATIO)
    return {"assoc": round(float(assoc), 3), "locked": locked, "assessable": True}


def compute_device_lock(df: pd.DataFrame, feature: str) -> dict:
    result = _grouped_lock_check(df, feature, "deviceModel")
    return {
        "device_model_assoc": result["assoc"],
        "is_device_locked": result["locked"] if result["assessable"] else np.nan,
        "device_lock_assessable": result["assessable"],
    }


def compute_ctx_confound(df: pd.DataFrame, feature: str) -> dict:
    best_assoc, best_col = np.nan, None
    for ctx_col in CTX_COLS:
        if ctx_col not in df.columns:
            continue
        result = _grouped_lock_check(df, feature, ctx_col)
        if pd.notna(result["assoc"]) and (pd.isna(best_assoc) or result["assoc"] > best_assoc):
            best_assoc, best_col = result["assoc"], ctx_col
    return {
        "max_ctx_assoc": best_assoc,
        "which_ctx_col": best_col,
        "high_ctx_assoc": pd.notna(best_assoc) and best_assoc >= HIGH_CTX_ASSOC_THRESHOLD,
    }


def compute_redundancy(corr_matrix: pd.DataFrame, feature: str, valid_pair_counts: pd.DataFrame) -> dict:
    """Strongest correlation with any other feature.

    corr_matrix is built with pandas' default pairwise-complete-observations
    behaviour: pandas.DataFrame.corr() silently computes each pairwise
    correlation from whatever non-null rows happen to overlap between those
    two specific columns, which can be as few as 2 for a near-empty column.
    A correlation from 2 points is degenerate -- always exactly +-1
    mathematically -- and when one of those points comes from a
    near-zero-variance column, floating-point precision pushes the computed
    value slightly past 1.0 (confirmed: coupling_gesture_mid_phase_energy_cv
    vs typing_dwell_digit_ms_slope, 2 overlapping rows, reported r=1.020).
    valid_pair_counts holds the actual overlap count for every pair so those
    can be masked out here rather than trusted."""
    row = corr_matrix[feature].drop(index=feature).abs().clip(upper=1.0)
    counts = valid_pair_counts[feature].drop(index=feature)
    row = row[counts >= MIN_PAIRWISE_N].dropna()
    if len(row) == 0:
        return {"max_abs_corr": np.nan, "max_abs_corr_with": None}
    best = row.idxmax()
    return {"max_abs_corr": round(float(row.loc[best]), 3), "max_abs_corr_with": best}


# =============================================================================
# Main
# =============================================================================

def build_feature_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = sorted(
        c for c in df.columns
        if c.startswith(FAMILY_PREFIXES) and c not in NON_FEATURE_COLS_SHARING_PREFIX
    )
    print(f"[diagnostics] scanning {len(feature_cols)} behavioural feature columns")

    n_windows_per_participant = df.groupby("participantId").size()

    # Cross-family correlation matrix, computed once up front rather than
    # per-feature -- this is the full 574x574 matrix (deliberately, not
    # within-family only: this project's standing concern is that apparent
    # multi-feature separability collapses to one latent factor like pace,
    # and pace-driven redundancy shows up ACROSS families -- e.g. a fast
    # typist is often also a fast tapper -- so restricting to within-family
    # would hide exactly the pattern worth catching).
    print("[diagnostics] computing cross-family correlation matrix...")
    corr_matrix = df[feature_cols].corr()

    # Pairwise non-null overlap count for every feature pair, via boolean
    # matrix multiplication (fast: one 574x2962 x 2962x574 matmul, not a
    # per-pair Python loop). pandas' .corr() computes each pairwise
    # correlation from whatever rows happen to be non-null in BOTH columns,
    # which can be as few as 2 for a near-empty column -- see
    # compute_redundancy's docstring for the resulting bug this guards
    # against (a >1.0 "correlation" from a 2-point degenerate fit).
    notna = df[feature_cols].notna().astype(int).to_numpy()
    valid_pair_counts = pd.DataFrame(
        notna.T @ notna, index=feature_cols, columns=feature_cols
    )

    rows = []
    for i, feature in enumerate(feature_cols):
        if i % 100 == 0:
            print(f"[diagnostics] {i}/{len(feature_cols)}...")

        family = get_family(feature)
        metric, stat_suffix = parse_metric_and_suffix(feature, family)

        row = {
            "feature": feature,
            "family": family,
            "metric": metric,
            "stat_suffix": stat_suffix,
            "governing_flag": GOVERNING_FLAG.get(family),
            "is_count_like": is_count_like(feature),
        }
        row.update(compute_availability(df, feature))
        row.update(compute_exposure_confound(df, feature, n_windows_per_participant))
        row.update(compute_device_lock(df, feature))
        row.update(compute_ctx_confound(df, feature))
        row.update(compute_redundancy(corr_matrix, feature, valid_pair_counts))
        row["known_issue"] = get_known_issue(feature)

        rows.append(row)

    out = pd.DataFrame(rows).set_index("feature")

    # A single needs_review flag folding together every red flag above, so
    # the table can be sorted/filtered at a glance without remembering which
    # of the individual columns matter. Doesn't replace the individual
    # columns -- just a fast way to triage.
    out["needs_review"] = (
        out["is_fully_empty"] | out["is_near_empty"] | out["is_count_like"]
        | out["high_exposure_corr"] | out["is_device_locked"].fillna(False).astype(bool)
        | out["high_ctx_assoc"]
        | out["known_issue"].notna()
    )

    col_order = [
        "family", "metric", "stat_suffix", "governing_flag",
        "non_null_pct_overall", "is_fully_empty", "is_near_empty",
        "n_participants_any_data", "min_participant_non_null_pct",
        "is_count_like", "corr_with_participant_n_windows", "high_exposure_corr",
        "device_model_assoc", "is_device_locked", "device_lock_assessable",
        "max_ctx_assoc", "which_ctx_col", "high_ctx_assoc",
        "max_abs_corr", "max_abs_corr_with",
        "known_issue", "needs_review",
    ]
    out = out[col_order]

    print(f"\n[diagnostics] done: {len(out)} features, {out['needs_review'].sum()} flagged for review")
    return out


if __name__ == "__main__":
    CANDIDATES = [
        "data/processed/windows.parquet",
        "../data/processed/windows.parquet",
        "/mnt/user-data/uploads/windows.parquet",
    ]
    path = next((p for p in CANDIDATES if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(f"windows.parquet not found in any of {CANDIDATES}")

    df = pd.read_parquet(path)
    print(f"Loaded {path}: {len(df):,} windows x {len(df.columns):,} columns")

    diagnostics = build_feature_diagnostics(df)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "feature_diagnostics.csv"
    diagnostics.to_csv(csv_path)
    print(f"Wrote {csv_path}")

    parquet_path = out_dir / "feature_diagnostics.parquet"
    diagnostics.reset_index().to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path}")
