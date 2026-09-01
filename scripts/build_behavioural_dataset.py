"""
scripts/build_behavioural_dataset.py

Filters windows.parquet down to the actual model-input feature set, driven
by feature_diagnostics.csv's screening results rather than hand-picked
prefix rules (unlike P1's build_behavioural_dataset.py, which classified
columns by prefix allow-list; P2 has the per-feature diagnostic table
instead, so filtering reads decisions off it directly).

Every column in windows.parquet is sorted into exactly one of three roles,
same three-way split as P1 Section 4.4:

  1. IDENTITY/METADATA -- kept in the output, never fed to a model as a
     behavioural signal. sessionId/participantId/deviceFamily/deviceModel,
     timing/task columns, and -- deliberately -- the *_afforded/*_observed/
     *_straddle_conflict flags. These flags are NOT dropped even though
     they aren't behavioural: the modelling code needs them to do
     per-modality eligibility gating (train/score each embedder only on
     windows where ITS modality was actually afforded and conflict-free),
     not blanket imputation across the eligible/ineligible boundary.
     ctx* context columns are also kept here for now -- deliberately NOT
     excluded, per the P2 design goal (context-as-fusion-weight, not pure
     metadata the way P1 treated it) -- but also not yet a resolved
     decision, so they ride along as metadata pending that design work.

  2. DROPPED -- driven directly off feature_diagnostics.csv:
       - fully-empty / near-empty (no real signal)
       - count-like (raw tallies -- leaks exposure, not behaviour)
       - high exposure-correlation (tracks how much data a participant
         contributed, not how they behaved)
       - device-locked (device/hardware fingerprint, not behaviour)
       - exact duplicate of a surviving feature (adds no information)

  3. BEHAVIOURAL CANDIDATE -- everything left. The actual model inputs.

OUTPUT
------
data/processed/behavioural_windows_candidate.parquet / .csv
    Identity/metadata columns + surviving behavioural feature columns only.
data/processed/feature_diagnostics.csv / .parquet
    The SAME diagnostics table, with two columns bolted on: `decision`
    (keep/drop) and `decision_reason` (which rule caught it, or
    "duplicate_of:<feature>", or "kept" if it survived everything).
    Overwrites the diagnostics file in place rather than writing a
    separate manifest, per direct request -- one file, one place to look.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Config
# =============================================================================

# Columns kept as identity/metadata in the output -- never behavioural
# candidates, but needed downstream (grouping, eligibility gating,
# chronological splitting). Built from the known skeleton/context columns
# rather than "everything not flagged as a feature", so a genuinely new
# column added upstream shows up as neither metadata nor a feature and gets
# caught by the consistency check in main(), instead of silently vanishing
# into one bucket or the other.
IDENTITY_COLS = [
    "sessionId", "participantId", "deviceFamily",
    "window_index", "window_start_s", "window_end_s",
    "taskIndex", "taskType", "activeArea", "task_pass",
    "task_start_s", "task_end_s",
    "n_active_modalities",  # present if windows.parquet was touched by the
                             # exploratory notebook's column; harmless if absent
]

ELIGIBILITY_FLAG_COLS = [
    "typing_afforded", "tapping_afforded", "scrolling_afforded",
    "typing_observed", "tapping_observed", "scrolling_observed",
    "task_boundary_straddle",
    "typing_straddle_conflict", "tapping_straddle_conflict", "scrolling_straddle_conflict",
]

CONTEXT_COLS = [
    "deviceModel", "devicePlatform",
    "appVersion", "consentVersion", "schemaVersion",
    "identitySource", "usableForSignalExtraction", "completedNormally",
    "sessionIndex", "sessionDurationMs",
    "ctxTimeOfDay", "ctxFatigue", "ctxFocusLevel", "ctxEnvironmentNoise",
    "ctxMovement", "ctxPosture", "ctxHandUse", "ctxInputDevice",
    "ctxCaffeine", "ctxAlcohol", "ctxPrivacy",
]

# Priority order for decision_reason when a feature matches more than one
# exclusion rule -- report the first that applies. Fully-empty is checked
# first as the most decisive, simplest reason; duplicate-of is checked LAST
# and only among features that already survived every other rule (see
# resolve_duplicates()), since flagging a dedup reason on a column already
# being dropped for being empty would be noise, not information.
EXCLUSION_PRIORITY = [
    ("is_fully_empty", "fully_empty"),
    ("is_near_empty", "near_empty"),
    ("is_count_like", "count_like"),
    ("high_exposure_corr", "high_exposure_corr"),
    ("is_device_locked", "device_locked"),
]


# =============================================================================
# Decision logic
# =============================================================================

def apply_diagnostic_exclusions(diag: pd.DataFrame) -> pd.DataFrame:
    """First pass: decide keep/drop from feature_diagnostics.csv's own
    columns, before the duplicate check (which needs to know what's already
    surviving)."""
    diag = diag.copy()
    diag["decision"] = "keep"
    diag["decision_reason"] = "kept"

    for flag_col, reason in EXCLUSION_PRIORITY:
        still_kept = diag["decision"] == "keep"
        hit = still_kept & diag[flag_col].fillna(False)
        diag.loc[hit, "decision"] = "drop"
        diag.loc[hit, "decision_reason"] = reason

    return diag


def resolve_duplicates(diag: pd.DataFrame) -> pd.DataFrame:
    """
    Second pass: among features that survived every other rule, drop one
    side of each MUTUAL exact-duplicate pair (max_abs_corr == 1.0 AND each
    is the other's best match -- not just "somewhere in the top of A's
    list", which could be a one-directional near-tie rather than a true
    duplicate).

    Tie-break for which side survives: higher non_null_pct_overall (more
    data support), then alphabetically first feature name for a
    deterministic result when support is equal (e.g. tap_hold_ms_* vs
    gesture_hold_ms_*, which share an identical eligibility gate in the
    code and so have near-identical support -- tap_hold_ms wins
    alphabetically, which also happens to be the less-derived measurement:
    gesture_hold_ms depends on an extra classification step tap_hold_ms
    doesn't).
    """
    diag = diag.copy()
    survivors = diag[diag["decision"] == "keep"].set_index("feature")
    partner_of = survivors["max_abs_corr_with"]

    resolved = set()
    for feat in survivors.index:
        if feat in resolved:
            continue
        if survivors.at[feat, "max_abs_corr"] != 1.0:
            continue
        partner_feat = partner_of.get(feat)
        if partner_feat is None or partner_feat not in survivors.index:
            continue
        # Mutual check: partner's own best match must point back to feat.
        if partner_of.get(partner_feat) != feat:
            continue

        a_support = survivors.at[feat, "non_null_pct_overall"]
        b_support = survivors.at[partner_feat, "non_null_pct_overall"]
        if a_support > b_support:
            winner, loser = feat, partner_feat
        elif b_support > a_support:
            winner, loser = partner_feat, feat
        else:
            winner, loser = sorted([feat, partner_feat])

        diag.loc[diag["feature"] == loser, "decision"] = "drop"
        diag.loc[diag["feature"] == loser, "decision_reason"] = f"duplicate_of:{winner}"
        resolved.add(feat)
        resolved.add(partner_feat)

    return diag


def resolve_duplicates_of_dropped(diag: pd.DataFrame) -> pd.DataFrame:
    """
    Third pass: a feature can survive resolve_duplicates() above (which only
    compares pairs where BOTH sides are still 'keep') while still being an
    exact duplicate of something dropped for an unrelated reason -- e.g.
    tap_coverage (a percentage) is a fixed linear transform of
    tap_cells_touched (cells_touched / 64, a constant denominator), so it's
    ALGEBRAICALLY redundant with a raw count, not coincidentally correlated.
    tap_cells_touched gets dropped as count_like; tap_coverage's name
    doesn't match the count-pattern rules, so without this pass it would
    survive despite encoding the same "how many different areas touched"
    information the count_like rule exists to exclude.

    Catches any remaining 'keep' feature whose exact (r=1.0) match is a
    MUTUAL partner (the dropped feature's own best match points back to
    this one) that is already 'drop' for any reason, and drops it too.
    """
    diag = diag.copy()
    by_feature = diag.set_index("feature")

    for idx, row in diag.iterrows():
        if row["decision"] != "keep":
            continue
        if row["max_abs_corr"] != 1.0:
            continue
        partner_feat = row["max_abs_corr_with"]
        if partner_feat not in by_feature.index:
            continue
        partner_row = by_feature.loc[partner_feat]
        if partner_row["decision"] != "drop":
            continue
        if str(partner_row["decision_reason"]).startswith("duplicate_of"):
            # Partner was dropped BECAUSE it lost to something in the
            # resolve_duplicates() head-to-head -- possibly this very
            # feature. Chaining off that would drop both sides of a pair
            # instead of keeping the winner (confirmed: without this guard,
            # gesture_hold_ms_* -- the winner over tap_hold_ms_* -- got
            # dropped too, because tap_hold_ms_* was marked drop for being
            # the loser). Only chain off PRIMARY drop reasons below.
            continue
        if partner_row["max_abs_corr_with"] != row["feature"]:
            continue  # not mutual -- partner's own best match isn't this feature

        diag.at[idx, "decision"] = "drop"
        diag.at[idx, "decision_reason"] = f"duplicate_of_dropped:{partner_feat}:{partner_row['decision_reason']}"

    return diag


def build_behavioural_dataset(windows: pd.DataFrame, diag: pd.DataFrame):
    diag = apply_diagnostic_exclusions(diag)
    diag = resolve_duplicates(diag)
    diag = resolve_duplicates_of_dropped(diag)

    # Consistency check -- every feature column in windows.parquet should
    # appear in the diagnostics table, and every identity/context/flag
    # column named above should actually exist. A genuinely new column
    # added upstream that fits neither bucket would otherwise silently
    # vanish rather than being caught here.
    known_non_feature = set(IDENTITY_COLS) | set(ELIGIBILITY_FLAG_COLS) | set(CONTEXT_COLS)
    diag_features = set(diag["feature"])
    window_cols = set(windows.columns)

    unaccounted = window_cols - diag_features - known_non_feature
    if unaccounted:
        print(f"[behavioural] WARNING: {len(unaccounted)} windows.parquet column(s) are neither "
              f"in feature_diagnostics.csv nor in the known identity/context/flag lists, and "
              f"will be silently excluded from the output: {sorted(unaccounted)}")

    missing_expected = known_non_feature - window_cols
    if missing_expected:
        print(f"[behavioural] NOTE: {len(missing_expected)} expected identity/context/flag "
              f"column(s) not present in this windows.parquet, skipped: {sorted(missing_expected)}")

    kept_features = diag.loc[diag["decision"] == "keep", "feature"].tolist()
    output_cols = (
        [c for c in IDENTITY_COLS if c in windows.columns]
        + [c for c in ELIGIBILITY_FLAG_COLS if c in windows.columns]
        + [c for c in CONTEXT_COLS if c in windows.columns]
        + [c for c in kept_features if c in windows.columns]
    )
    candidate = windows[output_cols].copy()

    n_dropped = (diag["decision"] == "drop").sum()
    print(f"[behavioural] {len(diag)} behavioural features scanned: "
          f"{len(kept_features)} kept, {n_dropped} dropped")
    print(diag.loc[diag["decision"] == "drop", "decision_reason"]
          .apply(lambda r: r.split(":")[0]).value_counts().to_string())

    return candidate, diag


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    WINDOWS_CANDIDATES = [
        "data/processed/windows.parquet", "windows.parquet",
        "/mnt/user-data/uploads/windows.parquet",
    ]
    DIAG_CANDIDATES = [
        "data/processed/feature_diagnostics.parquet", "feature_diagnostics.parquet",
        "/mnt/user-data/uploads/feature_diagnostics.parquet",
    ]
    windows_path = next((p for p in WINDOWS_CANDIDATES if Path(p).exists()), None)
    diag_path = next((p for p in DIAG_CANDIDATES if Path(p).exists()), None)
    if windows_path is None:
        print("No windows.parquet found -- run build_windows_dataset.py first.")
        sys.exit(0)
    if diag_path is None:
        print("No feature_diagnostics.parquet found -- run build_feature_diagnostics.py first.")
        sys.exit(0)

    windows = pd.read_parquet(windows_path)
    diag = pd.read_parquet(diag_path)
    print(f"Loaded {windows_path}: {len(windows):,} windows x {len(windows.columns):,} columns")
    print(f"Loaded {diag_path}: {len(diag):,} features scanned")

    candidate, diag_out = build_behavioural_dataset(windows, diag)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_parquet = out_dir / "behavioural_windows_candidate.parquet"
    candidate.to_parquet(cand_parquet, index=False)
    print(f"\nWrote {cand_parquet} ({candidate.shape[0]:,} rows x {candidate.shape[1]:,} columns)")

    if "--skip-csv" not in sys.argv:
        cand_csv = out_dir / "behavioural_windows_candidate.csv"
        candidate.to_csv(cand_csv, index=False)
        print(f"Wrote {cand_csv} ({cand_csv.stat().st_size / 1e6:.1f} MB)")

    # Overwrite feature_diagnostics.csv/.parquet in place with the two new
    # decision columns bolted on, per direct request -- one file, not a
    # separate manifest.
    diag_out.to_parquet(out_dir / "feature_diagnostics.parquet", index=False)
    diag_out.to_csv(out_dir / "feature_diagnostics.csv", index=False)
    print(f"Updated {out_dir / 'feature_diagnostics.csv'} / .parquet with decision/decision_reason columns")
