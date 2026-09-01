"""
scripts/modelling/build_modality_datasets.py

Splits behavioural_windows_candidate.parquet into four per-modality tables
(typing, tap, gesture, motion) -- the first piece of the fusion architecture
discussed in project conversation: separate embedders per modality, each
trained/scored only on windows where its own modality was actually eligible,
rather than one wide vector with imputed gaps across the eligibility
boundary (P1 already established missingness is often informative, not
incidental to be silently filled in).

COUPLING FOLD-IN: coupling_* is not a sixth independent modality -- it's
derived from tap/keystroke/gesture co-occurring with accelerometer/
orientation, and coupling.py itself already gates each coupling sub-family
on the SAME eligibility flags as its parent modality. Confirmed by direct
inspection: exactly three coupling sub-prefixes exist --
coupling_key_accel_*/coupling_key_orient_* (typing), coupling_tap_accel_*/
coupling_tap_orient_* (tap), coupling_gesture_mid_*/coupling_gesture_peak_*
(gesture) -- folded into their parent modality's table here rather than
kept as a separate block.

ELIGIBILITY GATES (confirmed by reading tap.py/gesture.py/coupling.py's own
eligibility masks, not assumed from column names):
  typing  : typing_afforded  & ~typing_straddle_conflict
  tap     : tapping_afforded & ~tapping_straddle_conflict
  gesture : tapping_afforded & ~tapping_straddle_conflict  (SAME gate as
            tap -- gesture.py's own eligible mask uses tapping_afforded,
            not a separate gesture-specific flag; scrolling_afforded exists
            as a column but is not actually used as a gate anywhere in the
            feature-computation code, confirmed by direct inspection)
  motion  : no gate -- unconditional/passive family, kept whenever any
            motion data exists (motion_idle_frac notna as the presence
            check, since that column is populated whenever the motion
            aggregation had enough samples to compute anything at all)

Windows that are NOT eligible for a given modality are DROPPED from that
modality's table entirely, not kept with NaN feature columns -- this is
the actual masking: an embedder never sees a row where its own modality
had nothing to say.

OUTPUT
------
data/processed/modelling/typing_windows.parquet
data/processed/modelling/tap_windows.parquet
data/processed/modelling/gesture_windows.parquet
data/processed/modelling/motion_windows.parquet

Each keeps the identity columns (sessionId, participantId, deviceFamily,
window timing/task columns) plus only that modality's own feature columns
(family's own + folded-in coupling sub-family). No cross-modality columns,
no eligibility flags from OTHER modalities, no context columns (deferred,
per project decision to set context aside for now).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================================================================
# Config
# =============================================================================

IDENTITY_COLS = [
    "sessionId", "participantId", "deviceFamily",
    "window_index", "window_start_s", "window_end_s",
    "taskIndex", "taskType", "activeArea", "task_pass",
    "task_start_s", "task_end_s",
]

# Each modality's own feature prefix(es), the coupling sub-prefix(es) folded
# in, and the eligibility gate (afforded_col, straddle_col) -- straddle_col
# is None for motion, which has no gate at all.
MODALITY_CONFIG = {
    "typing": {
        "feature_prefixes": ("typing_",),
        "coupling_prefixes": ("coupling_key_",),
        "afforded_col": "typing_afforded",
        "straddle_col": "typing_straddle_conflict",
    },
    "tap": {
        "feature_prefixes": ("tap_",),
        "coupling_prefixes": ("coupling_tap_",),
        "afforded_col": "tapping_afforded",
        "straddle_col": "tapping_straddle_conflict",
    },
    "gesture": {
        "feature_prefixes": ("gesture_",),
        "coupling_prefixes": ("coupling_gesture_",),
        "afforded_col": "tapping_afforded",  # same gate as tap -- see module docstring
        "straddle_col": "tapping_straddle_conflict",
    },
    "motion": {
        "feature_prefixes": ("motion_",),
        "coupling_prefixes": (),
        "afforded_col": None,
        "straddle_col": None,
        "presence_col": "motion_idle_frac",
    },
}

# Columns that share a modality's feature prefix by coincidence but are NOT
# feature columns -- the eligibility flags themselves. Excluded from the
# per-modality feature list; the gate is applied separately, not carried
# into the output as a "feature".
NON_FEATURE_COLS_SHARING_PREFIX = {
    "typing_afforded", "typing_observed", "typing_straddle_conflict",
}


# =============================================================================
# Per-modality extraction
# =============================================================================

def build_modality_table(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    cfg = MODALITY_CONFIG[modality]

    feature_cols = [
        c for c in df.columns
        if c.startswith(cfg["feature_prefixes"] + cfg["coupling_prefixes"])
        and c not in NON_FEATURE_COLS_SHARING_PREFIX
    ]
    if not feature_cols:
        raise ValueError(f"No feature columns found for modality '{modality}' -- "
                          f"check MODALITY_CONFIG prefixes against the input columns.")

    if cfg["afforded_col"] is not None:
        missing_gate_cols = [c for c in (cfg["afforded_col"], cfg["straddle_col"]) if c not in df.columns]
        if missing_gate_cols:
            raise ValueError(f"Modality '{modality}' eligibility gate column(s) missing from "
                              f"input: {missing_gate_cols}")
        eligible = (
            (df[cfg["afforded_col"]] == True)  # noqa: E712
            & (~df[cfg["straddle_col"]].fillna(False))
        )
    else:
        presence_col = cfg["presence_col"]
        if presence_col not in df.columns:
            raise ValueError(f"Modality '{modality}' presence column '{presence_col}' "
                              f"missing from input.")
        eligible = df[presence_col].notna()

    id_cols_present = [c for c in IDENTITY_COLS if c in df.columns]
    table = df.loc[eligible, id_cols_present + feature_cols].reset_index(drop=True)

    n_total = len(df)
    n_eligible = eligible.sum()
    print(f"[{modality:8s}] {len(feature_cols):3d} feature columns "
          f"({len(cfg['coupling_prefixes'])} coupling sub-prefix(es) folded in), "
          f"{n_eligible:,} of {n_total:,} windows eligible ({n_eligible / n_total:.1%})")

    return table


def build_all_modality_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {modality: build_modality_table(df, modality) for modality in MODALITY_CONFIG}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    CANDIDATES = [
        "data/processed/behavioural_windows_candidate.parquet",
        "../data/processed/behavioural_windows_candidate.parquet",
        "behavioural_windows_candidate.parquet",
        "/mnt/user-data/uploads/behavioural_windows_candidate.parquet",
    ]
    path = next((p for p in CANDIDATES if Path(p).exists()), None)
    if path is None:
        print("No behavioural_windows_candidate.parquet found -- "
              "run build_behavioural_dataset.py first.")
        sys.exit(0)

    df = pd.read_parquet(path)
    print(f"Loaded {path}: {len(df):,} windows x {len(df.columns):,} columns\n")

    tables = build_all_modality_tables(df)

    out_dir = Path("data/processed/modelling")
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    for modality, table in tables.items():
        out_path = out_dir / f"{modality}_windows.parquet"
        table.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({table.shape[0]:,} rows x {table.shape[1]:,} columns)")

        if "--skip-csv" not in sys.argv:
            csv_path = out_dir / f"{modality}_windows.csv"
            table.to_csv(csv_path, index=False)
