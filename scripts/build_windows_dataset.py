"""
build_windows_dataset.py

The P2 windowing pipeline entry point. This is the skeleton pass only:
window boundaries, task context, and the afforded/observed flags per
discrete modality -- deliberately NO behavioural feature columns yet.

Why the skeleton is built and checked alone first: every feature family
(typing, tap, gesture, coupling, motion/orientation) will sit on top of the
same window boundaries, the same task-context join, and the same afforded
flags. If any of those three things has a bug, it would silently corrupt
every feature family built on top of it, and a wrong-looking feature value
later would be ambiguous -- is the bug in the feature, or in the scaffold
underneath it? Getting the scaffold right and confirmed here means every
feature family added afterward only has to be checked against itself.

Depends on:
    corrections.py  -- pre-windowing correction pass (circular encoding,
                        causal baseline subtraction), must run before this
    metadata.py     -- task timeline, modality affordance, task-context stamping

Design decisions this encodes (confirmed in conversation, 2026-08):
  - Window length 15s, stride 7.5s (50% overlap), configurable -- doubles
    nb04's 7.5s-too-short finding, shorter than P1's 30s since P2 sessions
    run longer and more windows are wanted per session. Treated as a
    testable default, not final -- validate rate-percentile stability
    before locking in.
  - Row shape: flat, wide, one row per window. NaN for structurally-absent
    modalities. Explicit {modality}_afforded / {modality}_observed boolean
    columns per discrete modality, rather than a nested structure -- keeps
    this a single parquet table compatible with column-selective loading
    at scale, and trivial to split into per-modality feature subsets later
    by column-name prefix (what the per-modality embedder architecture
    needs).
  - Afforded is a per-window fact derived from the window's task context
    (via metadata.py's per-taskType affordance table), not from whether
    events actually occurred in that specific window -- that's what
    "observed" is for. A window can be afforded=True, observed=False
    (task allowed it, participant didn't do it -- hesitation, a real
    behavioural signal) or afforded=False, observed=True (should not
    normally happen; if it does, worth investigating as a labelling
    problem, not silently dropped).
  - Windows whose task context is unknown (inter-task gap, or before the
    first task) get NaN afforded flags, not False -- "unknown" and "known
    false" are different states and collapsing them would misrepresent
    inter-task gaps as tasks that structurally forbid a modality.
  - Straddling windows: median task duration (~7.5s) is shorter than the
    15s window, so most windows (70%) overlap more than one task instance
    -- straddling is the norm here, not an edge case. A blanket "exclude
    any straddling window" policy was tried and found to discard real,
    usable data: only windows whose overlapping tasks actually DISAGREE on
    a given modality's affordance are unsafe for that modality's features.
    So affordance-conflict is tracked per modality
    ({modality}_straddle_conflict), not as one blanket flag -- a window
    straddling two tasks that both afford typing is fine for typing
    features even though it "straddles". Feature-family modules should
    exclude on the per-modality conflict flag, not on the diagnostic-only
    task_boundary_straddle column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corrections import apply_corrections
from metadata import (
    MODALITY_KINDS,
    build_task_timeline,
    compute_modality_affordance,
    stamp_task_context,
)
from primitives.typing_features import aggregate_typing_features
from primitives.tap import aggregate_tap_features
from primitives.gesture import aggregate_gesture_features, build_classified_interactions
from primitives.coupling import aggregate_coupling_features
from primitives.motion import aggregate_motion_features

# --- window parameters -- see module docstring for rationale ---
WINDOW_LENGTH_S = 15.0
WINDOW_STRIDE_S = 7.5

SESSION_META_COLS = ["sessionId", "participantId", "deviceFamily"]


# =============================================================================
# Window boundary construction
# =============================================================================

def build_window_grid(
    raw_df: pd.DataFrame,
    window_length_s: float = WINDOW_LENGTH_S,
    stride_s: float = WINDOW_STRIDE_S,
) -> pd.DataFrame:
    """
    One row per window: sessionId + session metadata, window_index,
    window_start_s, window_end_s. Anchored on each session's own first
    event timestamp (t=0 at first event), stepping forward by stride_s
    until the next full window would exceed the session's last event --
    same anchoring convention P1 used (Section 4.3), so a session's last
    few seconds of data may not be covered by a full window rather than
    producing a short, non-comparable final window.
    """
    rows = []
    meta = raw_df.groupby("sessionId")[["participantId", "deviceFamily"]].first()
    extents = raw_df.groupby("sessionId")["tRelMs"].agg(t_min="min", t_max="max")

    for sid, ext in extents.iterrows():
        t0 = ext["t_min"] / 1000.0
        t_end_session = ext["t_max"] / 1000.0
        duration = t_end_session - t0

        if duration < window_length_s:
            # Session too short for even one full window -- skip, don't
            # pad or truncate. Worth surfacing rather than silently
            # producing a non-comparable short window.
            print(f"[windower] WARNING: session {sid} duration {duration:.1f}s "
                  f"< window length {window_length_s}s -- no windows produced.")
            continue

        n_windows = int(np.floor((duration - window_length_s) / stride_s)) + 1
        for i in range(n_windows):
            w_start = t0 + i * stride_s
            w_end = w_start + window_length_s
            rows.append({
                "sessionId": sid,
                "participantId": meta.loc[sid, "participantId"],
                "deviceFamily": meta.loc[sid, "deviceFamily"],
                "window_index": i,
                "window_start_s": w_start,
                "window_end_s": w_end,
            })

    grid = pd.DataFrame(rows)
    print(f"[windower] built {len(grid):,} windows across {grid['sessionId'].nunique()} sessions "
          f"({window_length_s}s length, {stride_s}s stride)")
    return grid


# =============================================================================
# Afforded flags (from task context + metadata's affordance table)
# =============================================================================

def attach_afforded_flags(windows: pd.DataFrame, timeline: pd.DataFrame,
                           affordance: pd.DataFrame) -> pd.DataFrame:
    """
    Stamps each window with its task context (via metadata.stamp_task_context,
    using the window's START time -- a window that starts inside task A and
    runs past its end is attributed to task A, not split), then joins the
    per-taskType affordance table to produce {modality}_afforded columns.

    Windows with no task context (inter-task gap, before first task) get
    NaN afforded flags rather than False -- see module docstring.
    """
    w = windows.copy()
    w["t_rel_s"] = w["window_start_s"]
    w = stamp_task_context(w, timeline, time_col="t_rel_s")

    for mod in MODALITY_KINDS:
        w[f"{mod}_afforded"] = w["taskType"].map(affordance[mod]).astype("boolean")
        # rows where taskType itself is NaN (no task context) automatically
        # come out NaN here too, via .map on a NaN key -- no extra handling
        # needed, but worth a comment since it's not obvious at a glance.

    return w.drop(columns=["t_rel_s"])


# =============================================================================
# Observed flags (did events of this modality's kinds actually occur
# inside this specific window)
# =============================================================================

def attach_observed_flags(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each window and each discrete modality, whether >=1 event of that
    modality's kinds fell inside [window_start_s, window_end_s) for that
    session. Vectorised per-session via searchsorted, same pattern as
    metadata.compute_modality_affordance -- avoids an O(n_windows) re-filter
    of the full raw event table.
    """
    w = windows.copy()

    for mod, kinds in MODALITY_KINDS.items():
        ev = raw_df.loc[raw_df["kind"].isin(kinds), ["sessionId", "tRelMs"]].copy()
        ev["t_s"] = ev["tRelMs"] / 1000.0

        observed = np.zeros(len(w), dtype=bool)
        for sid, idx in w.groupby("sessionId").groups.items():
            sess_ev = np.sort(ev.loc[ev["sessionId"] == sid, "t_s"].to_numpy())
            if len(sess_ev) == 0:
                continue
            starts = w.loc[idx, "window_start_s"].to_numpy()
            ends = w.loc[idx, "window_end_s"].to_numpy()
            pos = np.searchsorted(sess_ev, starts, side="left")
            hit = (pos < len(sess_ev)) & (sess_ev[np.clip(pos, 0, len(sess_ev) - 1)] < ends)
            observed[np.asarray(idx)] = hit

        w[f"{mod}_observed"] = observed

    return w


# =============================================================================
# Consistency check: afforded=False but observed=True should not normally
# happen -- surface it rather than let it pass silently.
# =============================================================================

def attach_straddle_flags(windows: pd.DataFrame, timeline: pd.DataFrame,
                           affordance: pd.DataFrame) -> pd.DataFrame:
    """
    Two flags per window, not one:

      task_boundary_straddle       -- diagnostic only: does this window's
                                       span overlap more than one task
                                       instance at all, regardless of
                                       whether that matters for anything.

      {modality}_straddle_conflict -- the one that actually matters:
                                       among the task instances this
                                       window overlaps, do they DISAGREE
                                       on whether this modality is
                                       afforded? If every overlapping task
                                       agrees (all afford it, or all don't),
                                       there's no conflict and the window's
                                       features for that modality are safe
                                       to compute even though it technically
                                       straddles a task boundary.

    This distinction matters a lot in practice: an earlier blanket
    "exclude every straddling window" pass found 70% of all windows
    straddle *some* task boundary (median task duration ~7.5s is shorter
    than the 15s window), but only ~52% of windows actually cross a
    TYPING-affordance conflict specifically -- the rest straddle two tasks
    that happen to agree on typing, so excluding them wastes real data for
    no reason. Feature-family modules should use the per-modality conflict
    flag, not the blanket one, to decide what to exclude.
    """
    w = windows.copy().reset_index(drop=True)
    w["task_boundary_straddle"] = (
        w["task_end_s"].notna() & (w["window_end_s"] > w["task_end_s"])
    )

    conflict_cols = {mod: np.full(len(w), np.nan) for mod in MODALITY_KINDS}

    for sid, g in w.groupby("sessionId"):
        tl = timeline.loc[timeline["sessionId"] == sid].sort_values("task_start_s")
        starts = tl["task_start_s"].to_numpy()
        ends = tl["task_end_s"].to_numpy()
        types = tl["taskType"].to_numpy()

        for pos, row in zip(g.index, g.itertuples()):
            w_start, w_end = row.window_start_s, row.window_end_s
            overlap_mask = (starts < w_end) & (ends > w_start)
            overlapping_types = types[overlap_mask]
            if len(overlapping_types) == 0:
                continue  # no task context at all -- leave NaN, same as afforded flags
            for mod in MODALITY_KINDS:
                vals = affordance.loc[overlapping_types, mod].to_numpy()
                vals = vals[~pd.isna(vals)]
                conflict = bool(len(np.unique(vals)) > 1) if len(vals) else False
                conflict_cols[mod][pos] = conflict

    for mod, arr in conflict_cols.items():
        w[f"{mod}_straddle_conflict"] = pd.array(arr, dtype="boolean")

    return w


def check_afforded_observed_consistency(windows: pd.DataFrame) -> None:
    for mod in MODALITY_KINDS:
        afforded_col = f"{mod}_afforded"
        observed_col = f"{mod}_observed"
        # Only check rows where afforded is a known False (not NaN) --
        # NaN-afforded windows (inter-task gaps) can legitimately have
        # observed=True or False either way.
        known_false = windows[afforded_col] == False  # noqa: E712 (nullable boolean)
        contradiction = known_false & windows[observed_col]
        n = int(contradiction.sum())
        if n > 0:
            conflict_share = windows.loc[contradiction, f"{mod}_straddle_conflict"].fillna(False).mean()
            print(f"[windower] NOTE: {n} windows have {mod}_afforded=False but "
                  f"{mod}_observed=True -- {conflict_share:.0%} of these have a real "
                  f"{mod}_straddle_conflict (window overlaps tasks that disagree on "
                  f"{mod} affordance), explaining the contradiction. Remainder, if "
                  f"any, warrants a closer look.")


# =============================================================================
# Public entry point
# =============================================================================

def build_skeleton(raw_df: pd.DataFrame,
                    window_length_s: float = WINDOW_LENGTH_S,
                    stride_s: float = WINDOW_STRIDE_S,
                    return_corrected: bool = False):
    """
    Runs the full skeleton pass: correction pass -> window grid -> task
    context + afforded flags -> observed flags -> consistency check.
    Returns one row per window with NO behavioural feature columns yet.

    If return_corrected=True, also returns the corrected raw event stream
    (apply_corrections() output) as a second value -- needed by
    build_full_dataset() below, which passes the CORRECTED stream (not the
    raw one) to every feature-family aggregator so downstream families can
    eventually be updated to consume payload_alpha_sin/cos,
    payload_beta_residual, etc. NOTE (2026-08): as of this writing, none of
    the five feature-family modules actually read those corrected columns
    yet -- they all still compute from raw payload_alpha/beta/gamma
    directly. Passing the corrected stream through is necessary groundwork
    but not sufficient; each family needs a deliberate decision about
    which of its features should switch to the corrected columns and
    which shouldn't (see build_full_dataset()'s docstring for specifics).
    """
    corrected = apply_corrections(raw_df)
    timeline = build_task_timeline(raw_df)
    affordance = compute_modality_affordance(raw_df, timeline)

    windows = build_window_grid(raw_df, window_length_s, stride_s)
    windows = attach_afforded_flags(windows, timeline, affordance)
    windows = attach_observed_flags(windows, raw_df)
    windows = attach_straddle_flags(windows, timeline, affordance)
    check_afforded_observed_consistency(windows)

    if return_corrected:
        return windows, corrected
    return windows


def build_full_dataset(raw_df: pd.DataFrame,
                        window_length_s: float = WINDOW_LENGTH_S,
                        stride_s: float = WINDOW_STRIDE_S) -> pd.DataFrame:
    """
    The actual pipeline entry point: skeleton + all five feature families,
    assembled into one windows table. This is what run_pipeline.sh's
    build_windows_dataset.py step should call and save.

    CORRECTED-STREAM USAGE (2026-08, open decision, not yet resolved):
    build_skeleton() returns the corrected event stream (payload_alpha_sin/
    cos, payload_beta_residual/baseline, orient_baseline_coverage_frac)
    alongside the windows grid, but every feature-family aggregator below
    is still called with the RAW stream, matching what each family's
    self-check has been validated against so far. Switching any family to
    consume the corrected stream instead is a deliberate per-family
    decision, not a blanket switch -- for instance:
      - motion.py's cross-axis correlation is explicitly diff-based and
        wrap-insensitive by construction (see its orientation_series()
        docstring), so it likely does NOT need the corrected columns.
      - typing.py/tap.py/gesture.py/coupling.py don't touch alpha/beta/
        gamma at all except coupling.py's orientation-coupling functions,
        which use their OWN short local baseline (a different question
        from corrections.py's 30s causal baseline -- see coupling.py's
        docstring) and so also likely don't need corrections.py's output.
    So there may be no live gap in practice -- but this hasn't been
    checked family-by-family, and the corrected stream is threaded through
    here so that check can happen without re-plumbing later.
    """
    windows, corrected = build_skeleton(raw_df, window_length_s, stride_s, return_corrected=True)

    print("\n[assembly] typing features...")
    windows = aggregate_typing_features(windows, raw_df)

    print("[assembly] tap features...")
    windows = aggregate_tap_features(windows, raw_df)

    print("[assembly] gesture features...")
    windows = aggregate_gesture_features(windows, raw_df)
    interactions = build_classified_interactions(raw_df)

    print("[assembly] coupling features...")
    windows = aggregate_coupling_features(windows, raw_df, interactions)

    print("[assembly] motion features...")
    windows = aggregate_motion_features(windows, raw_df)

    n_feature_cols = len(windows.columns) - 22  # 22 = skeleton/metadata column count
    print(f"\n[assembly] done: {len(windows):,} windows x {len(windows.columns):,} columns "
          f"({n_feature_cols:,} behavioural feature columns)")

    return windows


if __name__ == "__main__":
    import sys

    RAW_CANDIDATES = [
        "data/processed/raw_events.parquet",
        "raw_events.parquet",
        "/mnt/user-data/uploads/raw_events.parquet",
    ]
    raw_path = next((p for p in RAW_CANDIDATES if Path(p).exists()), None)
    if raw_path is None:
        print("No raw_events.parquet found -- skipping.")
        sys.exit(0)

    # Column-selective load -- the full 181-column export costs ~1.7GB in
    # pandas at this session count (documented project finding), and the
    # assembly step below builds several intermediate per-event tables on
    # top of that, which pushed a full-column load over this sandbox's
    # memory ceiling. Only the columns actually referenced anywhere in the
    # pipeline (corrections.py + metadata.py + build_windows_dataset.py +
    # every primitives/*.py file) are loaded.
    NEEDED_PAYLOAD_COLS = [
        "payload_alpha", "payload_ax", "payload_ay", "payload_az", "payload_beta",
        "payload_componentId", "payload_force", "payload_gamma", "payload_keyClass",
        "payload_radiusX", "payload_radiusY", "payload_repeat", "payload_scrollTop",
        "payload_taskType", "payload_x", "payload_xNorm", "payload_y", "payload_yNorm",
    ]
    NEEDED_COLS = [
        "sessionId", "participantId", "deviceFamily", "kind", "tRelMs",
        "taskId", "taskIndex", "activeArea",
    ] + NEEDED_PAYLOAD_COLS

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(raw_path).schema_arrow.names)
    cols = [c for c in NEEDED_COLS if c in available]
    missing = set(NEEDED_COLS) - available
    if missing:
        print(f"[assembly] WARNING: columns referenced in code but absent from this "
              f"parquet's schema: {sorted(missing)} -- affected features will be all-NaN, "
              f"not an error, but worth checking if unexpected.")

    raw = pd.read_parquet(raw_path, columns=cols)
    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions, "
          f"{raw['participantId'].nunique()} participants "
          f"({len(cols)} of {len(available)} available columns).")

    full_windows = build_full_dataset(raw)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "windows.parquet"
    full_windows.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({full_windows.shape[0]:,} rows x {full_windows.shape[1]:,} columns)")

    # CSV export for human viewing (Excel, quick grep, etc.) -- not needed by
    # any downstream pipeline stage, which all read the parquet. Respects
    # --skip-csv, matching build_raw_dataset.py's existing flag, which
    # run_pipeline.sh already passes through to this script but which was
    # never actually parsed until now.
    if "--skip-csv" not in sys.argv:
        csv_path = out_dir / "windows.csv"
        full_windows.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print("Skipped CSV export (--skip-csv)")
