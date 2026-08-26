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
                    stride_s: float = WINDOW_STRIDE_S) -> pd.DataFrame:
    """
    Runs the full skeleton pass: correction pass -> window grid -> task
    context + afforded flags -> observed flags -> consistency check.
    Returns one row per window with NO behavioural feature columns yet.
    """
    corrected = apply_corrections(raw_df)  # not yet consumed by the skeleton
                                            # itself, but run here so the
                                            # corrected stream exists ready
                                            # for feature families to use
                                            # once they're layered in
    timeline = build_task_timeline(raw_df)
    affordance = compute_modality_affordance(raw_df, timeline)

    windows = build_window_grid(raw_df, window_length_s, stride_s)
    windows = attach_afforded_flags(windows, timeline, affordance)
    windows = attach_observed_flags(windows, raw_df)
    windows = attach_straddle_flags(windows, timeline, affordance)
    check_afforded_observed_consistency(windows)

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

    raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions, "
          f"{raw['participantId'].nunique()} participants.\n")

    skeleton = build_skeleton(raw)

    print(f"\nSkeleton shape: {skeleton.shape}")
    print(f"Columns: {list(skeleton.columns)}\n")

    print("Afforded-flag coverage (fraction of windows in each state):")
    for mod in MODALITY_KINDS:
        vc = skeleton[f"{mod}_afforded"].value_counts(dropna=False)
        print(f"  {mod}_afforded: {dict(vc)}")

    print(f"\nDiagnostic-only task_boundary_straddle (any overlap with >1 task): "
          f"{skeleton['task_boundary_straddle'].mean():.1%} of windows")

    print("\nPer-modality straddle_conflict (window overlaps tasks that DISAGREE on "
          "this modality's affordance -- this is what feature modules should actually "
          "exclude on, not the blanket flag above):")
    for mod in MODALITY_KINDS:
        vc = skeleton[f"{mod}_straddle_conflict"].value_counts(dropna=False)
        print(f"  {mod}_straddle_conflict: {dict(vc)}")

    print("\nObserved-flag rate (fraction of windows with >=1 event):")
    for mod in MODALITY_KINDS:
        print(f"  {mod}_observed: {skeleton[f'{mod}_observed'].mean():.1%}")

    print("\nSample rows:")
    print(skeleton.head(8).to_string())
