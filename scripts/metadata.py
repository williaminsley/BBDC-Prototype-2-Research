"""
metadata.py

Task timeline, modality-affordance, and task-context-stamping layer for the
P2 windowing pipeline. This is what makes the afforded/observed flag pattern
possible: before a window can be marked "typing afforded, but nothing typed"
vs. "typing not afforded here at all", something has to know which task was
running at that point in time and which modalities that task type can
structurally produce events for.

Two notebooks had independently built pieces of this and had drifted apart:

  - 03_raw_time_series_explorer_v6.ipynb built the more complete task
    timeline (uses actual task_end events rather than inferring an end time
    from the next task's start, carries taskIndex/activeArea/participantId/
    deviceFamily, verifies the task sequence is identical across sessions,
    and derives task_pass -- which occurrence of a repeated task type this
    is, e.g. pass 1 vs pass 2 of a repeated task).
  - 04_behavioural_rhythm.ipynb built a simpler timeline (end-time inferred
    from the next task_start) plus the MODALITY_KINDS affordance concept,
    but its affordance_map() re-filters the full raw event table inside a
    per-row Python loop -- fine for a one-off notebook cell, not something
    to run inside a windowing pipeline at the full session count.

This module keeps 03's timeline (strictly better -- real task_end events,
task_pass, sequence verification) and 04's affordance concept, rewritten to
be vectorised instead of row-by-row.

Public entry points:
    build_task_timeline(raw_df)            -> one row per task instance
    compute_modality_affordance(raw_df, timeline, threshold=0.5)
                                            -> per-taskType boolean affordance table
    stamp_task_context(obs_df, timeline)   -> as-of join, task context onto any
                                               time-referenced table (windows included)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TASK_START_KIND = "task_start"
TASK_END_KIND = "task_end"

# Event kinds that structurally belong to each discrete, mutually-exclusive
# foreground modality. Ported from 04's MODALITY_KINDS. Note 'tapping' here
# covers all touchstart events, including the ones that later get
# fine-classified into gesture subtypes (drag_swipe, long_press, etc. --
# see 06_gesture_morphology's classify()) -- that finer classification is a
# downstream sub-typing of the touch stream, not a separate structural
# affordance at the task-UI level, so it isn't a fourth entry here.
MODALITY_KINDS = {
    "typing": ["keydown"],
    "tapping": ["touchstart"],
    "scrolling": ["scroll", "window_scroll", "carousel_scroll"],
}

# A taskType is classified as structurally affording a modality if at least
# this fraction of its instances (across the whole cohort) produced >=1
# event of that modality's kinds.
#
# DELIBERATELY LOW, not a majority threshold. "Afforded" is meant to answer
# a fact about the task's design/UI -- can this modality physically happen
# here -- not a fact about population behaviour -- do most people choose to
# do it. Those are different questions, and a 0.5 (majority) cut silently
# answers the second one while claiming to answer the first.
#
# Concretely: if a task is scrollable but only some participants scroll
# (short content, personal habit, whatever), the individual difference
# between scrollers and non-scrollers on an afforded task IS the
# behavioural signal worth learning -- exactly the kind of person-specific
# variation this project cares about. Calling that task "not afforded"
# because under half the cohort happened to scroll would mark every
# scroller's window as a structural anomaly instead of a valid behavioural
# choice, discarding the signal rather than preserving it for the model to
# learn from. It's also a subtler version of the exposure-confound problem
# already flagged for features: computing a structural label from
# population-average behaviour lets population patterns leak into what
# should be an individual signal, just one layer up at the metadata stage.
#
# So the bar here is only "does this occur often enough to be a real,
# repeatable possibility rather than a one-off logging fluke or mislabelled
# event" -- not "do most people do it." 0.05 (1 in 20 instances) draws that
# line generously toward keeping real-but-uncommon behaviour classified as
# afforded, consistent with the maximalist "generate everything, filter
# later" principle used everywhere else in this pipeline.
AFFORDANCE_THRESHOLD = 0.05


# =============================================================================
# Task timeline (ported from 03, unchanged in substance)
# =============================================================================

def build_task_timeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per task instance, with real start/end times from task_start/
    task_end events (not inferred from the next task's start), plus
    task_pass (1st vs 2nd+ occurrence of that taskType within the session).

    Also verifies the task sequence is identical across every session in
    raw_df and prints a warning (does not raise) if it isn't -- carried over
    from 03, since task-index comparisons downstream silently stop being
    like-for-like if the sequence varies, and that's worth surfacing loudly
    rather than failing quietly.
    """
    ts = raw_df.loc[
        raw_df["kind"] == TASK_START_KIND,
        ["sessionId", "participantId", "deviceFamily", "taskIndex", "taskId",
         "payload_taskType", "activeArea", "tRelMs"],
    ].copy()
    te = raw_df.loc[raw_df["kind"] == TASK_END_KIND, ["sessionId", "taskIndex", "tRelMs"]].copy()

    ts = ts.rename(columns={"payload_taskType": "taskType", "tRelMs": "task_start_ms"})
    te = te.rename(columns={"tRelMs": "task_end_ms"})

    timeline = ts.merge(te, on=["sessionId", "taskIndex"], how="left")
    timeline["task_start_s"] = timeline["task_start_ms"] / 1000.0
    timeline["task_end_s"] = timeline["task_end_ms"] / 1000.0
    timeline["task_duration_s"] = timeline["task_end_s"] - timeline["task_start_s"]
    timeline["taskIndex"] = pd.to_numeric(timeline["taskIndex"], errors="coerce")
    timeline = timeline.sort_values(["sessionId", "task_start_s"]).reset_index(drop=True)

    # --- sequence verification (03's canonical-sequence check) ---
    seqs = (
        timeline.sort_values("taskIndex")
        .groupby("sessionId")["taskType"]
        .apply(lambda s: tuple(s.astype(str)))
    )
    distinct = seqs.drop_duplicates()
    if len(distinct) == 1:
        print(f"[metadata] VERIFIED: all {len(seqs)} sessions share one identical "
              f"{len(distinct.iloc[0])}-task sequence.")
    else:
        print(f"[metadata] WARNING: {len(distinct)} distinct task sequences across "
              f"{len(seqs)} sessions -- task-index comparisons downstream are NOT "
              f"like-for-like. Investigate before trusting task-index-conditioned "
              f"features across sessions.")

    # --- task_pass: which occurrence of this taskType within this session ---
    timeline["task_pass"] = (
        timeline.sort_values(["sessionId", "taskIndex"])
        .groupby(["sessionId", "taskType"])
        .cumcount() + 1
    )

    return timeline


# =============================================================================
# Modality affordance (04's concept, vectorised)
# =============================================================================

def compute_modality_affordance(
    raw_df: pd.DataFrame,
    timeline: pd.DataFrame,
    threshold: float = AFFORDANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    For each taskType and each discrete modality, compute the fraction of
    task instances that produced >=1 event of that modality's kinds, and
    threshold it into a boolean affordance call.

    Vectorised replacement for 04's affordance_map(): instead of re-filtering
    raw_df inside a per-instance Python loop, this does one pass per modality
    using an interval-membership join (searchsorted per session), which
    scales to the full session count instead of being a notebook-only
    convenience.

    Returns a DataFrame indexed by taskType, with one boolean column per
    modality plus each modality's raw hit-fraction (suffixed _frac), so the
    graded cases (04's "scrolling is graded" finding) stay inspectable
    rather than being silently swallowed by the threshold.
    """
    records = []
    for mod, kinds in MODALITY_KINDS.items():
        ev = raw_df.loc[raw_df["kind"].isin(kinds), ["sessionId", "tRelMs"]].copy()
        ev["t_s"] = ev["tRelMs"] / 1000.0
        ev = ev.sort_values(["sessionId", "t_s"])

        hits = np.zeros(len(timeline), dtype=bool)
        for sid, tl_idx in timeline.groupby("sessionId").groups.items():
            sess_ev = ev.loc[ev["sessionId"] == sid, "t_s"].to_numpy()
            if len(sess_ev) == 0:
                continue
            sess_ev = np.sort(sess_ev)
            starts = timeline.loc[tl_idx, "task_start_s"].to_numpy()
            ends = timeline.loc[tl_idx, "task_end_s"].to_numpy()
            # For each task instance, does at least one event timestamp fall
            # in [start, end)? searchsorted gives us the insertion point for
            # `start`; if the event at that index exists and is < end, hit.
            pos = np.searchsorted(sess_ev, starts, side="left")
            in_range = (pos < len(sess_ev)) & (sess_ev[np.clip(pos, 0, len(sess_ev) - 1)] < ends)
            hits[np.asarray(tl_idx)] = in_range

        records.append(pd.Series(hits, index=timeline.index, name=mod))

    hit_df = pd.concat(records, axis=1)
    hit_df["taskType"] = timeline["taskType"].to_numpy()

    frac = hit_df.groupby("taskType")[list(MODALITY_KINDS.keys())].mean()
    frac.columns = [f"{c}_frac" for c in frac.columns]

    afforded = (frac >= threshold)
    afforded.columns = list(MODALITY_KINDS.keys())

    out = pd.concat([afforded, frac], axis=1)
    n_instances = timeline.groupby("taskType").size().rename("n_instances")
    out = out.join(n_instances)
    return out.sort_values("typing", ascending=False)


# =============================================================================
# Task-context stamping (ported from 03, unchanged in substance)
# =============================================================================

def stamp_task_context(obs_df: pd.DataFrame, timeline: pd.DataFrame,
                        time_col: str = "t_rel_s") -> pd.DataFrame:
    """
    As-of merge: assign each time-referenced row (a raw event, or a window's
    representative timestamp) the task instance running at that elapsed
    time, per session. Applied uniformly regardless of what obs_df actually
    contains, so no extractor or windower needs to carry its own task-lookup
    logic -- ported directly from 03, where this discipline was established
    ("no extractor carries its own task logic").

    obs_df must have `sessionId` and a numeric elapsed-seconds column named
    `time_col` (default 't_rel_s'; for windows, pass the window's start time
    or midpoint -- caller's choice, but be consistent).

    A row after its task's recorded end (i.e. in an inter-task gap) is left
    with taskIndex/taskType/task_pass set to NaN rather than attributed to
    the wrong task -- gaps are real and should be visible as such, not
    silently absorbed into whichever task happened to start most recently.
    """
    cols = ["taskIndex", "taskType", "activeArea", "task_pass", "task_start_s", "task_end_s"]
    pieces = []
    for sid, g in obs_df.sort_values(time_col).groupby("sessionId", sort=False):
        tl = timeline.loc[timeline["sessionId"] == sid, cols]
        if tl.empty:
            g = g.copy()
            for c in cols:
                g[c] = np.nan
            pieces.append(g)
            continue
        tl = tl.sort_values("task_start_s")
        merged = pd.merge_asof(
            g.reset_index(drop=True), tl.reset_index(drop=True),
            left_on=time_col, right_on="task_start_s", direction="backward",
        )
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True)
    past_end = out["task_end_s"].notna() & (out[time_col] > out["task_end_s"])
    out.loc[past_end, ["taskIndex", "taskType", "task_pass"]] = np.nan
    return out


# =============================================================================
# Self-check
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

    raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions, "
          f"{raw['participantId'].nunique()} participants.\n")

    timeline = build_task_timeline(raw)
    print(f"\n{len(timeline):,} task instances built.")
    print(f"task_pass value counts:\n{timeline['task_pass'].value_counts().sort_index()}\n")

    afford = compute_modality_affordance(raw, timeline)
    print("Modality affordance by taskType (boolean call + underlying hit-fraction):")
    print(afford.round(2).to_string())

    # Smoke-test stamp_task_context against the raw events themselves (using
    # each event's own elapsed time), and report what fraction land inside
    # a known task vs. an inter-task gap.
    probe = raw[["sessionId", "tRelMs"]].copy()
    probe["t_rel_s"] = probe["tRelMs"] / 1000.0
    stamped = stamp_task_context(probe.sample(min(20000, len(probe)), random_state=0), timeline)
    unstamped_frac = stamped["taskIndex"].isna().mean()
    print(f"\nstamp_task_context smoke test (20k sampled events): "
          f"{100 * (1 - unstamped_frac):.1f}% stamped to a task, "
          f"{100 * unstamped_frac:.1f}% fell in an inter-task gap or before the first task.")
