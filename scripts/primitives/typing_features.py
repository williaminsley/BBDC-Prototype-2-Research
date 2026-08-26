"""
scripts/primitives/typing_features.py

Typing feature family: event-level extraction ported from 04_behavioural_rhythm
(paired_durations -- dwell time) and 07_typing_structure (build_transitions,
backspace_runs, autorepeat), plus window-level aggregation.

Lives under scripts/primitives/ alongside the other feature-family modules
(tap.py, gesture.py, coupling.py, motion.py as they're built) -- separate
from the scaffold modules (corrections.py, metadata.py,
build_windows_dataset.py) that sit directly in scripts/, since those are
shared infrastructure every feature family depends on, while these are the
individual, independently-addable families themselves.

Ported functions are event/interval extractors (04's "already reusable"
category from the earlier mapping) -- they take a whole session and return a
table of timestamped events/intervals, which this module then re-scopes to
arbitrary window boundaries rather than the session-level summaries the
notebooks originally aggregated into.

Straddle handling (2026-08, refined): for any window flagged
typing_straddle_conflict=True by build_windows_dataset.py -- meaning the
window overlaps tasks that actually disagree on typing affordance -- typing
feature values are left NaN rather than computed from a mix of two tasks'
worth of keystrokes. This is the precise, per-modality version of the
straddle exclusion; a window that straddles a task boundary but whose
overlapping tasks AGREE on typing affordance is NOT excluded, since there's
nothing unsafe about the computation in that case. The window row itself is
kept either way -- only the feature values are withheld for genuinely
conflicting windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_DOWN_KIND = "keydown"
KEY_UP_KIND = "keyup"
MAX_TRANSITION_MS = 3000.0  # ported from 07 -- transitions slower than this
                             # aren't a single continuous typing act, exclude
                             # from transition-timing stats


# =============================================================================
# Event-level extractors (operate on one session's raw events, return a
# table of timestamped events/intervals -- these do NOT know about windows)
# =============================================================================

def extract_keydown_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    All keydown events across the whole corrected/raw stream, with the
    columns needed by every function below: sessionId, participantId,
    tRelMs, t_s, keyClass, is_repeat.
    """
    k = raw_df.loc[raw_df["kind"] == KEY_DOWN_KIND,
                    ["sessionId", "participantId", "tRelMs",
                     "payload_keyClass", "payload_repeat"]].copy()
    k["t_s"] = k["tRelMs"] / 1000.0
    k["keyClass"] = k["payload_keyClass"]
    k["is_repeat"] = k["payload_repeat"].astype(str).str.lower().isin(["true", "1"])
    return k.sort_values(["sessionId", "t_s"]).reset_index(drop=True)


def paired_durations(sess: pd.DataFrame, down_kind: str, up_kind: str) -> pd.DataFrame:
    """
    Ported from 04. For each down event, finds the next up event and returns
    the gap as a dwell/hold duration in ms. Returns a table with per-event
    timestamps (not just a bare array, unlike the notebook version) so the
    result can be re-scoped to window boundaries downstream.
    """
    d = sess.loc[sess["kind"] == down_kind, ["sessionId", "participantId", "tRelMs"]].copy()
    u = sess.loc[sess["kind"] == up_kind, "tRelMs"].sort_values().to_numpy(dtype=float)
    d = d.sort_values("tRelMs")
    rows = []
    for _, r in d.iterrows():
        t0 = float(r["tRelMs"])
        nxt = u[u > t0]
        if len(nxt):
            rows.append({
                "sessionId": r["sessionId"], "participantId": r["participantId"],
                "t_s": t0 / 1000.0, "dwell_ms": nxt[0] - t0,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sessionId", "participantId", "t_s", "dwell_ms"])


def build_transitions(keydown_events: pd.DataFrame,
                       max_transition_ms: float = MAX_TRANSITION_MS) -> pd.DataFrame:
    """
    Ported from 07. Consecutive-keydown class-transition timing
    (LETTER->LETTER, LETTER->SPACE, etc.), excluding gaps longer than
    max_transition_ms (not a single continuous typing act -- likely a pause
    to think, or a task-boundary gap).

    Each row's t_s is the timestamp of the SECOND event in the pair, so a
    transition is attributed to the window it lands in, i.e. the window
    where the transition-completing keystroke happened.
    """
    out = []
    for sid, g in keydown_events.groupby("sessionId"):
        g = g.sort_values("t_s")
        t = g["t_s"].to_numpy()
        kc = g["keyClass"].to_numpy()
        pid = g["participantId"].iloc[0]
        if len(t) < 2:
            continue
        dt_ms = np.diff(t) * 1000.0
        pairs = np.array([f"{a}->{b}" for a, b in zip(kc[:-1], kc[1:])])
        ok = dt_ms <= max_transition_ms
        out.append(pd.DataFrame({
            "sessionId": sid, "participantId": pid,
            "t_s": t[1:][ok], "transition": pairs[ok], "dt_ms": dt_ms[ok],
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["sessionId", "participantId", "t_s", "transition", "dt_ms"])


def backspace_runs(keydown_events: pd.DataFrame) -> pd.DataFrame:
    """
    Ported from 07. A 'run' is consecutive BACKSPACE keydowns with no other
    key class between them -- distinguishes a single-typo fix (run_length=1)
    from a deliberate field-wipe (long run). Each run is attributed by the
    timestamp of its FIRST backspace in the run, so it lands in the window
    where the correction episode began.
    """
    out = []
    for sid, g in keydown_events.groupby("sessionId"):
        g = g.sort_values("t_s")
        kc = g["keyClass"].to_numpy()
        t = g["t_s"].to_numpy()
        pid = g["participantId"].iloc[0]
        is_bs = kc == "BACKSPACE"
        if not is_bs.any():
            continue
        run_start_t, run_len = None, 0
        for i, b in enumerate(is_bs):
            if b:
                if run_len == 0:
                    run_start_t = t[i]
                run_len += 1
            elif run_len:
                out.append({"sessionId": sid, "participantId": pid,
                            "t_s": run_start_t, "run_length": run_len})
                run_len = 0
        if run_len:
            out.append({"sessionId": sid, "participantId": pid,
                        "t_s": run_start_t, "run_length": run_len})
    return pd.DataFrame(out) if out else pd.DataFrame(
        columns=["sessionId", "participantId", "t_s", "run_length"])


# =============================================================================
# Window-level aggregation
# =============================================================================

def _events_in_window(event_t_s: np.ndarray, w_start: float, w_end: float) -> np.ndarray:
    """Boolean mask: which events (sorted or not) fall in [w_start, w_end)."""
    return (event_t_s >= w_start) & (event_t_s < w_end)


def aggregate_typing_features(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every window in `windows` (must have sessionId, window_start_s,
    window_end_s, typing_afforded, task_boundary_straddle), compute typing
    features from events falling inside that window's span.

    Windows with task_boundary_straddle=True get NaN for every typing
    feature (option 1 -- see module docstring). Windows with
    typing_afforded in {False, NaN} also get NaN, since there's no
    meaningful typing behaviour to measure when the task didn't call for it
    or the task context itself is unknown.

    Maximalist: every feature computed here regardless of how "good" it
    looks; filtering happens at a later stage, not here.
    """
    keydown = extract_keydown_events(raw_df)
    transitions = build_transitions(keydown)
    bs_runs = backspace_runs(keydown)

    dwell = paired_durations(raw_df, KEY_DOWN_KIND, KEY_UP_KIND)

    feature_cols = [
        "typing_keydown_count",
        "typing_dwell_ms_mean", "typing_dwell_ms_median", "typing_dwell_ms_std",
        "typing_transition_ms_mean", "typing_transition_ms_median",
        "typing_letter_letter_ms_median",
        "typing_backspace_share",
        "typing_autorepeat_share",
        "typing_backspace_run_median", "typing_backspace_run_max", "typing_n_backspace_runs",
    ]
    for c in feature_cols:
        windows[c] = np.nan

    eligible = (
        (~windows["typing_straddle_conflict"].fillna(False))
        & (windows["typing_afforded"] == True)  # noqa: E712 (nullable boolean; NaN safely excluded)
    )

    for sid, sess_windows in windows.loc[eligible].groupby("sessionId"):
        kd_sess = keydown.loc[keydown["sessionId"] == sid]
        trans_sess = transitions.loc[transitions["sessionId"] == sid]
        runs_sess = bs_runs.loc[bs_runs["sessionId"] == sid]
        dwell_sess = dwell.loc[dwell["sessionId"] == sid]

        kd_t = kd_sess["t_s"].to_numpy()
        trans_t = trans_sess["t_s"].to_numpy()
        runs_t = runs_sess["t_s"].to_numpy()
        dwell_t = dwell_sess["t_s"].to_numpy()

        for idx, w in sess_windows.iterrows():
            w_start, w_end = w["window_start_s"], w["window_end_s"]

            kd_mask = _events_in_window(kd_t, w_start, w_end)
            kd_win = kd_sess.loc[kd_mask]
            n_keys = len(kd_win)
            windows.at[idx, "typing_keydown_count"] = n_keys
            if n_keys:
                windows.at[idx, "typing_backspace_share"] = (kd_win["keyClass"] == "BACKSPACE").mean()
                windows.at[idx, "typing_autorepeat_share"] = kd_win["is_repeat"].mean()

            dwell_mask = _events_in_window(dwell_t, w_start, w_end)
            dwell_win = dwell_sess.loc[dwell_mask, "dwell_ms"]
            if len(dwell_win):
                windows.at[idx, "typing_dwell_ms_mean"] = dwell_win.mean()
                windows.at[idx, "typing_dwell_ms_median"] = dwell_win.median()
                windows.at[idx, "typing_dwell_ms_std"] = dwell_win.std()

            trans_mask = _events_in_window(trans_t, w_start, w_end)
            trans_win = trans_sess.loc[trans_mask]
            if len(trans_win):
                windows.at[idx, "typing_transition_ms_mean"] = trans_win["dt_ms"].mean()
                windows.at[idx, "typing_transition_ms_median"] = trans_win["dt_ms"].median()
                ll = trans_win.loc[trans_win["transition"] == "LETTER->LETTER", "dt_ms"]
                if len(ll):
                    windows.at[idx, "typing_letter_letter_ms_median"] = ll.median()

            runs_mask = _events_in_window(runs_t, w_start, w_end)
            runs_win = runs_sess.loc[runs_mask, "run_length"]
            if len(runs_win):
                windows.at[idx, "typing_backspace_run_median"] = runs_win.median()
                windows.at[idx, "typing_backspace_run_max"] = runs_win.max()
                windows.at[idx, "typing_n_backspace_runs"] = len(runs_win)

    return windows


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # build_windows_dataset.py lives one directory up (scripts/), this file
    # lives in scripts/primitives/ -- add the parent to the path so the
    # bare import below resolves regardless of the caller's cwd.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from build_windows_dataset import build_skeleton  # noqa: E402

    RAW_CANDIDATES = [
        "data/processed/raw_events.parquet", "raw_events.parquet",
        "/mnt/user-data/uploads/raw_events.parquet",
    ]
    raw_path = next((p for p in RAW_CANDIDATES if Path(p).exists()), None)
    if raw_path is None:
        print("No raw_events.parquet found -- skipping.")
        sys.exit(0)

    raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(raw):,} events across {raw['sessionId'].nunique()} sessions.\n")

    windows = build_skeleton(raw)
    windows = aggregate_typing_features(windows, raw)

    eligible = (~windows["typing_straddle_conflict"].fillna(False)) & (windows["typing_afforded"] == True)
    print(f"\n{eligible.sum()} windows eligible for typing features "
          f"(afforded=True, no typing_straddle_conflict), of {len(windows)} total.")

    has_dwell = windows["typing_dwell_ms_mean"].notna()
    print(f"typing_dwell_ms_mean populated for {has_dwell.sum()} windows "
          f"({has_dwell.sum() / max(eligible.sum(), 1):.1%} of eligible windows -- "
          f"gap is windows with an afforded typing task but zero actual keystrokes, "
          f"a real hesitation signal, not a bug)")

    print("\nSample of eligible windows with typing features:")
    cols = ["sessionId", "window_index", "typing_keydown_count", "typing_dwell_ms_median",
            "typing_transition_ms_median", "typing_backspace_share", "typing_autorepeat_share"]
    print(windows.loc[eligible & has_dwell, cols].head(10).to_string(index=False))

    print("\nConflicting windows correctly left NaN (typing_straddle_conflict=True):")
    conflict_sample = windows.loc[windows["typing_straddle_conflict"] == True, cols]
    print(f"  {conflict_sample['typing_keydown_count'].isna().all()} "
          f"(all NaN, as expected -- {len(conflict_sample)} conflicting windows total)")
