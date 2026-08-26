"""
scripts/primitives/typing_features.py

Note ON THE FILENAME: deliberately NOT typing.py -- shadows Python's own
stdlib 'typing' module, breaking pandas/numpy imports. Confirmed directly.

Typing feature family: event-level extraction ported from 04_behavioural_rhythm
(paired_durations -- dwell time) and 07_typing_structure (build_transitions,
backspace_runs, autorepeat), plus window-level aggregation.

UNIFORM SUFFIX REBUILD (2026-08): a P1 slide confirmed the actual feature-
naming convention -- family + metric + summary-statistic suffix, with a
FIXED set of 9 suffixes (mean/std/median/iqr/p95/max/n/cv/slope) applied to
every raw per-event metric. Earlier versions of this file applied an
inconsistent subset per metric (dwell got mean/median/std; transitions got
only mean/median). This version uses scripts/primitives/_stats.py's
compute_summary_stats() so every per-event metric gets the same 9 columns,
not a hand-picked subset.

Per-event metrics that now get the full 9-suffix treatment:
  - dwell time, overall and split by all 7 key classes
  - transition timing, overall and split by all 9 LETTER/SPACE/BACKSPACE pairs
  - backspace run length

NOT given the 9-suffix treatment (already single numbers per window, no
within-window distribution to summarise): keydown count, backspace share,
autorepeat share, key-class composition shares. These stay as single
columns, per _stats.py's docstring on when the helper applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))  # ensures _stats resolves whether this
                                                    # module is run standalone or imported
                                                    # as part of the primitives package
from _stats import compute_summary_stats, summary_stat_columns

KEY_DOWN_KIND = "keydown"
KEY_UP_KIND = "keyup"
MAX_TRANSITION_MS = 3000.0

KEY_CLASSES = ["LETTER", "SPACE", "BACKSPACE", "DIGIT", "PUNCT_OR_SYMBOL", "ENTER", "OTHER"]
TRANSITION_MATRIX_CLASSES = ["LETTER", "SPACE", "BACKSPACE"]


# =============================================================================
# Event-level extractors (unchanged from prior version -- the extraction
# logic was correct, only the aggregation/suffix step needed fixing)
# =============================================================================

def extract_keydown_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    k = raw_df.loc[raw_df["kind"] == KEY_DOWN_KIND,
                    ["sessionId", "participantId", "tRelMs",
                     "payload_keyClass", "payload_repeat"]].copy()
    k["t_s"] = k["tRelMs"] / 1000.0
    k["keyClass"] = k["payload_keyClass"]
    k["is_repeat"] = k["payload_repeat"].astype(str).str.lower().isin(["true", "1"])
    return k.sort_values(["sessionId", "t_s"]).reset_index(drop=True)


def paired_durations(sess: pd.DataFrame, down_kind: str, up_kind: str) -> pd.DataFrame:
    d = sess.loc[sess["kind"] == down_kind,
                 ["sessionId", "participantId", "tRelMs", "payload_keyClass"]].copy()
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
                "keyClass": r.get("payload_keyClass", np.nan),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sessionId", "participantId", "t_s", "dwell_ms", "keyClass"])


def build_transitions(keydown_events: pd.DataFrame,
                       max_transition_ms: float = MAX_TRANSITION_MS) -> pd.DataFrame:
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
    return (event_t_s >= w_start) & (event_t_s < w_end)


def _build_feature_column_list() -> list[str]:
    cols = ["typing_keydown_count", "typing_backspace_share", "typing_autorepeat_share"]
    for kc in KEY_CLASSES:
        cols.append(f"typing_share_{kc.lower()}")

    # full 9-suffix treatment for every genuine per-event metric
    cols += summary_stat_columns("typing_dwell_ms")
    for kc in KEY_CLASSES:
        cols += summary_stat_columns(f"typing_dwell_{kc.lower()}_ms")
    cols += summary_stat_columns("typing_transition_ms")
    for a in TRANSITION_MATRIX_CLASSES:
        for b in TRANSITION_MATRIX_CLASSES:
            cols += summary_stat_columns(f"typing_transition_{a.lower()}_{b.lower()}_ms")
    cols += summary_stat_columns("typing_backspace_run_length")
    return cols


def aggregate_typing_features(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    keydown = extract_keydown_events(raw_df)
    transitions = build_transitions(keydown)
    bs_runs = backspace_runs(keydown)
    dwell = paired_durations(raw_df, KEY_DOWN_KIND, KEY_UP_KIND)

    feature_cols = _build_feature_column_list()
    windows = pd.concat([windows, pd.DataFrame(np.nan, index=windows.index, columns=feature_cols)], axis=1)

    eligible = (
        (~windows["typing_straddle_conflict"].fillna(False))
        & (windows["typing_afforded"] == True)  # noqa: E712
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
                class_counts = kd_win["keyClass"].value_counts()
                for kc in KEY_CLASSES:
                    windows.at[idx, f"typing_share_{kc.lower()}"] = class_counts.get(kc, 0) / n_keys

            dwell_mask = _events_in_window(dwell_t, w_start, w_end)
            dwell_win = dwell_sess.loc[dwell_mask]
            if len(dwell_win):
                rel_t = dwell_win["t_s"].to_numpy() - w_start
                stats = compute_summary_stats(dwell_win["dwell_ms"].to_numpy(), rel_t)
                for suf, val in stats.items():
                    windows.at[idx, f"typing_dwell_ms_{suf}"] = val
                for kc in KEY_CLASSES:
                    sub = dwell_win.loc[dwell_win["keyClass"] == kc]
                    if len(sub):
                        rel_t_kc = sub["t_s"].to_numpy() - w_start
                        stats_kc = compute_summary_stats(sub["dwell_ms"].to_numpy(), rel_t_kc)
                        for suf, val in stats_kc.items():
                            windows.at[idx, f"typing_dwell_{kc.lower()}_ms_{suf}"] = val

            trans_mask = _events_in_window(trans_t, w_start, w_end)
            trans_win = trans_sess.loc[trans_mask]
            if len(trans_win):
                rel_t = trans_win["t_s"].to_numpy() - w_start
                stats = compute_summary_stats(trans_win["dt_ms"].to_numpy(), rel_t)
                for suf, val in stats.items():
                    windows.at[idx, f"typing_transition_ms_{suf}"] = val
                for a in TRANSITION_MATRIX_CLASSES:
                    for b in TRANSITION_MATRIX_CLASSES:
                        pair_sub = trans_win.loc[trans_win["transition"] == f"{a}->{b}"]
                        if len(pair_sub):
                            rel_t_pair = pair_sub["t_s"].to_numpy() - w_start
                            stats_pair = compute_summary_stats(pair_sub["dt_ms"].to_numpy(), rel_t_pair)
                            for suf, val in stats_pair.items():
                                windows.at[idx, f"typing_transition_{a.lower()}_{b.lower()}_ms_{suf}"] = val

            runs_mask = _events_in_window(runs_t, w_start, w_end)
            runs_win = runs_sess.loc[runs_mask]
            if len(runs_win):
                rel_t = runs_win["t_s"].to_numpy() - w_start
                stats = compute_summary_stats(runs_win["run_length"].to_numpy(), rel_t)
                for suf, val in stats.items():
                    windows.at[idx, f"typing_backspace_run_length_{suf}"] = val

    return windows


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

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
    feature_cols = _build_feature_column_list()
    print(f"typing_features.py now generates {len(feature_cols)} candidate feature columns "
          f"(41 before the uniform-suffix rebuild).\n")

    windows = aggregate_typing_features(windows, raw)

    eligible = (~windows["typing_straddle_conflict"].fillna(False)) & (windows["typing_afforded"] == True)
    print(f"{eligible.sum()} windows eligible, of {len(windows)} total.\n")

    elig_windows = windows.loc[eligible]
    rates = elig_windows[feature_cols].notna().mean().sort_values(ascending=False)
    print("Non-null rate, top 15 and bottom 15 columns:")
    print(rates.head(15).round(3).to_string())
    print("...")
    print(rates.tail(15).round(3).to_string())
