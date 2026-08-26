"""
scripts/primitives/tap.py

Tap/touch feature family: event-level extraction ported from
04_behavioural_rhythm (paired_durations, reused for touch hold duration)
and 09_spatial_screen_behaviour (reach_stats, controlled-target aim-point).

NOTE: 05_1_tap_motion_coupling's tap_features() (impulse peak, impulse_auc,
decay_ratio, z_share, snr) is NOT ported here -- those are tap-MOTION
coupling features, built in coupling.py instead.

UNIFORM SUFFIX REBUILD (2026-08): applies the full 9-statistic suffix set
(mean/std/median/iqr/p95/max/n/cv/slope, via _stats.py) to every genuine
per-event metric -- hold duration, x/y position, radius/force -- rather
than the hand-picked 1-3 stats an earlier version used. Controlled-target
deviation and cells-touched/coverage stay as single columns: deviation is
already a per-window aggregate against a participant-level baseline (not
a within-window distribution across multiple raw values in the same
sense), and coverage is a single computed quantity, not a metric with
multiple raw observations to summarise.

Straddle handling: tapping_straddle_conflict is always False in this
cohort, so no window is ever excluded on this basis in practice.
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

TAP_DOWN_KIND = "touchstart"
TAP_UP_KIND = "touchend"
CONTROLLED_TARGET_COMPONENT = "continue_button"
GRID = 8


# =============================================================================
# Event-level extractors
# =============================================================================

def extract_tap_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    t = raw_df.loc[raw_df["kind"] == TAP_DOWN_KIND,
                    ["sessionId", "participantId", "tRelMs",
                     "payload_xNorm", "payload_yNorm", "payload_componentId",
                     "payload_radiusX", "payload_radiusY", "payload_force"]].copy()
    t["t_s"] = t["tRelMs"] / 1000.0
    t = t.rename(columns={
        "payload_xNorm": "xNorm", "payload_yNorm": "yNorm",
        "payload_componentId": "componentId",
        "payload_radiusX": "radiusX", "payload_radiusY": "radiusY",
        "payload_force": "force",
    })
    for c in ["radiusX", "radiusY", "force"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t.sort_values(["sessionId", "t_s"]).reset_index(drop=True)


def tap_hold_durations(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sess in raw_df.groupby("sessionId"):
        d = sess.loc[sess["kind"] == TAP_DOWN_KIND, ["participantId", "tRelMs"]].sort_values("tRelMs")
        u = sess.loc[sess["kind"] == TAP_UP_KIND, "tRelMs"].sort_values().to_numpy(dtype=float)
        if d.empty or len(u) == 0:
            continue
        for _, r in d.iterrows():
            t0 = float(r["tRelMs"])
            nxt = u[u > t0]
            if len(nxt):
                rows.append({"sessionId": sid, "participantId": r["participantId"],
                            "t_s": t0 / 1000.0, "hold_ms": nxt[0] - t0})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sessionId", "participantId", "t_s", "hold_ms"])


# =============================================================================
# Window-level aggregation
# =============================================================================

def _events_in_window(event_t_s: np.ndarray, w_start: float, w_end: float) -> np.ndarray:
    return (event_t_s >= w_start) & (event_t_s < w_end)


def _build_feature_column_list() -> list[str]:
    cols = ["tap_count", "tap_cells_touched", "tap_coverage",
            "tap_ctrl_target_x_dev", "tap_ctrl_target_y_dev"]
    for metric in ["tap_hold_ms", "tap_x", "tap_y", "tap_radiusX", "tap_radiusY", "tap_force"]:
        cols += summary_stat_columns(metric)
    return cols


def aggregate_tap_features(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    taps = extract_tap_events(raw_df)
    hold = tap_hold_durations(raw_df)

    ctrl = taps.loc[taps["componentId"] == CONTROLLED_TARGET_COMPONENT]
    baseline_x = ctrl.groupby("participantId")["xNorm"].median()
    baseline_y = ctrl.groupby("participantId")["yNorm"].median()

    feature_cols = _build_feature_column_list()
    windows = pd.concat([windows, pd.DataFrame(np.nan, index=windows.index, columns=feature_cols)], axis=1)

    eligible = (
        (~windows["tapping_straddle_conflict"].fillna(False))
        & (windows["tapping_afforded"] == True)  # noqa: E712
    )

    for sid, sess_windows in windows.loc[eligible].groupby("sessionId"):
        taps_sess = taps.loc[taps["sessionId"] == sid]
        hold_sess = hold.loc[hold["sessionId"] == sid]
        pid = sess_windows["participantId"].iloc[0]
        base_x = baseline_x.get(pid, np.nan)
        base_y = baseline_y.get(pid, np.nan)

        taps_t = taps_sess["t_s"].to_numpy()
        hold_t = hold_sess["t_s"].to_numpy()

        for idx, w in sess_windows.iterrows():
            w_start, w_end = w["window_start_s"], w["window_end_s"]

            tap_mask = _events_in_window(taps_t, w_start, w_end)
            tap_win = taps_sess.loc[tap_mask]
            n_taps = len(tap_win)
            windows.at[idx, "tap_count"] = n_taps

            if n_taps:
                rel_t = tap_win["t_s"].to_numpy() - w_start
                x = tap_win["xNorm"].to_numpy()
                y = tap_win["yNorm"].to_numpy()
                for suf, val in compute_summary_stats(x, rel_t).items():
                    windows.at[idx, f"tap_x_{suf}"] = val
                for suf, val in compute_summary_stats(y, rel_t).items():
                    windows.at[idx, f"tap_y_{suf}"] = val

                x_ok = x[np.isfinite(x)]
                y_ok = y[np.isfinite(y)]
                if len(x_ok) and len(y_ok) and len(x_ok) == len(y_ok):
                    gx = np.clip((x_ok * GRID).astype(int), 0, GRID - 1)
                    gy = np.clip((y_ok * GRID).astype(int), 0, GRID - 1)
                    cells = len(set(zip(gx, gy)))
                    windows.at[idx, "tap_cells_touched"] = cells
                    windows.at[idx, "tap_coverage"] = cells / (GRID * GRID)

                ctrl_win = tap_win.loc[tap_win["componentId"] == CONTROLLED_TARGET_COMPONENT]
                if len(ctrl_win):
                    if np.isfinite(base_x):
                        windows.at[idx, "tap_ctrl_target_x_dev"] = ctrl_win["xNorm"].mean() - base_x
                    if np.isfinite(base_y):
                        windows.at[idx, "tap_ctrl_target_y_dev"] = ctrl_win["yNorm"].mean() - base_y

                for field in ["radiusX", "radiusY", "force"]:
                    vals = tap_win[field].to_numpy()
                    for suf, val in compute_summary_stats(vals, rel_t).items():
                        windows.at[idx, f"tap_{field}_{suf}"] = val

            hold_mask = _events_in_window(hold_t, w_start, w_end)
            hold_win = hold_sess.loc[hold_mask]
            if len(hold_win):
                rel_t_h = hold_win["t_s"].to_numpy() - w_start
                for suf, val in compute_summary_stats(hold_win["hold_ms"].to_numpy(), rel_t_h).items():
                    windows.at[idx, f"tap_hold_ms_{suf}"] = val

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
    print(f"tap.py now generates {len(feature_cols)} candidate feature columns (18 before rebuild).\n")

    windows = aggregate_tap_features(windows, raw)

    eligible = (~windows["tapping_straddle_conflict"].fillna(False)) & (windows["tapping_afforded"] == True)
    print(f"{eligible.sum()} windows eligible, of {len(windows)} total.\n")

    elig_windows = windows.loc[eligible]
    rates = elig_windows[feature_cols].notna().mean().sort_values(ascending=False)
    print("Non-null rate, top 10 and bottom 10 columns:")
    print(rates.head(10).round(3).to_string())
    print("...")
    print(rates.tail(10).round(3).to_string())
