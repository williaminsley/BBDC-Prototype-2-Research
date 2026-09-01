"""
scripts/primitives/gesture.py

Gesture feature family: taxonomy classification (scroll_fling / scroll_drag
/ drag_swipe / long_press / tap) ported from 06_gesture_morphology, plus
stroke-endpoint spatial features from 09_spatial_screen_behaviour.

CANONICAL CLASSIFIER DECISION (2026-08): 06's build_interactions()+classify()
is used, NOT reliability_check.py's classify_gestures() -- confirmed via
direct diff on real data that the latter double-counts interactions when
touchstarts occur close together (searchsorted matching with no memory of
already-claimed touchends). See prior version's docstring / conversation
record for the full diff numbers.

UNIFORM SUFFIX REBUILD (2026-08): applies the full 9-statistic suffix set
(mean/std/median/iqr/p95/max/n/cv/slope, via _stats.py) to every genuine
per-event metric -- hold duration (overall and per gesture class),
displacement, path length, coast fraction, stroke endpoints -- rather than
the single median-only stat an earlier version used per metric.

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
from _stats import compute_summary_stats, summary_stat_columns, pair_sequential_events

TAP_DOWN_KIND = "touchstart"
TAP_UP_KIND = "touchend"
TAP_CANCEL_KIND = "touchcancel"
MOVE_KIND = "touchmove"
SCROLL_KINDS = ["scroll", "window_scroll"]  # excludes carousel_scroll -- see note in
                                              # earlier conversation record; ported
                                              # faithfully from 06's original scope

COAST_WINDOW_MS = 2000.0
COAST_FRACTION_FLING = 0.25
MIN_SCROLL_PX = 5.0
MIN_DRAG_PX = 20.0
LONG_PRESS_MS = 500.0

GESTURE_CLASSES = ["scroll_fling", "scroll_drag", "drag_swipe", "long_press", "tap"]


# =============================================================================
# Event-level extractors (unchanged -- extraction logic was correct)
# =============================================================================

def build_interactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Down/up pairing now delegates to _stats.pair_sequential_events() (2026-09
    refactor) -- the same fix described there: touchcancel is merged into the
    up-event search so the pairing pointer stops there correctly instead of
    overshooting to steal a later touch's touchend, and each up is consumed
    once so two close-together touchstarts can never claim the same touchend.
    Previously implemented as a standalone loop directly in this file (see
    git history for the original touchcancel bugfix); moved to the shared
    helper so tap.py and typing_features.py -- which had independently
    duplicated the OLDER, buggier version of this pairing pattern -- now call
    the same corrected implementation instead of maintaining their own copies.
    """
    rows = []
    for sid, s in raw_df.groupby("sessionId"):
        s = s.sort_values("tRelMs")
        pid = s["participantId"].iloc[0]
        downs = s.loc[s["kind"] == TAP_DOWN_KIND, "tRelMs"].to_numpy(dtype=float)
        up_events = s.loc[s["kind"].isin([TAP_UP_KIND, TAP_CANCEL_KIND]), ["tRelMs", "kind"]].sort_values("tRelMs")
        ups = up_events["tRelMs"].to_numpy(dtype=float)
        up_is_cancel = (up_events["kind"] == TAP_CANCEL_KIND).to_numpy()
        mv = s.loc[s["kind"] == MOVE_KIND, ["tRelMs", "payload_x", "payload_y"]]
        mv_t = mv["tRelMs"].to_numpy(dtype=float)
        mv_x = pd.to_numeric(mv["payload_x"], errors="coerce").to_numpy(dtype=float)
        mv_y = pd.to_numeric(mv["payload_y"], errors="coerce").to_numpy(dtype=float)
        sc = s.loc[s["kind"].isin(SCROLL_KINDS), ["tRelMs", "payload_scrollTop"]]
        sc_t = sc["tRelMs"].to_numpy(dtype=float)
        sc_v = pd.to_numeric(sc["payload_scrollTop"], errors="coerce").to_numpy(dtype=float)

        down_idx, up_val, is_cancel = pair_sequential_events(downs, ups, up_is_cancel)

        for di, u, cancelled in zip(down_idx, up_val, is_cancel):
            if cancelled:
                # Correctly consumed from the pairing sequence (so a later,
                # genuine touchstart/touchend pair can't be stolen), but not
                # emitted as an interaction -- a cancelled touch is not a
                # genuine completed press-release.
                continue

            d = downs[di]
            hold = u - d

            mm = (mv_t >= d) & (mv_t <= u)
            px, py = mv_x[mm], mv_y[mm]
            n_moves = int(mm.sum())
            if n_moves >= 2:
                dx, dy = np.diff(px), np.diff(py)
                path_len = float(np.nansum(np.hypot(dx, dy)))
                disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
            else:
                path_len = disp = 0.0

            dm = (sc_t >= d) & (sc_t <= u)
            cm = (sc_t > u) & (sc_t <= u + COAST_WINDOW_MS)

            def span(vals):
                v = vals[np.isfinite(vals)]
                return float(np.nanmax(v) - np.nanmin(v)) if len(v) > 1 else 0.0

            scroll_during = span(sc_v[dm])
            scroll_coast = span(sc_v[cm])

            rows.append({
                "sessionId": sid, "participantId": pid, "t_s": d / 1000.0,
                "hold_ms": hold, "path_len_px": path_len, "displacement_px": disp,
                "scroll_during_px": scroll_during, "scroll_coast_px": scroll_coast,
            })
    return pd.DataFrame(rows)


def classify(r: pd.Series) -> str:
    total_scroll = r["scroll_during_px"] + r["scroll_coast_px"]
    if total_scroll >= MIN_SCROLL_PX:
        frac = r["scroll_coast_px"] / total_scroll if total_scroll > 0 else 0.0
        return "scroll_fling" if frac >= COAST_FRACTION_FLING else "scroll_drag"
    if r["displacement_px"] >= MIN_DRAG_PX:
        return "drag_swipe"
    if r["hold_ms"] >= LONG_PRESS_MS:
        return "long_press"
    return "tap"


def build_classified_interactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    interactions = build_interactions(raw_df)
    if interactions.empty:
        interactions["gesture"] = pd.Series(dtype=str)
        return interactions
    interactions["gesture"] = interactions.apply(classify, axis=1)
    return interactions


def stroke_endpoints(raw_df: pd.DataFrame) -> pd.DataFrame:
    mv = raw_df.loc[raw_df["kind"] == MOVE_KIND,
                     ["sessionId", "participantId", "tRelMs", "payload_xNorm", "payload_yNorm"]].copy()
    rows = []
    for sid, g in mv.groupby("sessionId"):
        g = g.sort_values("tRelMs")
        t = g["tRelMs"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        brk = np.where(np.diff(t) > 300)[0]
        idx_groups = np.split(np.arange(len(g)), brk + 1)
        gx = g["payload_xNorm"].to_numpy()
        gy = g["payload_yNorm"].to_numpy()
        gt = t
        pid = g["participantId"].iloc[0]
        for run in idx_groups:
            if len(run) < 3:
                continue
            rows.append({
                "sessionId": sid, "participantId": pid, "t_s": gt[run[0]] / 1000.0,
                "x0": gx[run[0]], "y0": gy[run[0]], "x1": gx[run[-1]], "y1": gy[run[-1]],
                "n": len(run),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sessionId", "participantId", "t_s", "x0", "y0", "x1", "y1", "n"])


# =============================================================================
# Window-level aggregation
# =============================================================================

def _events_in_window(event_t_s: np.ndarray, w_start: float, w_end: float) -> np.ndarray:
    return (event_t_s >= w_start) & (event_t_s < w_end)


def _build_feature_column_list() -> list[str]:
    cols = ["gesture_n_interactions"]
    for g in GESTURE_CLASSES:
        cols.append(f"gesture_share_{g}")

    cols += summary_stat_columns("gesture_hold_ms")
    for g in GESTURE_CLASSES:
        cols += summary_stat_columns(f"gesture_hold_{g}_ms")

    cols += summary_stat_columns("gesture_drag_displacement_px")
    cols += summary_stat_columns("gesture_drag_path_len_px")
    cols += summary_stat_columns("gesture_scroll_fling_coast_frac")
    cols += summary_stat_columns("gesture_stroke_x0")
    cols += summary_stat_columns("gesture_stroke_y0")
    return cols


def aggregate_gesture_features(windows: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    interactions = build_classified_interactions(raw_df)
    strokes = stroke_endpoints(raw_df)

    feature_cols = _build_feature_column_list()
    windows = pd.concat([windows, pd.DataFrame(np.nan, index=windows.index, columns=feature_cols)], axis=1)

    eligible = (
        (~windows["tapping_straddle_conflict"].fillna(False))
        & (windows["tapping_afforded"] == True)  # noqa: E712
    )

    for sid, sess_windows in windows.loc[eligible].groupby("sessionId"):
        int_sess = interactions.loc[interactions["sessionId"] == sid]
        stroke_sess = strokes.loc[strokes["sessionId"] == sid]

        int_t = int_sess["t_s"].to_numpy()
        stroke_t = stroke_sess["t_s"].to_numpy()

        for idx, w in sess_windows.iterrows():
            w_start, w_end = w["window_start_s"], w["window_end_s"]

            int_mask = _events_in_window(int_t, w_start, w_end)
            int_win = int_sess.loc[int_mask]
            n_int = len(int_win)
            windows.at[idx, "gesture_n_interactions"] = n_int

            if n_int:
                rel_t = int_win["t_s"].to_numpy() - w_start
                class_counts = int_win["gesture"].value_counts()
                for g in GESTURE_CLASSES:
                    windows.at[idx, f"gesture_share_{g}"] = class_counts.get(g, 0) / n_int

                for suf, val in compute_summary_stats(int_win["hold_ms"].to_numpy(), rel_t).items():
                    windows.at[idx, f"gesture_hold_ms_{suf}"] = val

                for g in GESTURE_CLASSES:
                    sub = int_win.loc[int_win["gesture"] == g]
                    if len(sub):
                        rel_t_g = sub["t_s"].to_numpy() - w_start
                        for suf, val in compute_summary_stats(sub["hold_ms"].to_numpy(), rel_t_g).items():
                            windows.at[idx, f"gesture_hold_{g}_ms_{suf}"] = val

                drag = int_win.loc[int_win["gesture"] == "drag_swipe"]
                if len(drag):
                    rel_t_d = drag["t_s"].to_numpy() - w_start
                    for suf, val in compute_summary_stats(drag["displacement_px"].to_numpy(), rel_t_d).items():
                        windows.at[idx, f"gesture_drag_displacement_px_{suf}"] = val
                    for suf, val in compute_summary_stats(drag["path_len_px"].to_numpy(), rel_t_d).items():
                        windows.at[idx, f"gesture_drag_path_len_px_{suf}"] = val

                fling = int_win.loc[int_win["gesture"] == "scroll_fling"]
                if len(fling):
                    tot = fling["scroll_during_px"] + fling["scroll_coast_px"]
                    frac = (fling["scroll_coast_px"] / tot.replace(0, np.nan)).to_numpy()
                    rel_t_f = fling["t_s"].to_numpy() - w_start
                    for suf, val in compute_summary_stats(frac, rel_t_f).items():
                        windows.at[idx, f"gesture_scroll_fling_coast_frac_{suf}"] = val

            stroke_mask = _events_in_window(stroke_t, w_start, w_end)
            stroke_win = stroke_sess.loc[stroke_mask]
            if len(stroke_win):
                rel_t_s = stroke_win["t_s"].to_numpy() - w_start
                for suf, val in compute_summary_stats(stroke_win["x0"].to_numpy(), rel_t_s).items():
                    windows.at[idx, f"gesture_stroke_x0_{suf}"] = val
                for suf, val in compute_summary_stats(stroke_win["y0"].to_numpy(), rel_t_s).items():
                    windows.at[idx, f"gesture_stroke_y0_{suf}"] = val

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

    interactions = build_classified_interactions(raw)
    print(f"gesture taxonomy (whole cohort): {len(interactions):,} interactions")
    print(interactions["gesture"].value_counts().to_string())
    print()

    windows = build_skeleton(raw)
    feature_cols = _build_feature_column_list()
    print(f"gesture.py now generates {len(feature_cols)} candidate feature columns (17 before rebuild).\n")

    windows = aggregate_gesture_features(windows, raw)

    eligible = (~windows["tapping_straddle_conflict"].fillna(False)) & (windows["tapping_afforded"] == True)
    print(f"{eligible.sum()} windows eligible, of {len(windows)} total.\n")

    elig_windows = windows.loc[eligible]
    rates = elig_windows[feature_cols].notna().mean().sort_values(ascending=False)
    print("Non-null rate, top 10 and bottom 10 columns:")
    print(rates.head(10).round(3).to_string())
    print("...")
    print(rates.tail(10).round(3).to_string())
