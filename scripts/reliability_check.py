"""
reliability_check.py

A single, reusable version of the "is this result trustworthy" check developed
and corrected inside notebook 05.2. Every notebook so far has hand-written a
version of this; nb05_2 showed the naive version (just "does one participant
supply over 40% of the data") gives the WRONG answer -- it flagged
scroll_fling/scroll_drag as untrustworthy (pH3S4X4 supplied 50-60% of their
data) when they were the most solid results in that notebook, and passed
drag_swipe, which was actually the shakiest.

What actually distinguishes a trustworthy pooled result, established by
comparing all four gesture classes side by side in nb05_2:

  1. Share alone is not the test. A participant supplying most of the VOLUME
     is fine if every other contributing participant independently points the
     same direction.
  2. Direction of effect has to be checked across participants, not just
     pooled. If participants disagree on the sign, the pooled mean is
     meaningless regardless of how evenly the data is spread.
  3. "Every participant agrees" is not enough on its own if most of those
     participants are single-point estimates (nb05_2's long_press: 3 of 7
     contributing participants had exactly one gesture each).

This module makes that three-part check a function instead of a paragraph,
so it gets applied the same way everywhere -- including the systematic
register pass -- rather than re-derived per notebook.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


DOMINANCE_SHARE_THRESHOLD = 0.40   # a share above this is flagged, not disqualifying on its own
MIN_RELIABLE_N = 5                  # below this, a participant has no usable estimate of their own
MIN_CONTRIBUTING_PARTICIPANTS = 5   # fewer WELL-SUPPORTED participants than this and the check is underpowered
CONSENSUS_THRESHOLD = 0.80          # JUDGMENT CALL, not derived: below this share of well-supported
                                    # participants agreeing on direction, treat the pooled result as
                                    # contested. Set at 0.80 so that 7-of-10 agreement (nb05_2's
                                    # drag_swipe) reads as contested rather than settled -- roughly a
                                    # third of participants pointing the other way is real disagreement,
                                    # not noise. Tune deliberately if that turns out too strict.


@dataclass
class ReliabilityResult:
    value_col: str
    n_total: int
    n_participants_contributing: int
    n_participants_total: int
    max_share: float
    max_share_participant: str
    direction_unanimous: bool
    direction_consensus_pct: float          # fraction of contributing participants sharing the majority sign
    n_thin_participants: int                # contributing participants with n < MIN_RELIABLE_N
    thin_participants: list = field(default_factory=list)
    per_participant: pd.DataFrame = None
    verdict: str = ""
    reason: str = ""

    def __repr__(self):
        return (f"ReliabilityResult('{self.value_col}': {self.verdict} -- {self.reason})")


def check_reliability(df: pd.DataFrame, participant_col: str, value_col: str,
                      dominance_threshold: float = DOMINANCE_SHARE_THRESHOLD,
                      min_reliable_n: int = MIN_RELIABLE_N,
                      min_participants: int = MIN_CONTRIBUTING_PARTICIPANTS,
                      consensus_threshold: float = CONSENSUS_THRESHOLD) -> ReliabilityResult:
    """
    Check whether a pooled result in `df[value_col]`, grouped by
    `df[participant_col]`, is trustworthy -- not just whether one participant
    dominates the sample.

    Returns a ReliabilityResult with a `.verdict` in
    {"trustworthy", "contested", "underpowered", "dominated_and_contested"}
    and a `.reason` explaining which of the three checks drove it, so the
    verdict is auditable rather than a black box.

    Usage:
        r = check_reliability(taps_df, 'participantId', 'impulse_peak')
        print(r.verdict, r.reason)
        print(r.per_participant)
    """
    sub = df[[participant_col, value_col]].dropna()
    if sub.empty:
        return ReliabilityResult(value_col, 0, 0, df[participant_col].nunique(), np.nan, "",
                                 False, 0.0, 0, [], pd.DataFrame(),
                                 "underpowered", "no non-null data for this column")

    grouped = sub.groupby(participant_col)[value_col]
    counts = grouped.size()
    means = grouped.mean()
    per_p = pd.DataFrame({'n': counts, 'mean': means}).sort_values('mean')

    n_total = int(counts.sum())
    n_contributing = int(counts.shape[0])
    n_all = int(df[participant_col].nunique())

    max_share = float(counts.max() / n_total)
    max_share_p = str(counts.idxmax())

    thin = per_p.loc[per_p['n'] < min_reliable_n].index.tolist()
    n_thin = len(thin)

    # A participant with one observation has not established a direction for
    # themselves -- they cannot "agree" or "disagree" in any meaningful sense.
    # So both the participant count and the consensus calculation use only
    # participants with enough data to have their own estimate. This is the
    # fix for nb05_2's long_press case: 100% apparent agreement across 7
    # contributors, but 3 of those were single-gesture point estimates.
    reliable = per_p.loc[per_p['n'] >= min_reliable_n]
    n_reliable = int(len(reliable))

    rel_signs = np.sign(reliable['mean'].to_numpy())
    rel_signs = rel_signs[rel_signs != 0]
    if len(rel_signs) == 0:
        direction_unanimous, consensus_pct = True, 1.0
    else:
        pos_frac = (rel_signs > 0).mean()
        consensus_pct = float(max(pos_frac, 1 - pos_frac))
        direction_unanimous = consensus_pct == 1.0

    result = ReliabilityResult(
        value_col=value_col, n_total=n_total, n_participants_contributing=n_contributing,
        n_participants_total=n_all, max_share=max_share, max_share_participant=max_share_p,
        direction_unanimous=direction_unanimous, direction_consensus_pct=consensus_pct,
        n_thin_participants=n_thin, thin_participants=thin, per_participant=per_p,
    )

    # --- the verdict logic itself, mirroring nb05_2's corrected reading ------
    if n_reliable < min_participants:
        result.verdict = "underpowered"
        result.reason = (f"only {n_reliable} participants have >= {min_reliable_n} observations "
                         f"each (of {n_contributing} contributing at all, {n_all} total) -- "
                         f"apparent agreement may be single-point estimates, not replication")
    elif consensus_pct < consensus_threshold:
        result.verdict = "contested"
        result.reason = (f"participants disagree on direction ({consensus_pct:.0%} consensus among "
                         f"{n_reliable} well-supported participants, need >= "
                         f"{consensus_threshold:.0%}) -- the pooled mean is not trustworthy "
                         f"regardless of sample share")
    elif max_share > dominance_threshold and not direction_unanimous:
        result.verdict = "dominated_and_contested"
        result.reason = (f"{max_share_p} supplies {max_share:.0%} of the data AND direction is not "
                         f"unanimous -- do not trust this without more data from other participants")
    else:
        result.verdict = "trustworthy"
        share_note = (f" (despite {max_share_p} supplying {max_share:.0%} of the volume -- "
                      f"direction still replicates independently)" if max_share > dominance_threshold else "")
        result.reason = (f"{consensus_pct:.0%} of {n_reliable} well-supported participants agree "
                         f"on direction{share_note}")

    return result


def print_reliability_report(result: ReliabilityResult):
    """Human-readable version of a ReliabilityResult, matching the notebook print style."""
    print(f"--- {result.value_col} (n={result.n_total}) ---")
    print(f"contributed by {result.n_participants_contributing} of {result.n_participants_total} participants")
    print(result.per_participant.round(4).to_string())
    print(f"\nmax single-participant share: {result.max_share:.1%} ({result.max_share_participant})")
    print(f"direction consensus: {result.direction_consensus_pct:.1%}")
    if result.thin_participants:
        print(f"thin contributions (n < {MIN_RELIABLE_N}): {', '.join(result.thin_participants)}")
    print(f"\nVERDICT: {result.verdict.upper()}")
    print(f"  {result.reason}")
    print()


# =============================================================================
# self-validation: reproduce nb05_2's own four-class verdict exactly
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Rebuild nb05_2's GMETA table from raw data, using the same logic, to
    # validate this module reproduces the notebook's own corrected conclusions
    # rather than asserting the logic is right without checking.
    sys.path.insert(0, str(Path(__file__).parent))

    RAW_CANDIDATES = ["data/processed/raw_events.parquet", "raw_events.parquet",
                      "/mnt/user-data/uploads/raw_events.parquet"]
    raw_path = next((p for p in RAW_CANDIDATES if Path(p).exists()), None)
    if raw_path is None:
        print("No raw_events.parquet found for self-validation -- skipping.")
        sys.exit(0)

    raw_all = pd.read_parquet(raw_path)
    raw_all['deviceFamily'] = raw_all['deviceFamily'].astype(str).str.lower()
    raw_all['tRelMs'] = pd.to_numeric(raw_all['tRelMs'], errors='coerce')
    raw = raw_all.loc[(raw_all['deviceFamily'] == 'mobile') & raw_all['tRelMs'].notna()].copy()

    SCROLL_KINDS = ['scroll', 'window_scroll']
    COAST_WINDOW_S, COAST_FRACTION_FLING = 2.0, 0.25
    MIN_SCROLL_PX, MIN_DRAG_PX, LONG_PRESS_S = 5.0, 20.0, 0.5
    ACC = ['payload_ax', 'payload_ay', 'payload_az']

    def classify_gestures(sess):
        downs = np.sort(sess.loc[sess['kind'] == 'touchstart', 'tRelMs'].to_numpy(float)) / 1000.0
        ups = np.sort(sess.loc[sess['kind'] == 'touchend', 'tRelMs'].to_numpy(float)) / 1000.0
        sc = sess.loc[sess['kind'].isin(SCROLL_KINDS), ['tRelMs', 'payload_scrollTop']].copy()
        sc['tRelMs'] = sc['tRelMs'] / 1000.0
        sc['payload_scrollTop'] = pd.to_numeric(sc['payload_scrollTop'], errors='coerce')
        sc_t = sc['tRelMs'].to_numpy(float); sc_v = sc['payload_scrollTop'].to_numpy(float)
        mv = sess.loc[sess['kind'] == 'touchmove', ['tRelMs', 'payload_x', 'payload_y']].copy()
        mv['tRelMs'] = mv['tRelMs'] / 1000.0
        mv_t = mv['tRelMs'].to_numpy(float)
        mv_x = pd.to_numeric(mv['payload_x'], errors='coerce').to_numpy(float)
        mv_y = pd.to_numeric(mv['payload_y'], errors='coerce').to_numpy(float)
        rows = []
        for d in downs:
            idx = np.searchsorted(ups, d)
            if idx >= len(ups):
                continue
            u = ups[idx]
            dm = (sc_t >= d) & (sc_t <= u); cm = (sc_t > u) & (sc_t <= u + COAST_WINDOW_S)
            def span(vals):
                v = vals[np.isfinite(vals)]
                return float(np.nanmax(v) - np.nanmin(v)) if len(v) > 1 else 0.0
            scroll_during = span(sc_v[dm]); scroll_coast = span(sc_v[cm])
            total_scroll = scroll_during + scroll_coast
            mm = (mv_t >= d) & (mv_t <= u)
            disp = 0.0
            if mm.sum() >= 2:
                px, py = mv_x[mm], mv_y[mm]
                disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
            if total_scroll >= MIN_SCROLL_PX:
                frac = scroll_coast / total_scroll if total_scroll > 0 else 0.0
                cls = 'scroll_fling' if frac >= COAST_FRACTION_FLING else 'scroll_drag'
            elif disp >= MIN_DRAG_PX:
                cls = 'drag_swipe'
            elif (u - d) >= LONG_PRESS_S:
                cls = 'long_press'
            else:
                cls = 'tap'
            rows.append({'t_start': d, 't_end': u, 'duration_s': u - d, 'gesture': cls})
        return pd.DataFrame(rows)

    PHASE_BINS = np.linspace(-0.3, 1.5, 24)
    PRE_FRAC = 0.3

    def motion_arrays(sess):
        mo = sess.loc[sess['kind'] == 'devicemotion'].sort_values('tRelMs')
        t = mo['tRelMs'].to_numpy(float) / 1000.0
        a = [pd.to_numeric(mo[c], errors='coerce').to_numpy(float) for c in ACC]
        mag = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        ok = np.isfinite(mag) & np.isfinite(t)
        return t[ok], mag[ok]

    def phase_profile(t0, t1, t_m, mag):
        dur = t1 - t0
        if dur <= 0:
            return None
        sel = (t_m >= t0 - PRE_FRAC * dur) & (t_m < t0 + 1.5 * dur)
        if sel.sum() < 5:
            return None
        tt, vv = t_m[sel], mag[sel]
        phase = (tt - t0) / dur
        pre_m = phase < 0
        if pre_m.sum() < 2:
            return None
        base = vv[pre_m].mean()
        return np.interp(PHASE_BINS, phase, vv - base, left=np.nan, right=np.nan)

    print("Rebuilding gesture-motion table from raw data (validation run)...")
    gest_rows, real_rows, meta = [], [], []
    for sid, sess in raw.groupby('sessionId'):
        g = classify_gestures(sess)
        if not len(g):
            continue
        g['participantId'] = sess['participantId'].iloc[0]; g['sessionId'] = sid
        t_m, mag = motion_arrays(sess)
        if len(t_m) < 20:
            continue
        for r in g.itertuples():
            p = phase_profile(r.t_start, r.t_end, t_m, mag)
            if p is None:
                continue
            real_rows.append(p)
            meta.append({'participantId': r.participantId, 'sessionId': sid, 'gesture': r.gesture})
    REAL = np.vstack(real_rows)
    GMETA = pd.DataFrame(meta)
    mid_bin = np.argmin(np.abs(PHASE_BINS - 0.5))
    GMETA['mid_energy'] = REAL[:, mid_bin]

    print(f"{len(GMETA)} classified gestures with usable motion profiles\n")
    print("=" * 70)
    print("VALIDATION: does this module reproduce nb05_2's own four-class verdict?")
    print("=" * 70)
    expected = {'scroll_fling': 'trustworthy', 'scroll_drag': 'trustworthy',
               'drag_swipe': 'contested', 'long_press': 'underpowered'}
    all_match = True
    for cls, exp in expected.items():
        sub = GMETA.loc[GMETA['gesture'] == cls]
        r = check_reliability(sub, 'participantId', 'mid_energy')
        match = "OK" if r.verdict == exp else "MISMATCH"
        if r.verdict != exp:
            all_match = False
        print(f"\n{cls}: got '{r.verdict}' (expected '{exp}')  [{match}]")
        print(f"  {r.reason}")
    print("\n" + "=" * 70)
    print("ALL VERDICTS MATCH nb05_2" if all_match else "SOME VERDICTS DIFFER -- inspect above")
    print("=" * 70)
