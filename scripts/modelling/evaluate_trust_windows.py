"""
scripts/modelling/evaluate_trust_windows.py

Measures verification performance as a function of HOW MUCH ACCUMULATED
BEHAVIOUR the system has seen -- the "trust window" -- rather than scoring
every 15s window in isolation.

WHY THIS IS THE PRIMARY EXPERIMENT
----------------------------------
Continuous authentication does not need a verdict from a standing start
every 15 seconds; it accumulates confidence while a session is in use. An
exploratory pass over this cohort (raw-distance baseline, session-safe,
mobile-only, 11 evaluable identities) found accumulation buys far more than
multimodal fusion does on this data:

    single 15s window          AUC 0.701
    ~10 windows (~80s)         AUC 0.838
    ~20 windows (~160s)        AUC 0.863
    whole probe session        AUC 0.824

    equal-weight 4-modality fusion of single windows   AUC 0.682
    (i.e. WORSE than tap alone at 0.701)

So trust-window length is treated here as the main experimental variable,
and modality is a secondary one, rather than the other way round. The dip
at whole-session relative to ~160s is itself a finding worth reporting --
it suggests an optimum before within-session drift starts to hurt.

WHAT MODEL THIS USES
--------------------
Deliberately none: this is P1's Section 7.2 raw-prototype Euclidean
baseline -- median-impute, standardise, average the enrolment windows into
a reference vector, score a probe by negative Euclidean distance. There is
NO learned embedding here. That is the point: it establishes the floor that
a trained embedder must beat, using the identical scoring and evaluation
harness, so any later improvement is attributable to the embedding step
rather than to some other difference (P1's stated rationale for keeping the
baseline on the same harness).

LEAKAGE CONTROL
---------------
Imputation medians and standardisation statistics are fit on TRAIN-role
sessions ONLY, then applied unchanged to enrol and probe (P1 Section 8.4).
Fitting them on everything first would let probe-set distribution leak into
the scaling -- a quieter version of the same problem the session-safe split
exists to prevent. Where a modality has too few train windows to fit on
(the sparse-typing case), the script says so explicitly and falls back
rather than failing silently, and the fallback is recorded in the output.

TRUST WINDOW CONSTRUCTION
-------------------------
Ineligible windows were removed from each modality table upstream, so
consecutive ROWS are not necessarily adjacent in TIME. A trust window of k
here means k consecutive AVAILABLE windows within one probe session, and
the script records the actual median elapsed span each k corresponds to
rather than assuming k * stride. Trust windows never cross a session
boundary.

Trust windows at stride 1 overlap heavily (and the underlying 15s windows
already overlap at 7.5s stride), so the comparison counts are NOT
independent samples. They are reported, not treated as an effective sample
size, and --trust-stride is available to thin them.

NEGATIVE CONTROL
----------------
Participant labels are shuffled at SESSION level (preserving within-session
structure) and the whole evaluation re-run. If real-label performance is
genuine identity signal rather than an artefact of session structure or
window overlap, shuffled performance should collapse toward chance.

OUTPUT
------
data/processed/modelling/trust_window_results.csv
data/processed/modelling/trust_window_per_participant.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ID_COLS = {
    "sessionId", "participantId", "deviceFamily",
    "window_index", "window_start_s", "window_end_s",
    "taskIndex", "taskType", "activeArea", "task_pass",
    "task_start_s", "task_end_s",
}

DEFAULT_K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20]
MIN_FEATURE_COVERAGE = 0.5   # a feature must be populated in at least this
                              # fraction of the modality's own eligible
                              # windows to be used -- upstream screening used
                              # the full-table denominator, which is a looser
                              # bar than the per-modality one that matters here
MIN_TRAIN_WINDOWS = 30        # below this, train-only fitting is not
                              # meaningful; see fit_preprocessing()
MIN_PROBE_IDENTITIES = 3      # a row computed over fewer probe identities than
                              # this is not reported. Added after a run
                              # produced typing AUC=0.994 at k=20 off a SINGLE
                              # probe identity -- an impressive-looking number
                              # describing one person, which is worse than no
                              # number at all because it invites being quoted.


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUC, EER, and FAR at the 10%-FRR operating point (P1 Section 8.3)."""
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "eer": np.nan, "far_at_10pct_frr": np.nan}

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    frr = 1.0 - tpr

    # EER: where false-accept and false-reject rates cross.
    idx = int(np.nanargmin(np.abs(fpr - frr)))
    eer = float((fpr[idx] + frr[idx]) / 2.0)

    # FAR at FRR = 10%: interpolate FPR at TPR = 0.90.
    far_at_10 = float(np.interp(0.90, tpr, fpr))

    return {"auc": float(auc), "eer": eer, "far_at_10pct_frr": far_at_10}


# =============================================================================
# Core evaluation
# =============================================================================

def select_features(df: pd.DataFrame) -> list[str]:
    # `role` is merged in from the splits table and must be excluded here --
    # it is the label structure, not a feature. Also guard against any
    # non-numeric column slipping through (upstream tables are all-float
    # for features, but the merge makes that worth enforcing rather than
    # assuming).
    excluded = ID_COLS | {"role"}
    feats = [c for c in df.columns
             if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    coverage = df[feats].notna().mean()
    return coverage[coverage >= MIN_FEATURE_COVERAGE].index.tolist()


def fit_preprocessing(train_df: pd.DataFrame, all_df: pd.DataFrame, feats: list[str]):
    """Fit imputer+scaler on train-role windows only (P1 Section 8.4).

    Returns (transform_fn, fitted_on_label). If the train split is too thin
    for the fit to mean anything -- which happens for sparse modalities
    where most sessions carry few or no eligible windows -- this falls back
    to fitting on all available windows and SAYS SO, rather than either
    crashing or silently pretending the leakage control held."""
    if len(train_df) >= MIN_TRAIN_WINDOWS:
        basis, label = train_df, "train_only"
    else:
        basis, label = all_df, "all_windows_fallback"

    imputer = SimpleImputer(strategy="median").fit(basis[feats])
    scaler = StandardScaler().fit(imputer.transform(basis[feats]))

    def transform(sub: pd.DataFrame) -> np.ndarray:
        return scaler.transform(imputer.transform(sub[feats]))

    return transform, label


def build_trust_windows(probe_df: pd.DataFrame, transform, k: int, stride: int):
    """Rolling mean over k consecutive AVAILABLE windows within one session.

    Returns (vectors, spans_s) where spans_s is each trust window's actual
    elapsed wall-clock span -- not assumed from k, because ineligible
    windows were dropped upstream so rows may not be time-adjacent."""
    vectors, spans = [], []
    for _, sess in probe_df.groupby("sessionId"):
        sess = sess.sort_values("window_start_s")
        X = transform(sess)
        starts = sess["window_start_s"].to_numpy()
        ends = sess["window_end_s"].to_numpy()
        if len(sess) < k:
            continue
        for i in range(0, len(sess) - k + 1, stride):
            vectors.append(X[i:i + k].mean(axis=0))
            spans.append(ends[i + k - 1] - starts[i])
    return (np.array(vectors), np.array(spans)) if vectors else (np.empty((0, 0)), np.array([]))


def evaluate_modality(df: pd.DataFrame, splits: pd.DataFrame, k_values: list[int],
                       stride: int, shuffle_seed: int | None = None,
                       fixed_identities: set | None = None):
    """Returns (results_rows, per_participant_rows).

    fixed_identities, when given, restricts evaluation to exactly that set
    of probe identities at EVERY k. This exists to prevent a survivorship
    confound: a trust window of k needs k consecutive available windows, so
    as k grows, participants with short probe sessions silently drop out of
    the comparison. Without pinning the identity set, "AUC rises with k"
    would be partly measuring "low-volume participants stop being scored"
    rather than "more accumulated evidence helps". Confirmed to matter on
    this cohort: the unpinned probe-identity count falls from 11 at k=1 to
    7 at k=20."""
    df = df.merge(splits[["sessionId", "role"]], on="sessionId", how="inner")

    if shuffle_seed is not None:
        # NEGATIVE CONTROL: permute which identity each PROBE session is
        # claimed to belong to, leaving enrolment and training untouched.
        #
        # This asks exactly the question that matters: is a probe session
        # closer to its OWN enrolment reference than to somebody else's? If
        # real-label performance is genuine identity signal, breaking that
        # correspondence should collapse it toward chance.
        #
        # An earlier version shuffled participantId across ALL sessions
        # while keeping roles fixed, which was wrong: a pseudo-identity
        # could end up owning a probe session but no enrolment session (or
        # several of one and none of the other), so the genuine/impostor
        # structure became malformed rather than merely randomised. That
        # inflated the control to ~0.6 and would have made a real result
        # look barely better than chance. Permuting only within the probe
        # role preserves exactly one enrol and one probe per identity.
        rng = np.random.default_rng(shuffle_seed)
        probe_mask = df["role"] == "probe"
        probe_pids = df.loc[probe_mask].groupby("sessionId")["participantId"].first()
        shuffled = rng.permutation(probe_pids.values)
        mapping = dict(zip(probe_pids.index, shuffled))
        df = df.copy()
        df.loc[probe_mask, "participantId"] = df.loc[probe_mask, "sessionId"].map(mapping)

    feats = select_features(df)
    if not feats:
        return [], []

    transform, fit_basis = fit_preprocessing(df[df["role"] == "train"], df, feats)

    # Reference vector per identity, from that identity's enrol session.
    references = {}
    for pid, g in df[df["role"] == "enrol"].groupby("participantId"):
        references[pid] = transform(g).mean(axis=0)
    if len(references) < 2:
        return [], []

    results, per_participant = [], []
    for k in k_values:
        y_true, scores, probe_owner, trust_id = [], [], [], []
        spans_all = []

        for pid, g in df[df["role"] == "probe"].groupby("participantId"):
            if fixed_identities is not None and pid not in fixed_identities:
                continue
            vecs, spans = build_trust_windows(g, transform, k, stride)
            if len(vecs) == 0:
                continue
            spans_all.extend(spans)
            for vi, v in enumerate(vecs):
                for cand, ref in references.items():
                    y_true.append(int(cand == pid))
                    scores.append(-float(np.linalg.norm(v - ref)))
                    probe_owner.append(pid)
                    trust_id.append(f"{pid}|{vi}")

        if not y_true:
            continue

        y_true = np.array(y_true)
        scores = np.array(scores)
        probe_owner = np.array(probe_owner)
        trust_id = np.array(trust_id)

        n_probe_identities = len(set(probe_owner))
        if n_probe_identities < MIN_PROBE_IDENTITIES:
            # Reported as a skipped row rather than silently dropped, so the
            # sweep does not appear to simply stop for no stated reason.
            results.append({
                "k_windows": k,
                "median_span_s": float(np.median(spans_all)) if len(spans_all) else np.nan,
                "n_probe_identities": n_probe_identities,
                "skipped_reason": f"fewer than {MIN_PROBE_IDENTITIES} probe identities",
            })
            continue

        metrics = compute_metrics(y_true, scores)

        # COHORT-NORMALISED SCORES. Raw distances carry a per-probe-window
        # offset -- some windows sit further from EVERY reference than others
        # do (unusual behaviour, sparse window, whatever), and pooling raw
        # distances across windows and identities lets that offset dominate
        # the genuine/impostor separation the metric is meant to measure.
        # Normalising each probe window's scores across the candidate set
        # removes it, and is standard practice in biometric verification
        # rather than a trick. Measured on this cohort it is worth ~+0.14 AUC
        # at single-window level for no modelling work at all, so the two are
        # reported side by side: quoting only the raw number understates the
        # baseline an embedder has to beat.
        norm_scores = np.copy(scores).astype(float)
        for tid in set(trust_id):
            m = trust_id == tid
            block = scores[m]
            sd = block.std()
            norm_scores[m] = (block - block.mean()) / (sd if sd > 0 else 1.0)
        norm_metrics = compute_metrics(y_true, norm_scores)
        row = {
            "k_windows": k,
            "median_span_s": float(np.median(spans_all)) if len(spans_all) else np.nan,
            "n_comparisons": int(len(y_true)),
            "n_trust_windows": int(len(y_true) / max(len(references), 1)),
            "n_probe_identities": n_probe_identities,
            "n_reference_identities": len(references),
            "n_features": len(feats),
            "preprocessing_fit_on": fit_basis,
            **metrics,
            **{f"{k_}_cohort_norm": v for k_, v in norm_metrics.items()},
        }

        # Dominance check: recompute leaving out one probe identity at a
        # time, so a single high-volume participant cannot silently carry
        # the headline number. Run symmetrically on every result, not only
        # on ones that look surprising.
        loo = []
        for pid in set(probe_owner):
            mask = probe_owner != pid
            if len(np.unique(y_true[mask])) == 2:
                loo.append(roc_auc_score(y_true[mask], scores[mask]))
        if loo:
            row["auc_leave_one_out_min"] = float(np.min(loo))
            row["auc_leave_one_out_max"] = float(np.max(loo))
        results.append(row)

        # Per-probe-identity breakdown: that identity's genuine scores vs
        # its impostor scores.
        if shuffle_seed is None:
            for pid in sorted(set(probe_owner)):
                mask = probe_owner == pid
                if len(np.unique(y_true[mask])) == 2:
                    per_participant.append({
                        "k_windows": k,
                        "participantId": pid,
                        "auc": float(roc_auc_score(y_true[mask], scores[mask])),
                        "n_comparisons": int(mask.sum()),
                    })

    return results, per_participant


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=["tap", "gesture", "motion", "typing"])
    ap.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    ap.add_argument("--trust-stride", type=int, default=1,
                    help="Step between successive trust windows. 1 uses every position "
                          "(maximum data, heavily overlapping); larger values thin them.")
    ap.add_argument("--device-family", default="mobile",
                    help="Restrict to one device family, or 'all'. Defaults to mobile: P1's "
                          "device-dependence finding means pooling across families risks "
                          "scoring device rather than person, and both desktop participants "
                          "here have a single session so are unevaluable regardless.")
    ap.add_argument("--control-seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds for the shuffled-label negative control.")
    ap.add_argument("--data-dir", default="data/processed/modelling")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits_path = data_dir / "identity_splits.csv"
    if not splits_path.exists():
        print(f"No {splits_path} -- run build_identity_splits.py first.")
        return
    splits = pd.read_csv(splits_path)

    all_results, all_per_participant = [], []

    for modality in args.modalities:
        path = data_dir / f"{modality}_windows.parquet"
        if not path.exists():
            print(f"[{modality}] no {path}, skipping")
            continue

        df = pd.read_parquet(path)
        if args.device_family != "all" and "deviceFamily" in df.columns:
            df = df[df["deviceFamily"] == args.device_family]
        if df.empty:
            print(f"[{modality}] no windows after device filter, skipping")
            continue

        # Identities with enough probe windows to survive the LARGEST k --
        # the only set that can be scored identically at every k, so the
        # only set on which the AUC-vs-k comparison is like-for-like.
        max_k = max(args.k_values)
        probe_sessions = df.merge(splits[["sessionId", "role"]], on="sessionId", how="inner")
        probe_sessions = probe_sessions[probe_sessions["role"] == "probe"]
        per_identity_max = probe_sessions.groupby(["participantId", "sessionId"]).size() \
                                          .groupby("participantId").max()
        fixed_identities = set(per_identity_max[per_identity_max >= max_k].index)

        results, per_participant = evaluate_modality(
            df, splits, args.k_values, args.trust_stride
        )
        if not results:
            print(f"[{modality}] not evaluable (too few reference identities or no probe windows)")
            continue

        for r in results:
            r["modality"] = modality
            r["identity_set"] = "all_available"
        for p in per_participant:
            p["modality"] = modality

        fixed_results = []
        if len(fixed_identities) >= 2:
            fixed_results, _ = evaluate_modality(
                df, splits, args.k_values, args.trust_stride,
                fixed_identities=fixed_identities
            )
            for r in fixed_results:
                r["modality"] = modality
                r["identity_set"] = "fixed_across_k"
        else:
            print(f"[{modality}] fewer than 2 identities have >= {max_k} probe windows -- "
                  f"the like-for-like fixed-identity comparison cannot be run for this modality.")

        # Negative control, run separately for each identity-set variant so
        # each real number is compared against a control computed on the
        # same identities.
        for variant_rows, fixed in ((results, None), (fixed_results, fixed_identities)):
            if not variant_rows:
                continue
            control = {}
            for seed in args.control_seeds:
                ctrl_results, _ = evaluate_modality(
                    df, splits, args.k_values, args.trust_stride,
                    shuffle_seed=seed, fixed_identities=fixed
                )
                for r in ctrl_results:
                    if "auc" in r:  # skipped rows carry no metrics
                        control.setdefault(r["k_windows"], []).append(r["auc"])
            for r in variant_rows:
                if "skipped_reason" in r:
                    continue
                vals = control.get(r["k_windows"], [])
                r["auc_shuffled_mean"] = float(np.mean(vals)) if vals else np.nan
                r["auc_shuffled_std"] = float(np.std(vals)) if vals else np.nan

        all_results.extend(results)
        all_results.extend(fixed_results)
        all_per_participant.extend(per_participant)

        show = ["k_windows", "median_span_s", "auc", "auc_cohort_norm", "eer_cohort_norm",
                "far_at_10pct_frr_cohort_norm", "auc_shuffled_mean",
                "n_trust_windows", "n_probe_identities"]
        print(f"\n=== {modality} | all available identities at each k ===")
        rdf = pd.DataFrame(results)
        for c in show:
            if c not in rdf.columns:
                rdf[c] = np.nan
        print(rdf[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if "skipped_reason" in rdf.columns and rdf["skipped_reason"].notna().any():
            for _, sk in rdf[rdf["skipped_reason"].notna()].iterrows():
                print(f"  k={int(sk['k_windows'])}: NOT REPORTED -- {sk['skipped_reason']}")
        if rdf["preprocessing_fit_on"].iloc[0] != "train_only":
            print(f"  NOTE: preprocessing fit on '{rdf['preprocessing_fit_on'].iloc[0]}' "
                  f"-- train split too thin for a train-only fit on this modality.")
        if rdf["n_probe_identities"].notna().any() and rdf["n_probe_identities"].nunique() > 1:
            print(f"  WARNING: probe-identity count varies across k "
                  f"({rdf['n_probe_identities'].min()}-{rdf['n_probe_identities'].max()}) -- "
                  f"AUC is NOT like-for-like across rows here. Use the fixed-identity table below.")

        if fixed_results:
            fdf = pd.DataFrame(fixed_results)
            for c in show:
                if c not in fdf.columns:
                    fdf[c] = np.nan
            print(f"\n=== {modality} | fixed identity set "
                  f"({len(fixed_identities)} identities with >= {max_k} probe windows) ===")
            print(fdf[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if not all_results:
        print("\nNo results produced.")
        return

    out_dir = data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    res_df = pd.DataFrame(all_results)
    cols = ["modality", "identity_set", "k_windows", "median_span_s",
            "auc", "eer", "far_at_10pct_frr",
            "auc_cohort_norm", "eer_cohort_norm", "far_at_10pct_frr_cohort_norm",
            "skipped_reason",
            "auc_shuffled_mean", "auc_shuffled_std", "auc_leave_one_out_min",
            "auc_leave_one_out_max", "n_trust_windows", "n_comparisons",
            "n_probe_identities", "n_reference_identities", "n_features",
            "preprocessing_fit_on"]
    res_df = res_df[[c for c in cols if c in res_df.columns]]
    res_df.to_csv(out_dir / "trust_window_results.csv", index=False)
    print(f"\nWrote {out_dir / 'trust_window_results.csv'}")

    if all_per_participant:
        pd.DataFrame(all_per_participant).to_csv(
            out_dir / "trust_window_per_participant.csv", index=False)
        print(f"Wrote {out_dir / 'trust_window_per_participant.csv'}")


if __name__ == "__main__":
    main()
