"""
scripts/modelling/build_identity_splits.py

Assigns every session a role -- train / enrol / probe -- following P1's
session-safe rule (P1 reference doc Section 8.1). Windows within a session
overlap in time (15s length, 7.5s stride), so no window from a session may
appear both in what teaches the system a person's behaviour and in what
tests whether it recognises them. Splitting at SESSION level is what
guarantees that; splitting at window level would not.

The rule, per participant, by number of sessions available:

    1 session   -> all windows train-only. Contributes negative/impostor
                   examples but is never evaluated: there is no independent
                   second session to test against.
    2 sessions  -> no training contribution. First session enrols, second
                   probes.
    3+ sessions -> earlier sessions train, second-to-last enrols, last
                   probes.

CHRONOLOGICAL ORDER: derived from `sessionIndex`, the app's incrementing
per-browser session counter (logger.js). This is P1's documented
second-preference ordering signal (explicit order field > numeric session
index > recorded date > sessionId string, which is a random token and must
never be sorted on). Verified on this cohort: sessionIndex is constant
within each session and unique within each participant, with no duplicates
-- so it induces a strict order per participant.

Caveat worth knowing: sessionIndex comes from localStorage, so it would
reset if a participant cleared browser storage or switched browsers. The
`identitySource` field records when identity came from a fallback rather
than persistent localStorage, and is carried through to the output here so
that case can be audited rather than silently trusted.

ROLES ARE GLOBAL, NOT PER-MODALITY. Deliberate: a given session's role must
not change depending on which modality is being looked at, or the same
session could enrol one modality while probing another -- an obvious leak
across the fusion boundary. A consequence is that a session assigned
enrol/probe may contain no eligible windows for a sparse modality (typing
especially); that participant is then simply not evaluable on that
modality, which the evaluation harness reports rather than works around.

OUTPUT
------
data/processed/modelling/identity_splits.csv
    One row per session: participantId, sessionId, sessionIndex, role,
    n_windows, plus per-participant context (n_sessions, evaluable).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_splits(df: pd.DataFrame) -> pd.DataFrame:
    required = {"participantId", "sessionId", "sessionIndex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required column(s): {sorted(missing)}")

    sessions = (
        df.groupby(["participantId", "sessionId"])
        .agg(
            sessionIndex=("sessionIndex", "first"),
            n_windows=("sessionId", "size"),
            deviceFamily=("deviceFamily", "first") if "deviceFamily" in df.columns else ("sessionId", "size"),
            identitySource=("identitySource", "first") if "identitySource" in df.columns else ("sessionId", "size"),
        )
        .reset_index()
    )

    # Guard the ordering assumption rather than trusting it: if sessionIndex
    # ever repeats within a participant, the order is ambiguous and the
    # enrol/probe assignment below would be arbitrary.
    dupes = sessions.groupby("participantId")["sessionIndex"].apply(lambda s: s.duplicated().any())
    if dupes.any():
        bad = dupes[dupes].index.tolist()
        raise ValueError(
            f"Duplicate sessionIndex values within participant(s) {bad} -- chronological order "
            f"is ambiguous, so enrol/probe assignment cannot be made safely. Investigate before "
            f"proceeding (possible localStorage reset or browser switch; check identitySource)."
        )

    rows = []
    for pid, g in sessions.groupby("participantId"):
        g = g.sort_values("sessionIndex")
        n = len(g)
        for pos, (_, s) in enumerate(g.iterrows()):
            if n == 1:
                role = "train"
            elif n == 2:
                role = "enrol" if pos == 0 else "probe"
            else:
                if pos == n - 1:
                    role = "probe"
                elif pos == n - 2:
                    role = "enrol"
                else:
                    role = "train"
            rows.append({
                "participantId": pid,
                "sessionId": s["sessionId"],
                "sessionIndex": s["sessionIndex"],
                "session_position": pos + 1,
                "n_sessions_for_participant": n,
                "role": role,
                "n_windows": s["n_windows"],
                "deviceFamily": s.get("deviceFamily"),
                "identitySource": s.get("identitySource"),
                "evaluable": n >= 2,
            })

    splits = pd.DataFrame(rows).sort_values(["participantId", "sessionIndex"]).reset_index(drop=True)
    return splits


def summarise(splits: pd.DataFrame) -> None:
    n_participants = splits["participantId"].nunique()
    evaluable = splits.loc[splits["evaluable"], "participantId"].nunique()
    print(f"[splits] {len(splits)} sessions across {n_participants} participants")
    print(f"[splits] evaluable identities (2+ sessions): {evaluable} of {n_participants}")
    print()
    print(splits["role"].value_counts().rename("sessions").to_frame().to_string())
    print()
    print("windows by role:")
    print(splits.groupby("role")["n_windows"].sum().to_string())
    print()
    train_only = splits.loc[~splits["evaluable"], "participantId"].unique()
    if len(train_only):
        print(f"[splits] train-only participants (1 session, never evaluated): {sorted(train_only)}")


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
        print("No behavioural_windows_candidate.parquet found -- run build_behavioural_dataset.py first.")
        sys.exit(0)

    df = pd.read_parquet(path)
    print(f"Loaded {path}: {len(df):,} windows\n")

    splits = build_splits(df)
    summarise(splits)

    out_dir = Path("data/processed/modelling")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "identity_splits.csv"
    splits.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
