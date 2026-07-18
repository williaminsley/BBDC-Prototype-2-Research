# Setting up `BBDC-Prototype-2-Research`

This is a practical setup guide for a new repo that loads and processes P2 (Pulse Bank Review)
session data, developed in the same style as `BBDC-Prototype-1-Research`. It's written against
what's actually verified in both codebases — `williaminsley/BBDC-Prototype-1-Research`
(`repo_map.md`, `sync_storage_sessions.py`, `requirements.txt`) and the P2 app's `firebase.js` /
`schema.js` — not assumed from memory.

## 1. The one thing that's structurally different from P1, and why it changes everything downstream

P1's `docs/app.js` computed **windows in the browser** and uploaded two files per session:
`events.csv` (raw events) and `auth_windows.csv` (pre-aggregated 30s windows). The Python pipeline's
job was to *concatenate* those already-shaped files.

P2 uploads **one file per session**: `sessions_p2/{sessionId}/session.json`, containing a single
flat `events` array (7,000+ rows for a full session) plus session-level metadata. There is no
browser-side windowing — `schema.js`'s `recommendedWindowMs: 7500` / `recommendedStepMs: 2500` are
just recommendations for the *Python* pipeline to implement. This means:

- there's no `auth_windows.csv` equivalent to sync or concatenate — **window construction has to
  be written from scratch in the P2 pipeline**, operating on canonical raw events, the way P1's
  `build_windows_dataset.py` operates on already-windowed CSVs today. It's the same pipeline
  *position* (raw → windows) but genuinely new logic, not a port.
- `events` is nested inside JSON, not a flat CSV — the canonical-raw-dataset build step needs a
  flattening pass (see §4) that P1 didn't need, because P1's `events.csv` was already flat.
- P2's Firestore `sessions_p2` collection already carries useful manifest fields
  (`usableForSignalExtraction`, `deviceFamily`, `eventCount`, `schemaVersion`, `storagePath`) that
  P1 never had — P1 only ever listed Storage directly with no Firestore step. It's worth using this
  as the sync manifest instead of re-deriving it from Storage listing (see §3).

Also worth carrying over deliberately: the P2 `uploadStatus` field can be stuck at `"attempting"`
even for what look like complete, well-formed sessions (confirmed in `session2.json` — 26/26 tasks,
clean `qualitySummary`, but `uploadStatus: "attempting"`, not `"success"`). Don't filter sync on
`uploadStatus` alone; use the Firestore doc's presence plus `session.json`'s own `completedNormally`
and `qualitySummary` as the real signal, and log `uploadStatus` as metadata to investigate, not as a
gate.

## 2. Recommended repo structure

Mirror `BBDC-Prototype-1-Research`'s layout — same names where the role is the same, so a reader who
knows P1 can navigate P2 immediately:

```text
BBDC-Prototype-2-Research/
├── README.md
├── project_explanation.md
├── repo_map.md
├── requirements.txt
└── Behavioural-Biometrics-Analysis/
    ├── data/
    │   ├── raw/sessions/{sessionId}/
    │   │   ├── session.json          # synced from Storage
    │   │   └── firestore_meta.json   # synced from Firestore — new vs P1, see §3
    │   └── processed/
    │       ├── raw_events.parquet / .csv / _schema.json / _session_manifest.csv
    │       ├── windows.parquet / .csv / _schema.json        # NEW logic, not a port — see §1
    │       └── behavioural_windows_candidate.parquet / .csv / _schema.json
    ├── Feature-Dictionary/
    │   ├── behavioural_feature_dictionary.md
    │   ├── behavioural_feature_dictionary_with_examples.csv
    │   └── raw_feature_understanding.md
    ├── notebooks/
    │   ├── data-audit/00_data_audit.ipynb
    │   └── modelling/
    │       ├── 01_exploring_raw_data.ipynb
    │       ├── 02_raw_distribution_explorer.ipynb
    │       └── 03_raw_time_series_explorer.ipynb
    ├── reports/
    │   ├── qc_summary.md / .json
    │   ├── prelaunch_validation.md / .json
    │   └── behavioural_column_manifest.csv
    └── scripts/
        ├── sync_sessions_p2.py        # see §3, starter script provided
        ├── validate_raw_sessions.py   # adapt from P1, see §4
        ├── build_raw_dataset.py       # adapt from P1, see §4 — the flattening step
        ├── build_windows_dataset.py   # NEW logic, see §1
        ├── build_behavioural_dataset.py  # adapt from P1, see §5
        ├── run_qc.py                  # adapt from P1, see §5
        ├── run_pipeline.sh
        └── run_reports.sh
```

Two notebooks you've already built for P2 in this project slot straight in:
`01_raw_behavioural_explorer.ipynb` → `notebooks/data-audit/` (it's app-agnostic, structural), and
`02_bank_session_signal_explorer.ipynb` → `notebooks/modelling/` (it's schema-specific, descriptive).

## 3. `scripts/sync_sessions_p2.py`

P1's `sync_storage_sessions.py` lists Storage directly (`client.list_blobs(bucket, prefix="sessions/")`)
and infers session IDs from blob paths, because there was no Firestore manifest to use instead.

For P2, use the `sessions_p2` **Firestore collection** as the manifest and Storage only for the blob
download — it's already being written correctly by `firebase.js`'s `uploadSessionToFirebase`, and it
gives you `deviceFamily`, `usableForSignalExtraction`, `eventCount`, and `storagePath` per session
without re-deriving anything. A starter script implementing this is provided separately
(`sync_sessions_p2.py`) — same credential pattern as P1 (`GOOGLE_APPLICATION_CREDENTIALS` pointing
at a service-account JSON with Firestore read + Storage read on the `behavioural-biometrics-b52e4`
project), plus `google-cloud-firestore` as a new dependency alongside P1's existing
`google-cloud-storage`.

If you'd rather keep sync mechanically identical to P1 (simpler, one less dependency, no Firestore
coupling), the fallback is listing `sessions_p2/` in Storage directly the way P1 lists `sessions/` —
the starter script has this as a commented-out alternative path.

## 4. `validate_raw_sessions.py` and `build_raw_dataset.py`

**Validation** checks change shape but not spirit: instead of checking for `events.csv` +
`auth_windows.csv` existing with required columns, check that `session.json` parses, that
`schemaName == "continuous_auth_behavioural_biometrics_app"`, that `schemaVersion` meets a minimum
(`1.2.0` at time of writing), that `events` is a non-empty list, and that the `sessionId` inside the
JSON matches the folder name / Firestore doc ID — same "trust but verify the folder-name/session-ID
agreement" check P1 does, just against a different file.

**Canonical raw dataset build** now needs a flattening pass P1 never needed: for each event in
`session.json`'s `events` array, hoist `schema.js`'s `commonTopLevelFields` (`kind`, `tRelMs`,
`timestampIso`, `taskId`, `taskIndex`, `screenId`, `componentId`, `activeArea`, `instructionArea`)
into their own columns, and either keep `payload` as a JSON-string column or explode it — payload
shape varies by `kind` (compare `touchstart`'s `force`/`radiusX`/`centroidX` against
`deviceorientation`'s `alpha`/`beta`/`gamma`), so a single flat schema across all 44 event kinds
isn't realistic; family-specific flattening (grouped by `kind`, the same pattern your
`01_raw_behavioural_explorer.ipynb` already uses) is the more honest approach than forcing one wide
table.

## 5. `build_behavioural_dataset.py` and `run_qc.py`

Same role as P1 — feature-family inclusion/exclusion rules, and hard-fail/soft-warning QC gates —
but the concrete rules need re-deriving once `build_windows_dataset.py` exists and you can see what
real P2 window-level columns look like. Don't port P1's exact prefix lists
(`typing_`/`tap_`/`pointer_`) verbatim; P2 has new families (`drag_`, `swipe_`, `motion_`,
`orientation_`, `scroll_`) that P1's filter never had to classify, plus the context-as-fusion-weight
design goal means some context columns should deliberately stay in as model inputs this time, not
just kept as metadata the way P1 treats them.

## 6. Suggested build order

1. `sync_sessions_p2.py` — get real session JSONs landing in `data/raw/sessions/`
2. `validate_raw_sessions.py` — confirm structural integrity across whatever's been collected so far
3. `build_raw_dataset.py` — canonical flattened raw-events table
4. `00_data_audit.ipynb` (adapt `01_raw_behavioural_explorer.ipynb`) — look at what you actually have
   before writing windowing logic against assumptions
5. `build_windows_dataset.py` — the genuinely new piece, 7.5s/2.5s rolling windows
6. `build_behavioural_dataset.py` + `run_qc.py` — once window-level columns exist to filter and gate
