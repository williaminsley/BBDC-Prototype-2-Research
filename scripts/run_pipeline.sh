#!/usr/bin/env bash
#
# run_pipeline.sh
#
# Runs the BBDC Prototype 2 research pipeline end to end:
#
#   1. sync         sync_sessions_p2.py          Firebase -> data/raw/sessions/{sessionId}/
#   2. validate     validate_raw_sessions.py     PASS/WARN/FAIL report per session
#   3. raw          build_raw_dataset.py         canonical raw events (one row per event)
#   4. windows      build_windows_dataset.py     15s/7.5s sliding windows + features
#   5. qc           run_qc.py                    structural QC gate on the windowed dataset
#   6. diagnostics  build_feature_diagnostics.py per-feature screening table (availability,
#                                                 exposure/device/context confounds, redundancy)
#
# Halt semantics are deliberately not uniform across stages, because the scripts
# do not mean the same thing by a non-zero exit:
#
#   - validate_raw_sessions.py exits 1 if ANY session has structural or
#     behavioural issues. That is not a reason to stop: build_raw_dataset.py
#     intentionally admits behaviourally-poor sessions ("log everything, decide
#     later") and gates only on structural parseability. So this script reads
#     validate's JSON report instead of its exit code, prints every faulty
#     session with its reasons, and carries on with the good ones.
#
#   - build_raw_dataset.py exits 1 only when NO session survived its structural
#     gate. That is the real "nothing usable" condition, and it halts the run.
#
#   - sync and windows failures halt the run.
#
#   - run_qc.py exits 1 on any HARD FAIL (see scripts/run_qc.py for the
#     severity definitions) and 0 otherwise, including when it has soft
#     warnings -- so a QC run with only warnings does not halt the pipeline,
#     matching build_windows_dataset.py's completion being what "done"
#     means; qc is a check on that output, not a build step in its own right.
#
#   - build_feature_diagnostics.py failures halt the run, same as windows --
#     it is a straightforward build step (reads windows.parquet, writes
#     feature_diagnostics.csv/.parquet) with no partial-continuation logic.
#
# Usage:
#   ./run_pipeline.sh                    full run
#   ./run_pipeline.sh --skip-sync        rebuild from already-synced sessions
#   ./run_pipeline.sh --from raw         resume at the canonical raw build
#   ./run_pipeline.sh --skip-csv         parquet only (faster, smaller)
#
set -euo pipefail

# Resolve to the repo root so the run does not depend on the caller's cwd. This
# matters: sync_sessions_p2.py resolves paths from __file__, but the other three
# default to cwd-relative "data/...".
#
# The repo root is located by looking for the pipeline scripts themselves, so
# this file works whether it sits at the repo root or inside scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTINEL="scripts/sync_sessions_p2.py"

if [[ -f "${SCRIPT_DIR}/${SENTINEL}" ]]; then
    REPO_ROOT="$SCRIPT_DIR"                          # this file is at the repo root
elif [[ -f "${SCRIPT_DIR}/../${SENTINEL}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"      # this file is inside scripts/
else
    echo "ERROR: could not locate the repo root from ${SCRIPT_DIR}." >&2
    echo "       Expected to find ${SENTINEL} there or one level up." >&2
    echo "       Place run_pipeline.sh at the repo root or in scripts/." >&2
    exit 2
fi
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
SCRIPTS_DIR="scripts"
REPORTS_DIR="reports"
LOGS_DIR="reports/logs"
VALIDATION_REPORT="${REPORTS_DIR}/session_export_validation_p2.json"

ALL_STAGES=(sync validate raw windows qc diagnostics)
START_STAGE="sync"
SKIP_CSV=0

# ---------------------------------------------------------------- args ------

usage() {
    cat <<'EOF'
run_pipeline.sh - BBDC Prototype 2 research pipeline runner

  --skip-sync        Skip the Firebase sync; use sessions already on disk.
  --from STAGE       Start at STAGE. One of: sync, validate, raw, windows, qc, diagnostics.
  --skip-csv         Pass --skip-csv to the raw and windows builds
                     (parquet only; skips the large, slow CSV copies).
  -h, --help         Show this message.

Environment:
  PYTHON             Interpreter to use (default: python3).
  GOOGLE_APPLICATION_CREDENTIALS
                     Service-account JSON. Required unless sync is skipped.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sync) START_STAGE="validate"; shift ;;
        --from)
            [[ $# -ge 2 ]] || { echo "ERROR: --from needs a stage name." >&2; exit 2; }
            START_STAGE="$2"; shift 2 ;;
        --from=*)    START_STAGE="${1#*=}"; shift ;;
        --skip-csv)  SKIP_CSV=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 2 ;;
    esac
done

START_INDEX=-1
for i in "${!ALL_STAGES[@]}"; do
    [[ "${ALL_STAGES[$i]}" == "$START_STAGE" ]] && START_INDEX=$i
done
if [[ $START_INDEX -lt 0 ]]; then
    echo "ERROR: unknown stage '$START_STAGE'. Expected one of: ${ALL_STAGES[*]}" >&2
    exit 2
fi

should_run() {
    local target="$1"
    for i in "${!ALL_STAGES[@]}"; do
        if [[ "${ALL_STAGES[$i]}" == "$target" ]]; then
            [[ $i -ge $START_INDEX ]] && return 0 || return 1
        fi
    done
    return 1
}

# Plain string, not an array: bash < 4.4 (which includes macOS's stock /bin/bash
# 3.2) treats "${empty_array[@]}" as an unbound variable under `set -u` and aborts.
# The flag is a single word with no spaces, so word-splitting is safe here.
CSV_FLAG=""
[[ $SKIP_CSV -eq 1 ]] && CSV_FLAG="--skip-csv"

# ------------------------------------------------------------- logging ------

mkdir -p "$LOGS_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOGS_DIR}/pipeline_${RUN_ID}.log"

# The whole run is piped through tee at the bottom of this file rather than via
# `exec > >(tee ...)`. Process substitution is not waited on by bash, so the tail
# of a run can be lost or printed after the script has already exited.
#
# Unbuffered so per-session progress from the long stages (windows especially)
# appears live instead of arriving in one block when the stage finishes.
export PYTHONUNBUFFERED=1

RUN_START=$SECONDS

fmt_elapsed() {
    local s=$1
    printf '%dm%02ds' $((s / 60)) $((s % 60))
}

banner() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

stage_done() {
    echo "---- $1 OK (elapsed $(fmt_elapsed $(($SECONDS - $2)))) ----"
}

fail() {
    echo
    echo "!!!! PIPELINE FAILED at stage: $1"
    echo "!!!! $2"
    echo "!!!! Log: $LOG_FILE"
    exit 1
}

run_pipeline_main() {

echo "BBDC Prototype 2 pipeline"
echo "  repo root : $REPO_ROOT"
echo "  run id    : $RUN_ID"
echo "  log       : $LOG_FILE"
echo "  starting  : $START_STAGE"
echo "  csv       : $([[ $SKIP_CSV -eq 1 ]] && echo 'skipped (parquet only)' || echo 'written')"

# ------------------------------------------------------------ preflight -----

banner "PREFLIGHT"

command -v "$PYTHON" >/dev/null 2>&1 \
    || fail preflight "Interpreter '$PYTHON' not found. Activate the venv, or set PYTHON=..."
echo "bash      : ${BASH_VERSION}"
echo "python    : $("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "venv      : $VIRTUAL_ENV"
else
    echo "venv      : WARNING - no VIRTUAL_ENV set; using the interpreter above"
fi

for stage_script in sync_sessions_p2.py validate_raw_sessions.py \
                    build_raw_dataset.py build_windows_dataset.py run_qc.py \
                    build_feature_diagnostics.py; do
    [[ -f "${SCRIPTS_DIR}/${stage_script}" ]] \
        || fail preflight "Missing ${SCRIPTS_DIR}/${stage_script}"
done
echo "scripts   : all six present in ${SCRIPTS_DIR}/"

"$PYTHON" - <<'PY' || fail preflight "Core dependencies missing. Run: pip install -r requirements.txt"
import sys
missing = []
for mod in ("pandas", "pyarrow", "numpy"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print("Missing modules: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("deps      : pandas, pyarrow, numpy importable")
PY

if should_run sync; then
    # sync_sessions_p2.py returns 0 when google-cloud is not installed - it just
    # prints "Skipping sync" and moves on. In a pipeline that is a silent no-op
    # that would leave the rest of the run building on stale data, so check here
    # and hard-fail instead.
    "$PYTHON" - <<'PY' || fail preflight "google-cloud-firestore / google-cloud-storage not installed. Without them sync_sessions_p2.py silently no-ops and the pipeline would rebuild from stale sessions. Run: pip install -r requirements.txt (or use --skip-sync)."
import sys
try:
    from google.cloud import firestore, storage  # noqa: F401
except ImportError as e:
    print(f"google-cloud import failed: {e}", file=sys.stderr)
    sys.exit(1)
print("firebase  : google-cloud-firestore + storage importable")
PY

    CREDS="${GOOGLE_APPLICATION_CREDENTIALS:-}"
    [[ -n "$CREDS" ]] \
        || fail preflight "GOOGLE_APPLICATION_CREDENTIALS is not set (or use --skip-sync)."
    [[ -f "$CREDS" ]] \
        || fail preflight "GOOGLE_APPLICATION_CREDENTIALS points at a missing file: $CREDS"
    echo "creds     : $CREDS"
else
    echo "creds     : not checked (sync not in this run)"
fi

echo "PREFLIGHT OK"

# --------------------------------------------------------- 1. sync ---------

if should_run sync; then
    banner "STAGE 1/6  sync  (sync_sessions_p2.py)"
    T0=$SECONDS
    "$PYTHON" "${SCRIPTS_DIR}/sync_sessions_p2.py" \
        || fail sync "sync_sessions_p2.py exited non-zero (missing credentials, or at least one session failed to download). Fix the cause, or re-run with --skip-sync to build from what is already on disk."
    stage_done "sync" "$T0"
else
    echo
    echo "---- SKIPPED stage 1/6  sync ----"
fi

# ------------------------------------------------------ 2. validate --------

if should_run validate; then
    banner "STAGE 2/6  validate  (validate_raw_sessions.py)"
    T0=$SECONDS

    # Remove any previous run's report first. Otherwise, if this run's validate
    # crashes before writing, the checks below would silently read a stale report
    # and make halt/continue decisions on last week's sessions.
    rm -f "$VALIDATION_REPORT"

    # Exit code is deliberately ignored: it goes to 1 if ANY session has issues,
    # and a faulty session is not a reason to abandon the good ones. The JSON
    # report is written before that return, so it is present either way, and is
    # what we actually judge the run on.
    set +e
    "$PYTHON" "${SCRIPTS_DIR}/validate_raw_sessions.py"
    VALIDATE_RC=$?
    set -e
    echo "(validate_raw_sessions.py exit code: ${VALIDATE_RC} - not treated as fatal by itself)"

    [[ -f "$VALIDATION_REPORT" ]] \
        || fail validate "validate_raw_sessions.py did not write ${VALIDATION_REPORT}, so it failed before completing its checks. Exit code was ${VALIDATE_RC}."

    echo
    "$PYTHON" - "$VALIDATION_REPORT" <<'PY' || fail validate "Zero sessions available to build from. Run a sync, or check data/raw/sessions."
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
sessions = report.get("sessions", {})
scanned = report.get("sessions_scanned", 0)

print("VALIDATION SUMMARY")
print(f"  verdict      : {report.get('verdict')}")
print(f"  scanned      : {scanned} session(s)")
print(f"  participants : {report.get('participants_found')}")
print(f"  pass / warn / fail : "
      f"{report.get('pass_sessions')} / {report.get('warn_sessions')} / {report.get('fail_sessions')}")

for check in report.get("global_checks", []):
    print(f"  global       : {check}")

flagged = [
    (sid, info) for sid, info in sessions.items()
    if info.get("structural_issues") or info.get("behavioural_issues")
]

if flagged:
    print()
    print(f"FLAGGED SESSIONS ({len(flagged)}) - reasons below.")
    print("Structural issues are excluded by build_raw_dataset.py.")
    print("Behavioural issues are KEPT in the raw layer and filtered downstream.")
    for sid, info in flagged:
        print(f"\n  {sid}")
        ctx = info.get("context") or {}
        bits = [f"{k}={v}" for k, v in ctx.items() if v not in (None, "", "nan")]
        if bits:
            print(f"    context: {', '.join(bits)}")
        print(f"    stats  : events={info.get('eventCount')}, "
              f"tasks={info.get('taskCount')}/{info.get('expectedTaskCount')}, "
              f"completedNormally={info.get('completedNormally')}, "
              f"device={info.get('deviceFamily')}")
        for issue in info.get("structural_issues") or []:
            print(f"    STRUCTURAL : {issue}")
        for issue in info.get("behavioural_issues") or []:
            print(f"    BEHAVIOURAL: {issue}")
else:
    print("\n  No sessions flagged with structural or behavioural issues.")

if scanned == 0:
    print("\nHALT: zero sessions found in data/raw/sessions.", file=sys.stderr)
    sys.exit(1)

print("\nContinuing with all structurally-sound sessions.")
PY

    echo "  full report: ${VALIDATION_REPORT}"
    echo "               ${REPORTS_DIR}/session_export_validation_p2.md"
    stage_done "validate" "$T0"
else
    echo
    echo "---- SKIPPED stage 2/6  validate ----"
fi

# ----------------------------------------------------------- 3. raw --------

if should_run raw; then
    banner "STAGE 3/6  raw  (build_raw_dataset.py)"
    T0=$SECONDS
    # Exits 1 only when no session survived the structural gate - the real
    # "nothing usable" condition. Per-session EXCLUDED lines are printed above.
    "$PYTHON" "${SCRIPTS_DIR}/build_raw_dataset.py" $CSV_FLAG \
        || fail raw "build_raw_dataset.py exited non-zero. If it reported 'No included sessions', every session failed the structural gate - see the EXCLUDED lines above and the validation report."
    stage_done "raw" "$T0"
else
    echo
    echo "---- SKIPPED stage 3/6  raw ----"
fi

# ------------------------------------------------------- 4. windows --------

if should_run windows; then
    banner "STAGE 4/6  windows  (build_windows_dataset.py)"
    T0=$SECONDS
    "$PYTHON" "${SCRIPTS_DIR}/build_windows_dataset.py" $CSV_FLAG \
        || fail windows "build_windows_dataset.py exited non-zero. If it reported 'No windows produced', every session was shorter than one window length."
    stage_done "windows" "$T0"
else
    echo
    echo "---- SKIPPED stage 4/6  windows ----"
fi

# ------------------------------------------------------------ 5. qc --------

if should_run qc; then
    banner "STAGE 5/6  qc  (run_qc.py)"
    T0=$SECONDS
    # run_qc.py exits 1 on any HARD FAIL, 0 otherwise (soft warnings included).
    # Its full report goes to stdout, which this pipeline already captures via
    # the tee at the bottom of the file -- no separate report file to check for,
    # unlike validate's JSON report, since qc has nothing analogous to
    # validate's "keep going with the good sessions" partial-continuation logic:
    # the windowed dataset either passes structural QC as a whole or it does not.
    "$PYTHON" "${SCRIPTS_DIR}/run_qc.py" \
        || fail qc "run_qc.py reported a HARD FAIL against data/processed/windows.parquet. See the QC report above for which check(s) failed."
    stage_done "qc" "$T0"
else
    echo
    echo "---- SKIPPED stage 5/6  qc ----"
fi

# ------------------------------------------------------ 6. diagnostics -----

if should_run diagnostics; then
    banner "STAGE 6/6  diagnostics  (build_feature_diagnostics.py)"
    T0=$SECONDS
    # Straightforward build step, same halt semantics as windows: reads
    # data/processed/windows.parquet, writes feature_diagnostics.csv/.parquet.
    # No partial-continuation logic -- either it completes or it doesn't.
    "$PYTHON" "${SCRIPTS_DIR}/build_feature_diagnostics.py" \
        || fail diagnostics "build_feature_diagnostics.py exited non-zero. Check that data/processed/windows.parquet exists and is readable (the windows stage must have completed first)."
    stage_done "diagnostics" "$T0"
else
    echo
    echo "---- SKIPPED stage 6/6  diagnostics ----"
fi

# ---------------------------------------------------------------- done -----

banner "PIPELINE COMPLETE"
echo "  total elapsed : $(fmt_elapsed $(($SECONDS - $RUN_START)))"
echo
echo "  outputs:"
for f in data/processed/raw_events.parquet \
         data/processed/raw_events_session_manifest.csv \
         data/processed/raw_events_schema.json \
         data/processed/windows.parquet \
         data/processed/windows_schema.json \
         data/processed/feature_diagnostics.csv \
         data/processed/feature_diagnostics.parquet \
         "$VALIDATION_REPORT"; do
    if [[ -f "$f" ]]; then
        echo "    $(du -h "$f" | cut -f1)	$f"
    else
        echo "    --	$f  (not written)"
    fi
done
echo
echo "  log : $LOG_FILE"

}

run_pipeline_main 2>&1 | tee -a "$LOG_FILE"
# pipefail is set, so the pipeline's status is run_pipeline_main's, not tee's.
exit "${PIPESTATUS[0]}"
