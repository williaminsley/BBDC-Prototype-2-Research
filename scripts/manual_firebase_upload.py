#!/usr/bin/env python3
"""
manual_upload_recovered_session.py

For a session that was manually recovered off-device (e.g. via downloadSessionJson()
after an automatic upload failed) and never went through firebase.js's normal
anonymous-sign-in -> uploadBytes -> addDoc flow.

Mirrors that flow directly using the same service-account credentials
sync_sessions_p2.py already uses, so the result is indistinguishable from a normal
upload: the blob lands at the same Storage path, and the Firestore doc carries the
same fields validate_raw_sessions.py's consistency check (sessionId, eventCount,
usableForSignalExtraction, uploadStatus) expects.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    python3 manual_upload_recovered_session.py data/raw/sessions/<sessionId>/session.json

After running, a normal `sync_sessions_p2.py` (or the next full pipeline run without
--skip-sync) will download this session's firestore_meta.json like any other, and the
validation report's false "missing firestore_meta" flag for this session will clear.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from google.cloud import firestore, storage
except ModuleNotFoundError:
    print("google-cloud-firestore / google-cloud-storage not installed. "
          "Run: pip install -r requirements.txt")
    sys.exit(1)

BUCKET_NAME = "behavioural-biometrics-b52e4.firebasestorage.app"
FIRESTORE_COLLECTION = "sessions_p2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_json_path", type=str,
                    help="Path to the recovered session.json (e.g. data/raw/sessions/<id>/session.json)")
    p.add_argument("--dry-run", action="store_true",
                    help="Validate and print what would be uploaded, without writing to Firebase.")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.session_json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    session = json.loads(path.read_text(encoding="utf-8"))
    session_id = session.get("sessionId")
    participant_id = session.get("participantId")
    events = session.get("events") or []

    if not session_id:
        print("session.json has no sessionId — refusing to upload something unidentifiable.")
        return 1

    if path.parent.name != session_id:
        print(f"WARNING: folder name '{path.parent.name}' does not match sessionId "
              f"'{session_id}'. Continuing anyway, but check this is the file you intend.")

    storage_path = f"sessions_p2/{session_id}/session.json"
    device_family = (session.get("device") or {}).get("deviceFamily")
    usable = (session.get("qualitySummary") or {}).get("usableForSignalExtraction")

    firestore_doc = {
        "sessionId": session_id,
        "participantId": participant_id,
        "storagePath": storage_path,
        "deviceFamily": device_family,
        "eventCount": len(events),
        "usableForSignalExtraction": usable,
        "schemaVersion": session.get("schemaVersion"),
        # This is the real upload-success signal validate_raw_sessions.py looks for —
        # deliberately set only after the Storage blob write below succeeds, same
        # ordering firebase.js uses (addDoc only runs after uploadBytes succeeds).
        "uploadStatus": "success",
        "recoveryMethod": "manual_offline_recovery",
    }

    print(f"sessionId      : {session_id}")
    print(f"participantId  : {participant_id}")
    print(f"event count    : {len(events)}")
    print(f"storage path   : gs://{BUCKET_NAME}/{storage_path}")
    print(f"firestore doc  : {FIRESTORE_COLLECTION}/{session_id}")
    print(json.dumps(firestore_doc, indent=2))

    if args.dry_run:
        print("\n--dry-run set — nothing uploaded.")
        return 0

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_path)

    if blob.exists():
        print(f"\nERROR: {storage_path} already exists in Storage. Refusing to overwrite — "
              "if this session genuinely did upload already, no manual step is needed.")
        return 1

    blob.upload_from_string(path.read_text(encoding="utf-8"), content_type="application/json")
    print(f"\nUploaded blob to gs://{BUCKET_NAME}/{storage_path}")

    fs_client = firestore.Client()
    doc_ref = fs_client.collection(FIRESTORE_COLLECTION).document(session_id)
    if doc_ref.get().exists:
        print(f"ERROR: Firestore doc {FIRESTORE_COLLECTION}/{session_id} already exists. "
              "Blob was uploaded but the doc was left untouched — check manually.")
        return 1

    doc_ref.set(firestore_doc)
    print(f"Wrote Firestore doc {FIRESTORE_COLLECTION}/{session_id}")
    print("\nDone. Next sync_sessions_p2.py run will pick this up like any normal session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
