"""
sync_sessions_p2.py

Syncs P2 session data from Firebase into data/raw/sessions/{sessionId}/.

Primary approach: use the `sessions_p2` Firestore collection as the manifest
(written by firebase.js's uploadSessionToFirebase on every upload), then
download each session's single JSON blob from Storage using the
`storagePath` field on the Firestore doc. This avoids re-deriving session
IDs and metadata from Storage listing, unlike P1's sync_storage_sessions.py,
because P2's Firestore doc already carries deviceFamily, eventCount,
usableForSignalExtraction, schemaVersion, etc.

Requires:
    pip install google-cloud-firestore google-cloud-storage

Auth:
    Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON with
    Firestore read + Storage read access on the behavioural-biometrics-b52e4
    project (same project as P1 — check whether the existing P1
    service-account already has these roles before creating a new one).
"""

import json
import os
from pathlib import Path

try:
    from google.cloud import firestore, storage
    from google.auth.exceptions import DefaultCredentialsError
    from google.api_core.exceptions import Forbidden, NotFound
except ModuleNotFoundError:
    firestore = None
    storage = None
    DefaultCredentialsError = Exception
    Forbidden = Exception
    NotFound = Exception

# ---------------- CONFIG ----------------
BUCKET_NAME = "behavioural-biometrics-b52e4.firebasestorage.app"
FIRESTORE_COLLECTION = "sessions_p2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_ROOT = PROJECT_ROOT / "data" / "raw" / "sessions"
# -----------------------------------------


def check_credentials():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print(
            "Sync failed: GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Set it to a service-account JSON file with Firestore read + Storage read access."
        )
        return None
    if not Path(creds_path).is_file():
        print(f"Sync failed: GOOGLE_APPLICATION_CREDENTIALS path does not exist: {creds_path}")
        return None
    return creds_path


def sync_via_firestore_manifest():
    """Primary path: Firestore sessions_p2 collection as manifest, Storage for the blob."""
    fs_client = firestore.Client()
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    print(f"Reading manifest from Firestore collection '{FIRESTORE_COLLECTION}'...")
    docs = list(fs_client.collection(FIRESTORE_COLLECTION).stream())
    print(f"Found {len(docs)} session record(s) in Firestore")

    remote_session_ids = set()
    downloaded, skipped, failed = 0, 0, 0

    for doc in docs:
        data = doc.to_dict()
        sid = data.get("sessionId") or doc.id
        storage_path = data.get("storagePath")
        remote_session_ids.add(sid)

        if not storage_path:
            print(f"Skipping {sid}: no storagePath on Firestore doc")
            skipped += 1
            continue

        local_dir = LOCAL_ROOT / sid
        local_dir.mkdir(parents=True, exist_ok=True)

        # Save the Firestore doc alongside the raw session for later cross-checking
        # (e.g. uploadStatus on the Firestore doc vs completedNormally inside session.json —
        # these can disagree, don't assume they're always in sync).
        meta_path = local_dir / "firestore_meta.json"
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        session_path = local_dir / "session.json"
        if session_path.exists():
            skipped += 1
            continue

        blob = bucket.blob(storage_path)
        tmp_path = local_dir / "session.json.tmp"
        try:
            blob.download_to_filename(tmp_path)
            tmp_path.replace(session_path)
            print(f"Downloaded {storage_path}")
            downloaded += 1
        except (NotFound, Forbidden) as e:
            print(f"Failed downloading {storage_path}: {e}")
            tmp_path.unlink(missing_ok=True)
            failed += 1

    # Remove stale local sessions no longer present in Firestore
    if LOCAL_ROOT.exists():
        local_session_ids = {p.name for p in LOCAL_ROOT.iterdir() if p.is_dir()}
        stale = local_session_ids - remote_session_ids
        for sid in sorted(stale):
            print(f"Note: local session {sid} is no longer in Firestore manifest (not auto-deleted; review manually)")

    print(f"Done. Downloaded {downloaded}, skipped {skipped} (already local / no path), failed {failed}.")
    return 0 if failed == 0 else 1


def sync_via_storage_listing_fallback():
    """
    Fallback path, mechanically identical to P1's sync_storage_sessions.py:
    list Storage directly under sessions_p2/ instead of using Firestore.
    Use this if you'd rather not add a Firestore dependency, or if the
    Firestore manifest and Storage ever disagree and you need a ground-truth
    listing straight from Storage.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    print("Listing sessions_p2/ in Storage directly...")
    blobs = list(storage_client.list_blobs(bucket, prefix="sessions_p2/"))

    session_ids = set()
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) >= 3 and parts[1]:
            session_ids.add(parts[1])

    print(f"Found {len(session_ids)} session(s)")

    for sid in sorted(session_ids):
        local_dir = LOCAL_ROOT / sid
        local_dir.mkdir(parents=True, exist_ok=True)
        session_path = local_dir / "session.json"
        if session_path.exists():
            continue

        blob_path = f"sessions_p2/{sid}/session.json"
        blob = bucket.blob(blob_path)
        tmp_path = local_dir / "session.json.tmp"
        try:
            blob.download_to_filename(tmp_path)
            tmp_path.replace(session_path)
            print(f"Downloaded {blob_path}")
        except Exception as e:
            print(f"Failed downloading {blob_path}: {e}")
            tmp_path.unlink(missing_ok=True)

    print("Done.")
    return 0


def main():
    if firestore is None or storage is None:
        print(
            "Skipping sync: google-cloud-firestore / google-cloud-storage are not installed. "
            "Install dependencies from requirements.txt to enable sync."
        )
        return 0

    if check_credentials() is None:
        return 1

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        return sync_via_firestore_manifest()
    except DefaultCredentialsError as e:
        print(f"Sync failed: could not load credentials: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
