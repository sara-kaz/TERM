"""
GCS checkpoint sync helper — replaces gsutil in NRP pods.

Usage:
  python scripts/gcs_sync.py download --bucket BUCKET --prefix PREFIX \\
      --state_key state.json --dest /results/checkpoint
  exit 0  → checkpoint found and downloaded; set RESUME_FLAG
  exit 1  → no checkpoint; start fresh

  python scripts/gcs_sync.py upload --bucket BUCKET --prefix PREFIX \\
      --src /results/checkpoint
"""
import argparse
import os
import sys


def download(args):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    probe = args.prefix.rstrip("/") + "/" + args.state_key
    if not list(bucket.list_blobs(prefix=probe)):
        print(f"[gcs] No checkpoint ({args.state_key}) — starting fresh.")
        sys.exit(1)

    print(f"[gcs] Checkpoint found — downloading to {args.dest}")
    os.makedirs(args.dest, exist_ok=True)
    prefix = args.prefix.rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel:
            continue
        local_path = os.path.join(args.dest, rel)
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"  download: {rel}")
    sys.exit(0)


def upload(args):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = args.prefix.rstrip("/")

    for root, dirs, files in os.walk(args.src):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, args.src)
            blob_name = f"{prefix}/{rel}"
            bucket.blob(blob_name).upload_from_filename(local_path)
            print(f"  upload: {rel}")
    print("[gcs] Upload complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["download", "upload"])
    p.add_argument("--bucket",    required=True)
    p.add_argument("--prefix",    required=True)
    p.add_argument("--state_key", default="state.json",
                   help="Filename to probe for checkpoint existence")
    p.add_argument("--dest", help="Local destination dir (download only)")
    p.add_argument("--src",  help="Local source dir (upload only)")
    args = p.parse_args()

    if args.action == "download":
        download(args)
    else:
        upload(args)
