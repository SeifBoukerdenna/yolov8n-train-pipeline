# scripts/2_upload_to_gcs.py

import yaml
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm
import argparse

def upload_to_gcs(local_dir, bucket_name, prefix, keep_folders=True):
    """Upload directory to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_dir = Path(local_dir)
    files = list(local_dir.rglob("*.png"))

    if not files:
        print("❌ No PNG files found")
        return

    print(f"Uploading {len(files)} files...")

    for file_path in tqdm(files, desc="Uploading"):
        if keep_folders:
            # Keep folder structure: frames/video_name/image.png
            relative_path = file_path.relative_to(local_dir)
            blob_name = f"{prefix}{relative_path}"
        else:
            # Flat structure: frames/image.png
            blob_name = f"{prefix}{file_path.name}"

        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path))

    print(f"✅ Uploaded to gs://{bucket_name}/{prefix}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flat', action='store_true',
                       help='Upload files flat (no folder structure)')
    args = parser.parse_args()

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    upload_to_gcs(
        local_dir="data/frames",
        bucket_name=config['gcs']['bucket'],
        prefix=config['gcs']['prefix'],
        keep_folders=not args.flat
    )

if __name__ == "__main__":
    main()