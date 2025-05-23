from google.cloud import storage
import yaml
from pathlib import Path
from tqdm import tqdm
import hashlib
import json

class GCSSync:
    def __init__(self, config):
        self.bucket_name = config['gcs']['bucket']
        self.prefix = config['gcs']['prefix']
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        self.sync_state_file = Path(".gcs_sync_state.json")
        self.sync_state = self.load_sync_state()

    def load_sync_state(self):
        if self.sync_state_file.exists():
            return json.loads(self.sync_state_file.read_text())
        return {}

    def save_sync_state(self):
        self.sync_state_file.write_text(json.dumps(self.sync_state, indent=2))

    def get_file_hash(self, file_path):
        """Get MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def sync_directory(self, local_dir):
        """Sync directory with intelligent change detection"""
        local_dir = Path(local_dir)
        files_to_upload = []

        # Check which files need uploading
        for file_path in local_dir.rglob("*.png"):
            relative_path = file_path.relative_to(local_dir)
            file_hash = self.get_file_hash(file_path)

            if str(relative_path) not in self.sync_state or \
               self.sync_state[str(relative_path)] != file_hash:
                files_to_upload.append((file_path, relative_path, file_hash))

        if not files_to_upload:
            print("✓ All files already synced")
            return

        # Upload new/changed files
        print(f"Uploading {len(files_to_upload)} files...")
        for file_path, relative_path, file_hash in tqdm(files_to_upload):
            blob_name = f"{self.prefix}{relative_path}"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            self.sync_state[str(relative_path)] = file_hash

        self.save_sync_state()
        print(f"✓ Synced {len(files_to_upload)} files to GCS")

if __name__ == "__main__":
    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    syncer = GCSSync(config)
    syncer.sync_directory("data/raw_frames")