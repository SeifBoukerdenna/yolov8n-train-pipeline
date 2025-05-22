import argparse
import yaml
import time
from pathlib import Path
import subprocess
import sys
from datetime import datetime

class Pipeline:
    def __init__(self, config_path="configs/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.log_file = Path(f"logs/pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.log_file.parent.mkdir(exist_ok=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")

    def run_stage(self, stage_name, script_path, *args):
        """Run a pipeline stage"""
        self.log(f"Starting stage: {stage_name}")
        try:
            result = subprocess.run(
                [sys.executable, script_path] + list(args),
                capture_output=True,
                text=True,
                check=True
            )
            self.log(f"✓ {stage_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ {stage_name} failed: {e.stderr}")
            return False

    def extract_frames(self):
        return self.run_stage(
            "Frame Extraction",
            "scripts/extract_frames_smart.py"
        )

    def sync_to_gcs(self):
        return self.run_stage(
            "GCS Sync",
            "scripts/sync_gcs.py"
        )

    def sync_labelstudio(self):
        return self.run_stage(
            "Label Studio Sync",
            "scripts/labelstudio_sync.py"
        )

    def validate_annotations(self):
        return self.run_stage(
            "Annotation Validation",
            "scripts/validate_annotations.py"
        )

    def train_model(self):
        return self.run_stage(
            "Model Training",
            "scripts/incremental_train.py"
        )

    def run_full_pipeline(self):
        """Run complete pipeline"""
        stages = [
            self.extract_frames,
            self.sync_to_gcs,
            self.sync_labelstudio,
            self.validate_annotations,
            self.train_model
        ]

        for stage in stages:
            if not stage():
                self.log("Pipeline stopped due to error")
                return False

        self.log("✓ Full pipeline completed successfully!")
        return True

    def watch_mode(self):
        """Continuous monitoring mode"""
        self.log("Starting watch mode...")

        while True:
            # Check for new videos
            video_dir = Path("data/videos")
            video_count = len(list(video_dir.glob("*.mp4")))

            if video_count > 0:
                self.log(f"Found {video_count} new videos")
                self.run_full_pipeline()

                # Archive processed videos
                archive_dir = video_dir / "processed"
                archive_dir.mkdir(exist_ok=True)
                for video in video_dir.glob("*.mp4"):
                    video.rename(archive_dir / video.name)

            # Check for new annotations periodically
            time.sleep(300)  # Check every 5 minutes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['full', 'extract', 'sync', 'train', 'continuous'],
        default='full',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Enable watch mode for continuous processing'
    )

    args = parser.parse_args()
    pipeline = Pipeline()

    if args.mode == 'continuous' or args.watch:
        pipeline.watch_mode()
    elif args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.mode == 'extract':
        pipeline.extract_frames()
    elif args.mode == 'sync':
        pipeline.sync_to_gcs()
        pipeline.sync_labelstudio()
    elif args.mode == 'train':
        pipeline.train_model()

if __name__ == "__main__":
    main()