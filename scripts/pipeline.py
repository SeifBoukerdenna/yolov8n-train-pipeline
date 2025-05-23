# scripts/pipeline.py

import yaml
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False

    print(f"✅ Completed: {description}")
    if result.stdout:
        print(result.stdout)
    return True

def check_prerequisites():
    """Check if all required directories and files exist"""
    required_paths = [
        "configs/config.yaml",
        "data/videos"
    ]

    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print("❌ Missing required files/directories:")
        for item in missing:
            print(f"  - {item}")
        return False

    # Check for videos
    videos = list(Path("data/videos").glob("*.mp4"))
    if not videos:
        print("❌ No .mp4 files found in data/videos/")
        return False

    print(f"✅ Found {len(videos)} videos to process")
    return True

def run_full_pipeline(args):
    """Run the complete pipeline"""

    if not check_prerequisites():
        return False

    # Step 1: Extract frames
    extract_cmd = ["python", "scripts/1_extract_frames.py"]
    if args.frame_skip > 1:
        extract_cmd.extend(["--frame-skip", str(args.frame_skip)])
    if args.randomize:
        extract_cmd.append("--randomize")

    if not run_command(extract_cmd, "Extracting frames from videos"):
        return False

    # Step 2: Upload to GCS
    upload_cmd = ["python", "scripts/2_upload_to_gcs.py"]
    if args.flat:
        upload_cmd.append("--flat")

    if not run_command(upload_cmd, "Uploading frames to Google Cloud Storage"):
        return False

    # Step 3: Import to Label Studio
    import_cmd = ["python", "scripts/3_import_to_labelstudio.py"]
    if args.keep_existing:
        import_cmd.append("--keep-existing")

    if not run_command(import_cmd, "Importing images to Label Studio"):
        return False

    print(f"\n⏸️  MANUAL STEP REQUIRED:")
    print(f"🎨 Go to Label Studio and label your images:")
    print(f"   http://localhost:8080")
    print(f"\n📝 Once labeling is complete, run:")
    print(f"   python scripts/pipeline.py --continue")

    return True

def continue_pipeline(args):
    """Continue pipeline after labeling"""

    # Step 4: Export annotations
    if not run_command(["python", "scripts/4_export_annotations.py"],
                      "Exporting annotations from Label Studio"):
        return False

    # Step 5: Split dataset
    split_cmd = ["python", "scripts/5_split_dataset.py"]
    if args.train_ratio:
        split_cmd.extend(["--train-ratio", str(args.train_ratio)])
    if args.val_ratio:
        split_cmd.extend(["--val-ratio", str(args.val_ratio)])

    if not run_command(split_cmd, "Splitting dataset into train/validation"):
        return False

    # Step 6: Train model
    if not run_command(["python", "scripts/6_train_model.py", "train"],
                      "Training YOLO model"):
        return False

    print(f"\n🎉 Pipeline complete!")
    print(f"📁 Model saved to: models/best.pt")
    print(f"\n🧪 Test your model:")
    print(f"   python scripts/7_test_model.py --source path/to/test/images")

    return True

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Training Pipeline')
    parser.add_argument('--continue', action='store_true', dest='continue_pipeline',
                       help='Continue pipeline after labeling (skip to export)')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Extract 1 frame out of every N frames')
    parser.add_argument('--randomize', action='store_true',
                       help='Randomize frame filenames')
    parser.add_argument('--flat', action='store_true',
                       help='Upload files flat (no folder structure)')
    parser.add_argument('--keep-existing', action='store_true',
                       help='Keep existing Label Studio tasks')
    parser.add_argument('--train-ratio', type=float,
                       help='Training set ratio (default from config)')
    parser.add_argument('--val-ratio', type=float,
                       help='Validation set ratio (default from config)')

    args = parser.parse_args()

    if args.continue_pipeline:
        success = continue_pipeline(args)
    else:
        success = run_full_pipeline(args)

    if success:
        print(f"\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()