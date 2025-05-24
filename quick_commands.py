# quick_commands.py

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """Execute command and show output"""
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    if len(sys.argv) < 2:
        print("""
Quick Commands for YOLO Pipeline:

python quick_commands.py nuke                             # Delete ALL data (DANGER!)
python quick_commands.py sanitize [percentage]             # Clean empty labels
python quick_commands.py extract [--skip N] [--random]    # Extract frames
python quick_commands.py upload                           # Upload to GCS
python quick_commands.py import [--keep]                  # Import to Label Studio
python quick_commands.py export                           # Export annotations
python quick_commands.py split                            # Split dataset
python quick_commands.py train                            # Train model
python quick_commands.py test <image_path>                # Test model
python quick_commands.py pipeline                         # Full pipeline
python quick_commands.py status                           # Check status

Examples:
  python quick_commands.py sanitize 25
  python quick_commands.py extract --skip 10 --random
  python quick_commands.py test data/new_images
        """)
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "extract":
        base_cmd = "python scripts/1_extract_frames.py"
        if "--skip" in args:
            idx = args.index("--skip")
            base_cmd += f" --frame-skip {args[idx+1]}"
        if "--random" in args:
            base_cmd += " --randomize"
        run_cmd(base_cmd)

    elif cmd == "upload":
        run_cmd("python scripts/2_upload_to_gcs.py")

    elif cmd == "import":
        base_cmd = "python scripts/3_import_to_labelstudio.py"
        if "--keep" in args:
            base_cmd += " --keep-existing"
        run_cmd(base_cmd)

    elif cmd == "nuke":
        run_cmd("python scripts/nuke.py")

    elif cmd == "sanitize":
        base_cmd = "python scripts/sanitize.py"
        if args:
            base_cmd += f" {args[0]}"
        run_cmd(base_cmd)

    elif cmd == "export":
        run_cmd("python scripts/4_export_annotations.py")

    elif cmd == "split":
        run_cmd("python scripts/5_split_dataset.py")

    elif cmd == "train":
        run_cmd("python scripts/6_train_model.py train")

    elif cmd == "test":
        if args:
            run_cmd(f"python scripts/7_test_model.py --source {args[0]} --save-images")
        else:
            print("❌ Provide image path: python quick_commands.py test <path>")

    elif cmd == "pipeline":
        run_cmd("python scripts/pipeline.py")

    elif cmd == "status":
        print("📊 Pipeline Status:")

        # Check videos
        videos = list(Path("data/videos").glob("*.mp4"))
        print(f"Videos: {len(videos)} found")

        # Check frames
        frames = list(Path("data/frames").rglob("*.png"))
        print(f"Frames: {len(frames)} extracted")

        # Check exports
        exports = list(Path("data/annotations").glob("export_*"))
        print(f"Exports: {len(exports)} available")

        # Check models
        models = list(Path("models").glob("*.pt"))
        print(f"Models: {len(models)} trained")

        # Check latest export structure
        if exports:
            latest = sorted(exports)[-1]
            has_split = (latest / "train").exists() and (latest / "val").exists()
            print(f"Latest export split: {'✅' if has_split else '❌'}")

    else:
        print(f"❌ Unknown command: {cmd}")

if __name__ == "__main__":
    main()