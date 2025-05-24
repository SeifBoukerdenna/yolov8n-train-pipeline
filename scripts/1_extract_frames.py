# scripts/1_extract_frames.py

import cv2
import yaml
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=2, frame_skip=1, video_index=1):
    """Extract frames from video with optional frame skipping"""
    cap = cv2.VideoCapture(str(video_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_video / fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    saved_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing CR_{video_index}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Apply frame skipping - only save every nth frame that would be extracted
            if saved_count % frame_skip == 0:
                output_path = output_dir / f"CR_{video_index}_{saved_count:05d}.png"
                cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Return actual number of files saved (accounting for frame_skip)
    actual_saved = len(list(output_dir.glob("*.png")))
    return actual_saved

def randomize_filenames(directory):
    """Randomize filenames in directory while keeping extensions"""
    directory = Path(directory)
    files = list(directory.glob("*.png"))

    if not files:
        return 0

    print(f"🎲 Randomizing {len(files)} filenames...")

    # Create list of random numbers
    random_indices = list(range(len(files)))
    random.shuffle(random_indices)

    # Create temporary names to avoid conflicts
    temp_names = []
    for i, file_path in enumerate(files):
        temp_name = directory / f"temp_{random_indices[i]:05d}.png"
        file_path.rename(temp_name)
        temp_names.append(temp_name)

    # Rename to final randomized names
    for i, temp_file in enumerate(temp_names):
        final_name = directory / f"{directory.name}_random_{i:05d}.png"
        temp_file.rename(final_name)

    return len(files)

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Extract 1 frame out of every N frames (default: 1, extract all)')
    parser.add_argument('--randomize', action='store_true',
                       help='Randomize frame filenames after extraction')
    args = parser.parse_args()

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    video_dir = Path("data/videos")
    frames_dir = Path("data/frames")

    if not video_dir.exists():
        print("❌ data/videos/ directory not found")
        return

    # Support multiple video formats
    video_extensions = ["*.mp4", "*.mov", "*.MP4", "*.MOV"]
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir.glob(ext))

    if not videos:
        print("❌ No video files found in data/videos/ (supported: .mp4, .mov, .MP4, .MOV)")
        return

    print(f"Found {len(videos)} videos")
    if args.frame_skip > 1:
        print(f"📊 Frame skip: extracting 1 out of every {args.frame_skip} frames")

    total_extracted = 0
    for i, video_path in enumerate(videos, 1):
        output_dir = frames_dir / f"CR_{i}"
        count = extract_frames(
            video_path,
            output_dir,
            config['extraction']['fps'],
            args.frame_skip,
            i
        )
        total_extracted += count
        print(f"✓ CR_{i}: {count} frames extracted")

        # Randomize filenames if requested
        if args.randomize:
            randomized_count = randomize_filenames(output_dir)
            print(f"  🎲 Randomized {randomized_count} filenames")

    print(f"\n✅ Total frames extracted: {total_extracted}")
    if args.randomize:
        print("🎲 All filenames randomized for better training diversity")

if __name__ == "__main__":
    main()