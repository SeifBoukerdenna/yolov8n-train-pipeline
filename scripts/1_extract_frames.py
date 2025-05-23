# scripts/1_extract_frames.py

import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=2):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_video / fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    saved_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = output_dir / f"{video_path.stem}_{saved_count:05d}.png"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return saved_count

def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    video_dir = Path("data/videos")
    frames_dir = Path("data/frames")

    if not video_dir.exists():
        print("❌ data/videos/ directory not found")
        return

    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("❌ No .mp4 files found in data/videos/")
        return

    print(f"Found {len(videos)} videos")

    for video_path in videos:
        output_dir = frames_dir / video_path.stem
        count = extract_frames(video_path, output_dir, config['extraction']['fps'])
        print(f"✓ {video_path.name}: {count} frames extracted")

    print(f"\n✅ All frames extracted to data/frames/")

if __name__ == "__main__":
    main()