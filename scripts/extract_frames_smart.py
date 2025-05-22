import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
from tqdm import tqdm
import yaml

def extract_smart_frames(video_path, output_dir, fps=2, similarity_threshold=0.95):
    """Extract frames with duplicate detection"""
    cap = cv2.VideoCapture(str(video_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_video / fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    saved_count = 0
    last_frame = None
    frame_hashes = set()

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Check similarity with last frame
            if last_frame is not None:
                similarity = calculate_similarity(frame, last_frame)
                if similarity > similarity_threshold:
                    pbar.update(1)
                    frame_count += 1
                    continue

            # Check hash for exact duplicates
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            if frame_hash in frame_hashes:
                pbar.update(1)
                frame_count += 1
                continue

            # Save frame
            output_path = output_dir / f"{video_path.stem}_{saved_count:05d}.png"
            cv2.imwrite(str(output_path), frame)

            frame_hashes.add(frame_hash)
            last_frame = frame.copy()
            saved_count += 1

        pbar.update(1)
        frame_count += 1

    cap.release()
    pbar.close()
    return saved_count

def calculate_similarity(img1, img2):
    """Calculate structural similarity between frames"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]

def process_videos_parallel(video_dir, output_dir, config):
    """Process multiple videos in parallel"""
    video_paths = list(video_dir.glob("*.mp4"))

    with ProcessPoolExecutor(max_workers=config['parallel_workers']) as executor:
        futures = []
        for video_path in video_paths:
            out_dir = output_dir / video_path.stem
            future = executor.submit(
                extract_smart_frames,
                video_path,
                out_dir,
                config['fps'],
                config['similarity_threshold']
            )
            futures.append((video_path, future))

        for video_path, future in futures:
            count = future.result()
            print(f"✓ {video_path.name}: {count} unique frames extracted")

if __name__ == "__main__":
    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    process_videos_parallel(
        Path("data/videos"),
        Path("data/processed"),
        config['extraction']
    )