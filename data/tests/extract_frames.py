import os
import cv2
import argparse

def extract_frames(video_path: str, output_dir: str, interval: int = 15):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done: extracted {saved} frames to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract one frame every N frames from a video."
    )
    parser.add_argument("video", help="Path to input .mp4 file")
    parser.add_argument(
        "-o", "--output", default="tests",
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "-n", "--interval", type=int, default=15,
        help="Frame interval (every Nth frame is saved)"
    )
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.interval)
