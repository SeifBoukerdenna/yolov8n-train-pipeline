#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo root is: $ROOT"
echo "Looking for video files in: $ROOT/videos"
mkdir -p "$ROOT/data/raw_frames"

shopt -s nullglob
for vid in "$ROOT"/data/videos/*.[mM][pP]4; do
  name="$(basename "$vid" .mp4)"
  name="${name%.MP4}"
  outdir="$ROOT/data/raw_frames/$name"
  mkdir -p "$outdir"
  ffmpeg -i "$vid" -vf fps=2 "$outdir/${name}_%05d.png"
done

echo "✔ extracted frames into data/raw_frames/"
