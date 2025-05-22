#!/usr/bin/env bash
set -e

mkdir -p data/raw_frames
for vid in videos/*.mp4; do
  name=$(basename "$vid" .mp4)
  outdir="data/raw_frames/$name"
  mkdir -p "$outdir"
  # extract at 2 fps (adjust fps as you like)
  ffmpeg -i "$vid" -vf fps=2 "$outdir/${name}_%05d.png"
done
