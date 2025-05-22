#!/usr/bin/env bash
set -e

# Clear old
rm -rf data/images/{train,val} data/labels/{train,val}
mkdir -p data/images/{train,val} data/labels/{train,val}

# Adjust this ratio as needed
VAL_PCT=0.1

for img in data/raw_frames/*/*.png; do
  if (( RANDOM % 100 < VAL_PCT * 100 )); then
    target="val"
  else
    target="train"
  fi
  cp "$img" data/images/$target/
done
