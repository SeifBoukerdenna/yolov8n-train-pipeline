#!/usr/bin/env bash
set -euo pipefail

# 1) define paths
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_FRAMES="$ROOT/data/raw_frames"
EXPORT_IMG="$ROOT/data/export/images"
LABELS="$ROOT/data/export/labels"

# 2) make sure destination exists
mkdir -p "$EXPORT_IMG"

# 3) loop through each annotation
for lbl in "$LABELS"/*.txt; do
  base="$(basename "$lbl" .txt)"
  # try png then jpg
  img=$(find "$RAW_FRAMES" -type f \( -iname "${base}.png" -o -iname "${base}.jpg" \) -print -quit || true)
  if [[ -n "$img" ]]; then
    echo "Moving $img → $EXPORT_IMG/"
    mv "$img" "$EXPORT_IMG/"
  else
    echo "⚠️  WARNING: no image found for $base"
  fi
done

echo "✔ Done moving export images."
