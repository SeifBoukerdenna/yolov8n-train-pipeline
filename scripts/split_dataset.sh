#!/usr/bin/env bash
set -euo pipefail

SRC_IMG="${1:-data/export/images}"
SRC_LBL="${2:-data/export/labels}"
VAL_RATIO="${3:-0.1}"  # can be "0.1" or "10" or "25"

# 1) Convert VAL_RATIO into an integer percent 0–100
if [[ "$VAL_RATIO" == *.* ]]; then
  # e.g. "0.1" → "1", "0.25" → "25"
  VAL_PCT=${VAL_RATIO#*.}
else
  # already integer
  VAL_PCT="$VAL_RATIO"
fi

echo "Using validation split: $VAL_PCT%"

# 2) Prepare output folders
rm -rf data/images/{train,val} data/labels/{train,val}
mkdir -p data/images/{train,val} data/labels/{train,val}

# 3) Loop images (must match .txt names)
for img in "$SRC_IMG"/*.{png,jpg,jpeg}; do
  [[ -e "$img" ]] || continue
  name="$(basename "$img")"
  base="${name%.*}"
  rnd=$(( RANDOM % 100 ))
  if (( rnd < VAL_PCT )); then
    split="val"
  else
    split="train"
  fi
  # copy image + its label
  cp "$img" "data/images/$split/$name"
  cp "$SRC_LBL/$base.txt" "data/labels/$split/$base.txt"
done

echo "✔ Dataset split complete:"
echo "  Train images: $(ls data/images/train | wc -l)"
echo "  Val   images: $(ls data/images/val   | wc -l)"
