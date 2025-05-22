#!/usr/bin/env bash
# Drop videos here and run this script

if [ -z "$(ls -A data/videos/*.mp4 2>/dev/null)" ]; then
    echo "❌ No videos found in data/videos/"
    echo "📹 Please add .mp4 files to data/videos/ first"
    exit 1
fi

echo "🎬 Processing videos..."
python scripts/pipeline.py --mode full
