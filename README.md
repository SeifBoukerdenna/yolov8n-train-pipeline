# YOLOv8n Training Pipeline

An end-to-end workflow from raw gameplay videos to YOLOv8n training & validation.

---

## Prerequisites

- Install [ffmpeg](https://ffmpeg.org/download.html) and [Google Cloud SDK (gsutil)](https://cloud.google.com/sdk/docs/install)
- Python 3.8+
- A Google Cloud Storage bucket (e.g. `bucket-clash-royal-ai`)
- Label Studio configured with your bucket’s `raw_frames/` prefix
- make sure to run:
```bash
    git clone https://github.com/ultralytics/ultralytics.git
```


## 1. Drop raw gameplay videos

Place your `.mp4` files into:

```bash
yolov8n-train-pipeline/videos/
```

## 2. Extract frames

```bash
cd yolov8n-train-pipeline
chmod +x scripts/extract_frames.sh
./scripts/extract_frames.sh
```
- Input: videos/*.mp4
- Output: data/raw_frames/{video_name}/*.png (2 fps by default)

## 3. Upload frames for labeling
Make the script executable and run it:

```bash
chmod +x scripts/upload_frames.sh
./scripts/upload_frames.sh
```

Syncs: data/raw_frames/ → gs://bucket-clash-royal-ai/match_1/raw_frames/

## 4. Label on LS
Unzip into data/export/ so you have:
```bash
data/export/images/*.png
data/export/labels/*.txt
```

## 5. Split into train / val
Make the script executable and run it (default 10% validation split):
```bash
chmod +x scripts/split_dataset.sh
./scripts/split_dataset.sh data/export/images data/export/labels 0.1
```

- Args:

- 1. Source images dir (data/export/images)

- 2. Source labels dir (data/export/labels)

- 3. Validation ratio (0.1 = 10%)

- output:
    ```bash
    data/images/train/
    data/images/val/
    data/labels/train/
    data/labels/val/
    ```

## 6. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Train YOLOv8n
```bash
EPOCHS=50 IMG_SIZE=640 BATCH_SIZE=16 python scripts/train.py
```
- Output: models/exp1/weights/best.pt


## 8. Evaluate your model
```bash
WEIGHTS_PATH=models/exp1/weights/best.pt python scripts/evaluate.py
```
- Prints mAP, precision, recall, etc.







