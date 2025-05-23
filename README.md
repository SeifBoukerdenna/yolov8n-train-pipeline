# YOLOv8n Training Pipeline User Guide

## Quick Start

1. **Add videos** to `data/videos/` folder
2. **Run scripts 1-5** in order
3. **Label images** in Label Studio between steps 3-4
4. **Train model** with step 5

## Prerequisites

- Python 3.8+
- Google Cloud SDK with authentication
- Label Studio running on localhost:8080
- Required packages: `pip install ultralytics google-cloud-storage label-studio-sdk opencv-python pyyaml requests`

## Configuration

Edit `configs/config.yaml`:
- Set your GCS bucket name
- Update Label Studio API key and project ID
- Adjust training parameters (epochs, batch size, etc.)

## Step-by-Step Workflow

### 1. Extract Frames
```bash
python scripts/1_extract_frames.py
```
**What it does:** Converts videos to PNG frames at 2 FPS
- **Input:** `data/videos/*.mp4`
- **Output:** `data/frames/video_name/*.png`

### 2. Upload to Google Cloud
```bash
python scripts/2_upload_to_gcs.py
```
**Options:**
- Default: Keeps folder structure
- `--flat`: All images in one folder

**What it does:** Uploads frames to GCS bucket

### 3. Import to Label Studio
```bash
python scripts/3_import_to_labelstudio.py
```
**What it does:** Creates signed URLs and imports images to Label Studio
- Clears existing tasks
- Creates new labeling tasks

### 4. Label Images
1. Open `http://localhost:8080`
2. Click on images to label
3. Draw bounding boxes around objects
4. Submit each task

### 5. Export Annotations
```bash
python scripts/4_export_annotations.py
```
**What it does:** Downloads labeled data in YOLO format
- **Output:** `data/annotations/export_TIMESTAMP/`
- Downloads both images and label files
- Matches filenames for training

### 6. Train Model
```bash
python scripts/5_train_model.py train
```
**What it does:** Trains YOLOv8n model on your data
- Uses latest export automatically
- Saves model to `models/best.pt`
- Auto-validates after training

## Model Operations

### Validate Model Performance
```bash
python scripts/5_train_model.py validate --model models/best.pt
```
Shows metrics: mAP@50, precision, recall

### Run Detection on New Images
```bash
python scripts/5_train_model.py detect --source data/frames --model models/best.pt
```
**Options:**
- `--source`: Directory with images to detect
- `--model`: Model file to use (default: models/best.pt)

**Output:** Annotated images in `runs/detect/`

## Directory Structure
```
pipeline/
├── configs/config.yaml          # Main configuration
├── scripts/
│   ├── 1_extract_frames.py      # Video → frames
│   ├── 2_upload_to_gcs.py       # Frames → cloud
│   ├── 3_import_to_labelstudio.py # Cloud → Label Studio
│   ├── 4_export_annotations.py  # Export labeled data
│   └── 5_train_model.py         # Train/validate/detect
├── data/
│   ├── videos/                  # Input videos
│   ├── frames/                  # Extracted frames
│   └── annotations/             # Exported labels
└── models/                      # Trained models
```

## Troubleshooting

**Authentication errors:** Set `GOOGLE_APPLICATION_CREDENTIALS` to service account JSON
**Label Studio connection:** Check API key and project ID in config
**No images exported:** Ensure tasks are completed (not skipped) in Label Studio
**Training fails:** Verify images and labels have matching filenames

## Tips

- Label at least 50+ images for decent results
- Use consistent labeling across all images
- Keep class names simple and clear
- Back up your `models/best.pt` file