# YOLOv8n Training Pipeline User Guide

## Quick Start

1. **Add videos** to `data/videos/` folder
2. **Run complete pipeline:** `./run_pipeline.sh full`
3. **Label images** in Label Studio (http://localhost:8080)
4. **Continue pipeline:** `./run_pipeline.sh continue`
5. **Test model:** `./run_pipeline.sh test --source path/to/test/images`

## Super Quick Commands
```bash
# Complete workflow
./run_pipeline.sh full --frame-skip 10 --randomize
# [Label images in Label Studio]
./run_pipeline.sh continue

# Test your model
./run_pipeline.sh test --source data/test_images --save-images

# Quick operations
python quick_commands.py status
python quick_commands.py extract --skip 15 --random
python quick_commands.py test data/new_images
```

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
**Enhanced Options:**
- `--frame-skip N`: Extract 1 frame out of every N frames (default: 1, extract all)
- `--randomize`: Randomize frame filenames after extraction for better training diversity

**Examples:**
```bash
python scripts/1_extract_frames.py --frame-skip 10 --randomize
python scripts/1_extract_frames.py --frame-skip 5
```

**What it does:** Converts videos to PNG frames at 2 FPS
- **Input:** `data/videos/*.mp4`
- **Output:** `data/frames/video_name/*.png`
- **Frame skipping:** Reduces dataset size by extracting fewer frames
- **Randomization:** Shuffles filenames to improve training diversity

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
**Enhanced Options:**
- Default: Clears existing tasks and imports new ones
- `--keep-existing`: Preserves existing tasks and adds new ones

**Examples:**
```bash
python scripts/3_import_to_labelstudio.py --keep-existing
python scripts/3_import_to_labelstudio.py  # Clears existing tasks
```

**What it does:** Creates signed URLs and imports images to Label Studio
- **Default behavior:** Clears existing tasks and creates new labeling tasks
- **With --keep-existing:** Preserves existing labeled work and adds new tasks
- Shows total task count after import

### 4. Label Images
1. Open `http://localhost:8080`
2. Click on images to label
3. Draw bounding boxes around objects
4. Submit each task

### 5. Export Annotations
```bash
python scripts/4_export_annotations.py
```

### 6. Sanitize Dataset (Optional but Recommended)
```bash
python scripts/sanitize.py 25
```
**What it does:** Removes empty label pairs to improve training
- **25%**: Keeps 25% of empty pairs, deletes 75%
- **0%**: Deletes ALL empty pairs
- **100%**: Keeps ALL empty pairs (no deletion)

### 7. Split Dataset
```bash
python scripts/5_split_dataset.py
```
**Enhanced Options:**
- `--train-ratio 0.8`: Training set ratio (default: 0.8)
- `--val-ratio 0.2`: Validation set ratio (default: 0.2)
- `--seed 42`: Random seed for reproducibility (default: 42)
- `--export-dir PATH`: Specific export directory to split (default: latest)

**Examples:**
```bash
python scripts/5_split_dataset.py --train-ratio 0.9 --val-ratio 0.1
python scripts/5_split_dataset.py --seed 123
```

**What it does:** Splits exported data into proper train/validation sets
- **Creates:** `train/` and `val/` subdirectories with `images/` and `labels/`
- **Maintains:** Image-label pair integrity
- **Analyzes:** Class distribution across splits
- **Generates:** `dataset.yaml` config file for training

### 7. Train Model
```bash
python scripts/6_train_model.py train
```
**What it does:** Trains YOLOv8n model on your split data
- Uses proper train/val split automatically
- Saves model to `models/best.pt`
- Auto-validates after training

## Model Operations

### Validate Model Performance
```bash
python scripts/6_train_model.py validate --model models/best.pt
```
Shows metrics: mAP@50, precision, recall

### Run Detection on New Images
```bash
python scripts/6_train_model.py detect --source data/frames --model models/best.pt
```
**Options:**
- `--source`: Directory with images to detect
- `--model`: Model file to use (default: models/best.pt)

**Output:** Annotated images in `runs/detect/`

## Advanced Usage Examples

### Optimal Training Workflow
```bash
# Extract limited frames with randomization
python scripts/1_extract_frames.py --frame-skip 15 --randomize
python scripts/2_upload_to_gcs.py
python scripts/3_import_to_labelstudio.py

# Label images in Label Studio, then:
python scripts/4_export_annotations.py
python scripts/5_split_dataset.py --train-ratio 0.85 --val-ratio 0.15
python scripts/6_train_model.py train
```

### Incremental Labeling Workflow
```bash
# First batch of images
python scripts/1_extract_frames.py --frame-skip 20
python scripts/2_upload_to_gcs.py
python scripts/3_import_to_labelstudio.py

# Label first batch, then add more images
python scripts/1_extract_frames.py --frame-skip 10  # Extract more frames
python scripts/2_upload_to_gcs.py
python scripts/3_import_to_labelstudio.py --keep-existing  # Keep previous work
```

## Directory Structure
```
pipeline/
├── configs/config.yaml          # Main configuration
├── scripts/
│   ├── 1_extract_frames.py      # Video → frames
│   ├── 2_upload_to_gcs.py       # Frames → cloud
│   ├── 3_import_to_labelstudio.py # Cloud → Label Studio
│   ├── 4_export_annotations.py  # Export labeled data
│   ├── 5_split_dataset.py       # Split into train/val
│   ├── 6_train_model.py         # Train/validate/detect
│   ├── 7_test_model.py          # Robust model testing (NEW)
│   └── pipeline.py              # Complete workflow automation (NEW)
├── quick_commands.py            # One-line operations (NEW)
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
**Training fails:** Verify images and labels have matching filenames, and dataset is properly split
**No train/val split:** Run `python scripts/5_split_dataset.py` after exporting annotations
**Too many frames:** Use `--frame-skip` to reduce dataset size
**Poor training diversity:** Use `--randomize` flag when extracting frames

## Tips

- **Dataset Size:** Label at least 50+ images for decent results
- **Frame Selection:** Use `--frame-skip 10-20` for large videos to get manageable datasets
- **Training Diversity:** Always use `--randomize` to improve model generalization
- **Proper Splitting:** Always split your dataset with `5_split_dataset.py` for better validation
- **Split Ratios:** Use 80/20 or 85/15 train/val split for most cases
- **Incremental Labeling:** Use `--keep-existing` when adding more data to existing projects
- **Consistency:** Use consistent labeling across all images
- **Class Names:** Keep class names simple and clear
- **Backup:** Back up your `models/best.pt` file