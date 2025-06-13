# Clash Royale AI - Deployment Detection Pipeline

An end-to-end machine learning pipeline for training YOLOv8 models to detect card deployments in Clash Royale gameplay videos. This system automates the entire process from raw video files to a trained object detection model.

## 🎯 What This Does

This repository provides a complete pipeline for:

1. **Processing Clash Royale gameplay videos** into individual frames
2. **Managing large-scale image datasets** via Google Cloud Storage
3. **Streamlining annotation workflows** with Label Studio integration
4. **Training custom YOLOv8 models** to detect card deployments
5. **Testing and validating** model performance on new footage

The primary use case is detecting when and where players deploy cards/troops in Clash Royale matches, which can be valuable for:
- Game analysis and strategy research
- Automated replay analysis
- Player behavior studies
- AI game assistant development

## 🏗️ Architecture Overview

```
Raw Videos → Frame Extraction → Cloud Storage → Label Studio → Annotations → Model Training → Deployment Detection
```

### Pipeline Components

1. **Video Processing** (`1_extract_frames.py`)
   - Extracts frames from MP4 videos at configurable intervals
   - Supports frame skipping to manage dataset size
   - Optional filename randomization for training diversity

2. **Cloud Storage** (`2_upload_to_gcs.py`)
   - Uploads extracted frames to Google Cloud Storage
   - Manages large datasets efficiently
   - Maintains organized folder structures

3. **Annotation Platform** (`3_import_to_labelstudio.py`)
   - Integrates with Label Studio for manual annotation
   - Creates labeling tasks automatically
   - Manages incremental dataset additions

4. **Data Export** (`4_export_annotations.py`)
   - Exports labeled data in YOLO format
   - Downloads images and corresponding annotations
   - Organizes data for training pipeline

5. **Dataset Management** (`5_split_dataset.py`)
   - Splits data into training and validation sets
   - Analyzes class distribution
   - Creates proper YOLO dataset configuration

6. **Model Training** (`6_train_model.py`)
   - Trains YOLOv8n models on deployment detection
   - Supports training, validation, and inference modes
   - Optimized for deployment detection tasks

7. **Testing & Validation** (`7_test_model.py`)
   - Comprehensive model testing on new footage
   - Generates detailed performance reports
   - Creates visualizations and annotated outputs

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud SDK with authentication
- Label Studio running on localhost:8080
- Required packages: `pip install ultralytics google-cloud-storage label-studio-sdk opencv-python pyyaml requests`

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Google Cloud authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Start Label Studio
label-studio start
```

### Configuration

Edit `configs/config.yaml`:
- Set your GCS bucket name
- Update Label Studio API key and project ID
- Adjust training parameters (epochs, batch size, etc.)

```yaml
gcs:
  bucket: your-gcs-bucket-name
  prefix: frames/

labelstudio:
  url: http://localhost:8080
  api_key: your-api-key
  project_id: your-project-id

training:
  epochs: 30
  batch_size: 4
  device: mps  # Use 'mps' for M1/M2 Macs, 'cpu' as fallback

classes:
  - deployment  # Currently detects card deployments
```

### Basic Workflow

1. **Add your Clash Royale videos** to `data/videos/` (MP4 format)

2. **Run the complete pipeline:**
   ```bash
   ./run_pipeline.sh full --frame-skip 10 --randomize
   ```

3. **Label deployments** in Label Studio (http://localhost:8080)
   - Draw bounding boxes around card deployment locations
   - Mark the exact moment/location where troops are placed

4. **Continue training pipeline:**
   ```bash
   ./run_pipeline.sh continue
   ```

5. **Test your model:**
   ```bash
   ./run_pipeline.sh test --source path/to/test/images --save-images
   ```

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

## 📋 Step-by-Step Workflow

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

**What it does:** Converts videos to PNG frames at configured FPS
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

**What it does:** Uploads frames to GCS bucket for Label Studio access

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
3. Draw bounding boxes around deployment locations
4. Submit each task

### 5. Export Annotations
```bash
python scripts/4_export_annotations.py
```

**What it does:** Exports labeled data in YOLO format with progress tracking

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

### 8. Train Model
```bash
python scripts/6_train_model.py train
```
**What it does:** Trains YOLOv8n model on your split data
- Uses proper train/val split automatically
- Saves model to `models/best.pt`
- Auto-validates after training

## 🔧 Model Operations

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

### Comprehensive Model Testing
```bash
python scripts/7_test_model.py --source path/to/test/images --save-images
```
**Options:**
- `--source`: Path to test images directory or single image
- `--model`: Path to model file (default: models/best.pt)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--save-images`: Save annotated images with detections
- `--max-images`: Maximum images to save (default: 300)

**What it generates:**
- Detailed performance reports
- Confidence distribution plots
- Class distribution analysis
- Annotated output images

## 📁 Repository Structure

```
clash-royale-ai/
├── configs/
│   └── config.yaml             # Main configuration
├── scripts/
│   ├── 1_extract_frames.py     # Video → frames conversion
│   ├── 2_upload_to_gcs.py      # Cloud storage management
│   ├── 3_import_to_labelstudio.py # Annotation workflow
│   ├── 4_export_annotations.py # Data export
│   ├── 5_split_dataset.py      # Dataset preparation
│   ├── 6_train_model.py        # Model training/validation/detection
│   ├── 7_test_model.py         # Comprehensive model testing
│   ├── pipeline.py             # Full workflow automation
│   ├── sanitize.py             # Dataset cleanup utility
│   └── nuke.py                 # Clean slate utility
├── data/
│   ├── videos/                 # Input Clash Royale videos
│   ├── frames/                 # Extracted frames
│   └── annotations/            # Exported labeled datasets
├── models/                     # Trained models
├── runs/                       # Training/detection outputs
├── test_results/               # Model testing reports
├── run_pipeline.sh             # Convenient pipeline runner
├── quick_commands.py           # One-line operations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎮 Clash Royale Specifics

### What Gets Detected

The model is trained to identify **deployment events** - the specific moments and locations where players place cards on the battlefield. This includes:

- Troop deployments (any card placement)
- Building placements
- Spell target locations
- Defensive placements

### Video Requirements

- **Format:** MP4 files
- **Content:** Clash Royale gameplay footage
- **Quality:** Clear view of the game board
- **Perspective:** Standard gameplay view (not replays with camera movement)

### Labeling Guidelines

When annotating in Label Studio:
1. Draw tight bounding boxes around the deployment location
2. Label the exact frame where the deployment visual effect appears
3. Focus on the deployment indicator, not the deployed unit
4. Be consistent with box sizes and positioning

## 🛠️ Advanced Usage

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

### Optimizing for Large Datasets

```bash
# Extract fewer frames for faster processing
python scripts/1_extract_frames.py --frame-skip 20 --randomize

# Sanitize dataset to remove empty annotations
python scripts/sanitize.py 25  # Keep 25% of empty labels
```

### Model Evaluation

```bash
# Comprehensive testing with reports
python scripts/7_test_model.py --source data/test_videos --save-images

# Quick validation
python scripts/6_train_model.py validate --model models/best.pt
```

## 🚨 Troubleshooting

**Authentication errors:** Set `GOOGLE_APPLICATION_CREDENTIALS` to service account JSON
**Label Studio connection:** Check API key and project ID in config
**No images exported:** Ensure tasks are completed (not skipped) in Label Studio
**Training fails:** Verify images and labels have matching filenames, and dataset is properly split
**No train/val split:** Run `python scripts/5_split_dataset.py` after exporting annotations
**Too many frames:** Use `--frame-skip` to reduce dataset size
**Poor training diversity:** Use `--randomize` flag when extracting frames

## 💡 Tips & Best Practices

- **Dataset Size:** Label at least 50+ images for decent results
- **Frame Selection:** Use `--frame-skip 10-20` for large videos to get manageable datasets
- **Training Diversity:** Always use `--randomize` to improve model generalization
- **Proper Splitting:** Always split your dataset with `5_split_dataset.py` for better validation
- **Split Ratios:** Use 80/20 or 85/15 train/val split for most cases
- **Incremental Labeling:** Use `--keep-existing` when adding more data to existing projects
- **Consistency:** Use consistent labeling across all images
- **Class Names:** Keep class names simple and clear
- **Backup:** Back up your `models/best.pt` file

## 📊 Performance Monitoring

The pipeline includes built-in analytics:

- **Dataset statistics** (frame counts, annotation distribution)
- **Training metrics** (mAP, precision, recall)
- **Test reports** with confidence distributions
- **Visualizations** for model performance analysis

## 🔧 Customization

### Adding New Classes

To detect other game elements beyond deployments:

1. Update `configs/config.yaml`:
   ```yaml
   classes:
     - deployment
     - tower_damage
     - spell_effect
   ```

2. Retrain with new annotations:
   ```bash
   python scripts/5_split_dataset.py
   python scripts/6_train_model.py train
   ```

### Hardware Optimization

The pipeline supports various hardware configurations:

- **Apple Silicon:** `device: mps` (default for M1/M2 Macs)
- **NVIDIA GPU:** `device: 0` (or specific GPU index)
- **CPU fallback:** `device: cpu`

## 🤝 Contributing

When contributing new features:

1. Follow the existing script numbering convention
2. Update `run_pipeline.sh` with new commands
3. Add configuration options to `config.yaml`
4. Include comprehensive error handling and progress indicators

## 📜 License

This project is designed for research and educational purposes in game AI development.

## 🎯 Use Cases

- **Esports Analysis:** Automated deployment pattern analysis
- **Player Training:** Identify optimal deployment timing and positioning
- **Game AI:** Training bots to recognize deployment opportunities
- **Research:** Large-scale analysis of Clash Royale gameplay strategies
