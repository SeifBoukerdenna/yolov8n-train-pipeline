#!/usr/bin/env bash
# setup_enhanced_pipeline.sh - Quick setup for enhanced pipeline

set -euo pipefail

echo "🚀 Setting up Enhanced YOLOv8n Pipeline..."

# 1. Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {data/{videos,raw_frames,processed,annotations,datasets},models/versions,logs,configs,scripts}

# 2. Download the enhanced scripts
echo "📥 Creating enhanced scripts..."

# Create pipeline.yaml template
cat > configs/pipeline.yaml << 'EOF'
# Google Cloud Storage
gcs:
  bucket: bucket-clash-royal-ai
  prefix: match_1/raw_frames/

# Label Studio
labelstudio:
  url: http://localhost:8080
  api_key: YOUR_API_KEY_HERE    # Get from Label Studio UI
  project_id: YOUR_PROJECT_ID    # Get from Label Studio project URL

# Frame Extraction
extraction:
  fps: 2
  quality: 95
  similarity_threshold: 0.95     # Skip similar frames (0.9-0.99)
  parallel_workers: 4

# Training
training:
  initial_epochs: 50
  incremental_epochs: 20
  batch_size: 16
  img_size: 640
  auto_increment_epochs: true
  early_stopping_patience: 10
  model_versioning: true

# Monitoring (optional)
monitoring:
  webhook_url: null              # Slack/Discord webhook
  email_alerts: null
EOF

# 3. Create requirements
cat > requirements-enhanced.txt << 'EOF'
# Base requirements
ultralytics>=8.0.20
google-cloud-storage>=2.10.0
label-studio-sdk>=0.0.32
pandas>=2.0.0
tqdm>=4.65.0

# Enhanced pipeline
opencv-python>=4.8.0
scikit-image>=0.21.0
pyyaml>=6.0
numpy>=1.24.0
Pillow>=10.0.0
requests>=2.31.0
click>=8.1.0
watchdog>=3.0.0

# Monitoring & logging
tensorboard>=2.13.0
wandb>=0.15.0
colorlog>=6.7.0
matplotlib>=3.7.0

# Testing & validation
pytest>=7.4.0
pytest-cov>=4.1.0
EOF

# 4. Install requirements
echo "📦 Installing requirements..."
pip install -r requirements-enhanced.txt

# 5. Set up Google Cloud auth
echo "🔐 Setting up Google Cloud authentication..."
if ! command -v gcloud &> /dev/null; then
    echo "⚠️  Google Cloud SDK not found. Please install it from:"
    echo "   https://cloud.google.com/sdk/docs/install"
else
    echo "Authenticating with Google Cloud..."
    gcloud auth application-default login
fi

# 6. Create convenience scripts
echo "🔧 Creating convenience scripts..."

# Create run script
cat > run_pipeline.sh << 'EOF'
#!/usr/bin/env bash
# Convenience script to run pipeline

case "${1:-full}" in
    watch)
        echo "👀 Starting continuous watch mode..."
        python scripts/pipeline.py --mode continuous --watch
        ;;
    train)
        echo "🏋️ Running training only..."
        python scripts/incremental_train.py
        ;;
    validate)
        echo "🔍 Validating annotations..."
        python scripts/validate_annotations.py
        ;;
    export)
        echo "📦 Exporting model..."
        python scripts/incremental_train.py --export "${2:-onnx}"
        ;;
    compare)
        echo "📊 Comparing model versions..."
        python scripts/incremental_train.py --compare
        ;;
    *)
        echo "🚀 Running full pipeline..."
        python scripts/pipeline.py --mode full
        ;;
esac
EOF

chmod +x run_pipeline.sh

# 7. Create example video processor
cat > process_new_videos.sh << 'EOF'
#!/usr/bin/env bash
# Drop videos here and run this script

if [ -z "$(ls -A data/videos/*.mp4 2>/dev/null)" ]; then
    echo "❌ No videos found in data/videos/"
    echo "📹 Please add .mp4 files to data/videos/ first"
    exit 1
fi

echo "🎬 Processing videos..."
python scripts/pipeline.py --mode full
EOF

chmod +x process_new_videos.sh

# 8. Print setup instructions
cat << 'EOF'

✅ Enhanced Pipeline Setup Complete!

📋 Next Steps:

1. Configure Label Studio API:
   - Open Label Studio (http://localhost:8080)
   - Go to User Settings > Access Tokens
   - Create new token and add to configs/pipeline.yaml
   - Copy your project ID from the URL

2. Drop your videos in:
   data/videos/

3. Run the pipeline:
   ./run_pipeline.sh         # Full pipeline
   ./run_pipeline.sh watch   # Continuous mode (recommended)

4. Other commands:
   ./run_pipeline.sh validate  # Check annotations
   ./run_pipeline.sh train     # Train only
   ./run_pipeline.sh compare   # Compare versions
   ./run_pipeline.sh export    # Export model

📊 Monitor training:
   tensorboard --logdir models/versions/

🚀 Pro tip: Use continuous mode for hands-free operation!

EOF

echo "🎉 Ready to go! Drop videos in data/videos/ and run ./run_pipeline.sh"