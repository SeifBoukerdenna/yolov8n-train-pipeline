# scripts/5_train_model.py

import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil

def create_dataset_yaml(export_dir, classes):
    """Create dataset.yaml for training"""
    dataset_yaml = export_dir / "dataset.yaml"

    yaml_content = {
        'train': str(export_dir.absolute() / 'images'),
        'val': str(export_dir.absolute() / 'images'),  # Using same for simplicity
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(dataset_yaml, 'w') as f:
        yaml.dump(yaml_content, f)

    return dataset_yaml

def train_model(dataset_yaml, config, output_dir):
    """Train YOLO model"""
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=str(dataset_yaml),
        epochs=config['training']['epochs'],
        imgsz=config['training']['img_size'],
        batch=config['training']['batch_size'],
        device=config['training']['device'],
        project=str(output_dir),
        name='train',
        exist_ok=True
    )

    return results

def validate_model(model_path, dataset_yaml):
    """Validate trained model"""
    model = YOLO(model_path)
    results = model.val(data=str(dataset_yaml))

    print(f"\n📊 Validation Results:")
    print(f"mAP@50: {results.box.map50:.3f}")
    print(f"mAP@50-95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")

def detect_images(model_path, source_dir, output_dir):
    """Run detection on images"""
    model = YOLO(model_path)

    results = model.predict(
        source=str(source_dir),
        save=True,
        project=str(output_dir),
        name='detect',
        exist_ok=True
    )

    print(f"✅ Detection results saved to {output_dir}/detect/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'validate', 'detect'],
                       help='Mode: train, validate, or detect')
    parser.add_argument('--model', help='Model path for validate/detect')
    parser.add_argument('--source', help='Source directory for detect')

    args = parser.parse_args()

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        # Find latest export
        annotations_dir = Path("data/annotations")
        exports = sorted(annotations_dir.glob("export_*"))

        if not exports:
            print("❌ No exported annotations found")
            return

        latest_export = exports[-1]
        print(f"Using export: {latest_export}")

        # Create dataset YAML
        dataset_yaml = create_dataset_yaml(latest_export, config['classes'])

        # Train
        print("🚀 Starting training...")
        results = train_model(dataset_yaml, config, Path("models"))

        # Copy best model
        best_model = Path("models/train/weights/best.pt")
        final_model = Path("models/best.pt")
        shutil.copy(best_model, final_model)

        print(f"✅ Training complete! Model saved to {final_model}")

        # Auto-validate
        validate_model(final_model, dataset_yaml)

    elif args.mode == 'validate':
        model_path = args.model or "models/best.pt"

        # Find dataset
        annotations_dir = Path("data/annotations")
        exports = sorted(annotations_dir.glob("export_*"))
        latest_export = exports[-1]
        dataset_yaml = latest_export / "dataset.yaml"

        validate_model(model_path, dataset_yaml)

    elif args.mode == 'detect':
        model_path = args.model or "models/best.pt"
        source_dir = args.source or "data/frames"

        detect_images(model_path, source_dir, Path("runs"))

if __name__ == "__main__":
    main()