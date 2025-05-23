# scripts/6_train_model.py

import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil

def find_dataset_yaml(export_dir=None):
    """Find the dataset.yaml file"""
    if export_dir:
        dataset_yaml = Path(export_dir) / "dataset.yaml"
        if dataset_yaml.exists():
            return dataset_yaml

    # Find latest export with dataset.yaml
    annotations_dir = Path("data/annotations")
    exports = sorted(annotations_dir.glob("export_*"), reverse=True)

    for export in exports:
        dataset_yaml = export / "dataset.yaml"
        if dataset_yaml.exists():
            return dataset_yaml

    return None

def create_legacy_dataset_yaml(export_dir, classes):
    """Create dataset.yaml for old-style exports (backward compatibility)"""
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
    parser.add_argument('--export-dir', help='Specific export directory to use')

    args = parser.parse_args()

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        # Find dataset.yaml
        dataset_yaml = find_dataset_yaml(args.export_dir)

        if not dataset_yaml:
            # Try to find latest export and check if it needs splitting
            annotations_dir = Path("data/annotations")
            exports = sorted(annotations_dir.glob("export_*"))

            if not exports:
                print("❌ No exported annotations found")
                return

            latest_export = exports[-1]

            # Check if it has train/val structure
            if (latest_export / "train").exists() and (latest_export / "val").exists():
                # Has split structure but no dataset.yaml - create it
                dataset_yaml = create_dataset_yaml(latest_export, config['classes'])
            else:
                print("⚠️  Export found but not split into train/val folders")
                print("🔧 Please run: python scripts/5_split_dataset.py")
                print("   Then try training again")
                return

        export_dir = dataset_yaml.parent
        print(f"📂 Using dataset: {export_dir}")
        print(f"📄 Using config: {dataset_yaml}")

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
        dataset_yaml = find_dataset_yaml(args.export_dir)

        if not dataset_yaml:
            print("❌ No dataset.yaml found. Please run training first or split your dataset.")
            return

        validate_model(model_path, dataset_yaml)

    elif args.mode == 'detect':
        model_path = args.model or "models/best.pt"
        source_dir = args.source or "data/frames"

        detect_images(model_path, source_dir, Path("runs"))

def create_dataset_yaml(export_dir, classes):
    """Create dataset.yaml for properly split dataset"""
    dataset_yaml = export_dir / "dataset.yaml"

    yaml_content = {
        'train': str((export_dir / 'train' / 'images').absolute()),
        'val': str((export_dir / 'val' / 'images').absolute()),
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(dataset_yaml, 'w') as f:
        yaml.dump(yaml_content, f)

    return dataset_yaml

if __name__ == "__main__":
    main()