# scripts/5_split_dataset.py

import yaml
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(export_dir, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    """Split dataset into train/val folders"""

    # Set random seed for reproducibility
    random.seed(random_seed)

    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Images or labels directory not found in export")
        return None

    # Get all image files
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    if not image_files:
        print("❌ No image files found")
        return None

    # Filter to only images that have corresponding labels
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))

    if not valid_pairs:
        print("❌ No matching image-label pairs found")
        return None

    print(f"📊 Found {len(valid_pairs)} image-label pairs")

    # Shuffle the pairs
    random.shuffle(valid_pairs)

    # Calculate split indices
    total = len(valid_pairs)
    train_count = int(total * train_ratio)
    val_count = total - train_count

    print(f"📈 Split: {train_count} train, {val_count} validation")

    # Create train/val directories
    train_images_dir = export_dir / "train" / "images"
    train_labels_dir = export_dir / "train" / "labels"
    val_images_dir = export_dir / "val" / "images"
    val_labels_dir = export_dir / "val" / "labels"

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy files to train set
    print("📁 Copying training files...")
    for i in range(train_count):
        img_file, label_file = valid_pairs[i]
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)

    # Copy files to validation set
    print("📁 Copying validation files...")
    for i in range(train_count, total):
        img_file, label_file = valid_pairs[i]
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)

    return {
        'train_count': train_count,
        'val_count': val_count,
        'train_path': export_dir / "train",
        'val_path': export_dir / "val"
    }

def create_dataset_yaml(export_dir, classes, split_info):
    """Create dataset.yaml for training with proper train/val split"""
    dataset_yaml = export_dir / "dataset.yaml"

    yaml_content = {
        'train': str((export_dir / 'train' / 'images').absolute()),
        'val': str((export_dir / 'val' / 'images').absolute()),
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(dataset_yaml, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"📄 Created dataset.yaml")
    return dataset_yaml

def analyze_split(export_dir):
    """Analyze the class distribution in train/val splits"""
    train_labels_dir = export_dir / "train" / "labels"
    val_labels_dir = export_dir / "val" / "labels"

    def count_classes(labels_dir):
        class_counts = defaultdict(int)
        total_objects = 0

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        total_objects += 1

        return class_counts, total_objects

    train_classes, train_objects = count_classes(train_labels_dir)
    val_classes, val_objects = count_classes(val_labels_dir)

    print(f"\n📊 Dataset Analysis:")
    print(f"Training: {len(list(train_labels_dir.glob('*.txt')))} images, {train_objects} objects")
    print(f"Validation: {len(list(val_labels_dir.glob('*.txt')))} images, {val_objects} objects")

    if train_classes or val_classes:
        print(f"\nClass distribution:")
        all_classes = set(train_classes.keys()) | set(val_classes.keys())
        for class_id in sorted(all_classes):
            train_count = train_classes.get(class_id, 0)
            val_count = val_classes.get(class_id, 0)
            total = train_count + val_count
            train_pct = (train_count / total * 100) if total > 0 else 0
            val_pct = (val_count / total * 100) if total > 0 else 0
            print(f"  Class {class_id}: {train_count} train ({train_pct:.1f}%), {val_count} val ({val_pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val folders')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--export-dir', type=str,
                       help='Specific export directory to split (default: latest)')

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio - 1.0) > 0.001:
        print("❌ Train and validation ratios must sum to 1.0")
        return

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Find export directory
    if args.export_dir:
        export_dir = Path(args.export_dir)
    else:
        annotations_dir = Path("data/annotations")
        exports = sorted(annotations_dir.glob("export_*"))

        if not exports:
            print("❌ No exported annotations found")
            return

        export_dir = exports[-1]

    if not export_dir.exists():
        print(f"❌ Export directory not found: {export_dir}")
        return

    print(f"📂 Using export: {export_dir}")

    # Check for sanitization
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    if images_dir.exists() and labels_dir.exists():
        # Count empty labels
        empty_count = 0
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                if not f.read().strip():
                    empty_count += 1

        total_labels = len(list(labels_dir.glob("*.txt")))

        if empty_count > 0:
            empty_percentage = (empty_count / total_labels * 100) if total_labels > 0 else 0
            print(f"\n⚠️  DATA SANITIZATION NOTICE:")
            print(f"   Found {empty_count} empty labels out of {total_labels} ({empty_percentage:.1f}%)")
            print(f"   Consider running: python scripts/sanitize.py 25")
            print(f"   This will remove 75% of empty label pairs")

            response = input("\nContinue with split anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("❌ Split cancelled. Consider sanitizing data first.")
                return

    # Split dataset
    split_info = split_dataset(
        export_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )

    if split_info:
        # Create dataset.yaml
        dataset_yaml = create_dataset_yaml(export_dir, config['classes'], split_info)

        # Analyze the split
        analyze_split(export_dir)

        print(f"\n✅ Dataset split complete!")
        print(f"📁 Training data: {split_info['train_path']}")
        print(f"📁 Validation data: {split_info['val_path']}")
        print(f"📄 Dataset config: {dataset_yaml}")
        print(f"\n🚀 Ready for training with: python scripts/6_train_model.py train")

if __name__ == "__main__":
    main()