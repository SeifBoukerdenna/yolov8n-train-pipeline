import yaml
from pathlib import Path
import cv2
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

class AnnotationValidator:
    def __init__(self, images_dir, labels_dir, class_names):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def validate_dataset(self):
        """Run all validation checks"""
        print("🔍 Validating annotations...")

        # Get all image files
        image_files = list(self.images_dir.glob("*.png")) + \
                      list(self.images_dir.glob("*.jpg"))

        if not image_files:
            self.errors.append("No image files found!")
            return False

        # Check each image/label pair
        for img_path in tqdm(image_files, desc="Checking files"):
            self.validate_pair(img_path)

        # Analyze class distribution
        self.check_class_balance()

        # Generate report
        self.generate_report()

        return len(self.errors) == 0

    def validate_pair(self, img_path):
        """Validate image/label pair"""
        # Check if label exists
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            self.warnings.append(f"No label for {img_path.name}")
            self.stats['missing_labels'] += 1
            return

        # Load and validate image
        img = cv2.imread(str(img_path))
        if img is None:
            self.errors.append(f"Cannot read image: {img_path.name}")
            return

        h, w = img.shape[:2]

        # Validate annotations
        with open(label_path) as f:
            lines = f.readlines()

        if not lines:
            self.warnings.append(f"Empty label file: {label_path.name}")
            self.stats['empty_labels'] += 1
            return

        for line_num, line in enumerate(lines):
            self.validate_annotation_line(line, line_num, label_path, w, h)

    def validate_annotation_line(self, line, line_num, label_path, img_w, img_h):
        """Validate single annotation line"""
        parts = line.strip().split()

        if len(parts) != 5:
            self.errors.append(
                f"{label_path.name}:{line_num+1} - Invalid format "
                f"(expected 5 values, got {len(parts)})"
            )
            return

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            self.errors.append(
                f"{label_path.name}:{line_num+1} - Invalid number format"
            )
            return

        # Check class ID
        if class_id >= len(self.class_names):
            self.errors.append(
                f"{label_path.name}:{line_num+1} - Invalid class ID {class_id}"
            )
            return

        self.stats[f'class_{class_id}_count'] += 1

        # Check normalized coordinates
        if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
            self.errors.append(
                f"{label_path.name}:{line_num+1} - Coordinates out of range [0,1]"
            )
            return

        # Check for tiny boxes
        pixel_w = width * img_w
        pixel_h = height * img_h

        if pixel_w < 10 or pixel_h < 10:
            self.warnings.append(
                f"{label_path.name}:{line_num+1} - Very small box "
                f"({pixel_w:.0f}x{pixel_h:.0f} pixels)"
            )
            self.stats['tiny_boxes'] += 1

        # Check for boxes near edge
        if (x_center - width/2 < 0.02 or x_center + width/2 > 0.98 or
            y_center - height/2 < 0.02 or y_center + height/2 > 0.98):
            self.warnings.append(
                f"{label_path.name}:{line_num+1} - Box very close to image edge"
            )
            self.stats['edge_boxes'] += 1

    def check_class_balance(self):
        """Check if classes are reasonably balanced"""
        class_counts = [self.stats.get(f'class_{i}_count', 0)
                        for i in range(len(self.class_names))]

        if not any(class_counts):
            self.errors.append("No annotations found in dataset!")
            return

        total = sum(class_counts)

        for i, count in enumerate(class_counts):
            percentage = (count / total * 100) if total > 0 else 0
            self.stats[f'class_{i}_percentage'] = percentage

            if count == 0:
                self.warnings.append(
                    f"Class '{self.class_names[i]}' has no annotations!"
                )
            elif percentage < 5:
                self.warnings.append(
                    f"Class '{self.class_names[i]}' is underrepresented "
                    f"({percentage:.1f}% of annotations)"
                )

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*50)
        print("📊 VALIDATION REPORT")
        print("="*50)

        # Summary stats
        total_images = len(list(self.images_dir.glob("*.png")) +
                          list(self.images_dir.glob("*.jpg")))
        total_labels = len(list(self.labels_dir.glob("*.txt")))

        print(f"\n📁 Dataset Summary:")
        print(f"  - Total images: {total_images}")
        print(f"  - Total labels: {total_labels}")
        print(f"  - Missing labels: {self.stats['missing_labels']}")
        print(f"  - Empty labels: {self.stats['empty_labels']}")

        # Class distribution
        print(f"\n📊 Class Distribution:")
        for i, name in enumerate(self.class_names):
            count = self.stats.get(f'class_{i}_count', 0)
            pct = self.stats.get(f'class_{i}_percentage', 0)
            print(f"  - {name}: {count} ({pct:.1f}%)")

        # Issues summary
        print(f"\n⚠️  Issues Found:")
        print(f"  - Errors: {len(self.errors)}")
        print(f"  - Warnings: {len(self.warnings)}")
        print(f"  - Tiny boxes: {self.stats['tiny_boxes']}")
        print(f"  - Edge boxes: {self.stats['edge_boxes']}")

        # Show first few errors/warnings
        if self.errors:
            print(f"\n❌ First 5 Errors:")
            for err in self.errors[:5]:
                print(f"  - {err}")

        if self.warnings:
            print(f"\n⚠️  First 5 Warnings:")
            for warn in self.warnings[:5]:
                print(f"  - {warn}")

        # Save full report
        report = {
            'summary': {
                'total_images': total_images,
                'total_labels': total_labels,
                'missing_labels': self.stats['missing_labels'],
                'empty_labels': self.stats['empty_labels'],
            },
            'class_distribution': {
                name: {
                    'count': self.stats.get(f'class_{i}_count', 0),
                    'percentage': self.stats.get(f'class_{i}_percentage', 0)
                }
                for i, name in enumerate(self.class_names)
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': dict(self.stats)
        }

        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Full report saved to: {report_path}")

        # Final verdict
        if self.errors:
            print("\n❌ VALIDATION FAILED - Please fix errors before training!")
            return False
        elif len(self.warnings) > 20:
            print("\n⚠️  VALIDATION PASSED WITH MANY WARNINGS - Review before training")
            return True
        else:
            print("\n✅ VALIDATION PASSED - Dataset ready for training!")
            return True

def visualize_annotations(images_dir, labels_dir, class_names, num_samples=10):
    """Visualize random annotations for manual inspection"""
    import random
    import matplotlib.pyplot as plt

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Get random samples
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))

    for idx, img_path in enumerate(samples):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load annotations
        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h

                        # Draw box
                        x1 = int(x_center - box_w/2)
                        y1 = int(y_center - box_h/2)
                        x2 = int(x_center + box_w/2)
                        y2 = int(y_center + box_h/2)

                        cv2.rectangle(img, (x1, y1), (x2, y2),
                                      (colors[class_id][:3] * 255).astype(int).tolist(), 2)
                        cv2.putText(img, class_names[class_id], (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (colors[class_id][:3] * 255).astype(int).tolist(), 2)

        axes[idx].imshow(img)
        axes[idx].set_title(img_path.name, fontsize=10)
        axes[idx].axis('off')

    plt.suptitle("Sample Annotations - Visual Inspection", fontsize=16)
    plt.tight_layout()
    plt.savefig("annotation_samples.png", dpi=150)
    print("\n🖼️  Sample visualizations saved to: annotation_samples.png")

if __name__ == "__main__":
    # Load config
    with open("configs/data.yaml") as f:
        data_config = yaml.safe_load(f)

    # Validate latest export
    export_dirs = sorted(Path("data/annotations").glob("export_*"))

    if not export_dirs:
        print("❌ No exported annotations found!")
        exit(1)

    latest_export = export_dirs[-1]
    print(f"Validating: {latest_export}")

    validator = AnnotationValidator(
        latest_export / "images",
        latest_export / "labels",
        data_config['names']
    )

    # Run validation
    is_valid = validator.validate_dataset()

    # Generate visualizations
    if is_valid or len(validator.errors) < 10:
        visualize_annotations(
            latest_export / "images",
            latest_export / "labels",
            data_config['names']
        )

    exit(0 if is_valid else 1)