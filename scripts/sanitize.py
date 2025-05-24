# scripts/sanitize.py

import argparse
import random
from pathlib import Path

def find_labeled_and_empty_pairs(export_dir):
    """Find both labeled and empty image/label pairs"""
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Images or labels directory not found")
        return [], []

    labeled_pairs = []
    empty_pairs = []

    # Check all label files
    for label_file in labels_dir.glob("*.txt"):
        # Check if file is empty or contains only whitespace
        with open(label_file, 'r') as f:
            content = f.read().strip()

        # Find corresponding image
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_file = None
        for ext in image_extensions:
            potential_image = images_dir / f"{label_file.stem}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break

        if image_file:
            if content:  # Has labels
                labeled_pairs.append((image_file, label_file))
            else:  # Empty file
                empty_pairs.append((image_file, label_file))

    return labeled_pairs, empty_pairs

def sanitize_dataset(export_dir, keep_percentage=25, random_seed=42):
    """Remove empty label pairs based on percentage of labeled pairs"""

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Find labeled and empty pairs
    labeled_pairs, empty_pairs = find_labeled_and_empty_pairs(export_dir)

    if not empty_pairs:
        print("✅ No empty label files found")
        return

    total_labeled = len(labeled_pairs)
    total_empty = len(empty_pairs)

    # Calculate how many empty pairs to keep based on labeled count
    target_empty_count = int(total_labeled * keep_percentage / 100)
    keep_count = min(target_empty_count, total_empty)
    delete_count = total_empty - keep_count

    print(f"📊 Dataset Analysis:")
    print(f"   Labeled pairs: {total_labeled}")
    print(f"   Empty pairs: {total_empty}")
    print(f"   Target empty ratio: {keep_percentage}% of labeled count")
    print(f"   Target empty count: {target_empty_count}")
    print(f"✅ Keeping: {keep_count} empty pairs")
    print(f"🗑️  Deleting: {delete_count} empty pairs")

    if delete_count == 0:
        print("✅ No files to delete")
        return

    # Randomly select pairs to delete
    random.shuffle(empty_pairs)
    pairs_to_delete = empty_pairs[:delete_count]

    # Delete selected pairs
    deleted_count = 0
    for image_file, label_file in pairs_to_delete:
        try:
            image_file.unlink()
            label_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"❌ Failed to delete {image_file.name}: {e}")

    print(f"✅ Successfully deleted {deleted_count} image/label pairs")

    # Show final stats
    remaining_labeled, remaining_empty = find_labeled_and_empty_pairs(export_dir)
    total_remaining = len(remaining_labeled) + len(remaining_empty)
    empty_ratio = (len(remaining_empty) / len(remaining_labeled) * 100) if remaining_labeled else 0

    print(f"\n📈 Final Statistics:")
    print(f"   Total pairs: {total_remaining}")
    print(f"   Labeled pairs: {len(remaining_labeled)}")
    print(f"   Empty pairs: {len(remaining_empty)}")
    print(f"   Empty/Labeled ratio: {empty_ratio:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Sanitize dataset by removing empty label pairs')
    parser.add_argument('keep_percentage', type=float, default=25, nargs='?',
                       help='Percentage of empty pairs to keep (0-100, default: 25)')
    parser.add_argument('--export-dir', type=str,
                       help='Specific export directory to sanitize (default: latest)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')

    args = parser.parse_args()

    # Validate percentage
    if not 0 <= args.keep_percentage <= 100:
        print("❌ Keep percentage must be between 0 and 100")
        return

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

    print(f"📂 Sanitizing export: {export_dir}")

    # Confirm action
    print(f"⚠️  This will permanently delete empty image/label pairs")
    print(f"   Keep percentage: {args.keep_percentage}%")

    response = input("Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("❌ Cancelled")
        return

    # Perform sanitization
    sanitize_dataset(export_dir, args.keep_percentage, args.seed)

if __name__ == "__main__":
    main()