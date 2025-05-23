# scripts/nuke.py

import shutil
import argparse
from pathlib import Path

def get_directories_to_delete():
    """Get list of directories that will be deleted"""
    return [
        "data/videos",
        "data/frames",
        "data/annotations",
        "models",
        "runs",
        "test_results"
    ]

def calculate_deletion_size():
    """Calculate total size of files to be deleted"""
    total_size = 0
    file_count = 0

    for dir_path in get_directories_to_delete():
        path = Path(dir_path)
        if path.exists():
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

    # Convert to MB
    size_mb = total_size / (1024 * 1024)
    return file_count, size_mb

def show_deletion_preview():
    """Show what will be deleted"""
    print("🗑️  NUKE OPERATION - DELETION PREVIEW:")
    print("=" * 50)

    file_count, size_mb = calculate_deletion_size()

    for dir_path in get_directories_to_delete():
        path = Path(dir_path)
        if path.exists():
            files = list(path.rglob("*"))
            file_count_dir = len([f for f in files if f.is_file()])
            print(f"📁 {dir_path}: {file_count_dir} files")
        else:
            print(f"📁 {dir_path}: not found")

    print("=" * 50)
    print(f"📊 TOTAL: {file_count} files ({size_mb:.1f} MB)")
    print("=" * 50)

def nuke_directories(dry_run=False):
    """Delete all data directories"""
    deleted_dirs = []

    for dir_path in get_directories_to_delete():
        path = Path(dir_path)
        if path.exists():
            if dry_run:
                print(f"[DRY RUN] Would delete: {dir_path}")
            else:
                try:
                    shutil.rmtree(path)
                    print(f"✅ Deleted: {dir_path}")
                    deleted_dirs.append(dir_path)
                except Exception as e:
                    print(f"❌ Failed to delete {dir_path}: {e}")
        else:
            print(f"⏭️  Skipped: {dir_path} (not found)")

    return deleted_dirs

def main():
    parser = argparse.ArgumentParser(description='NUKE: Delete all pipeline data')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt')

    args = parser.parse_args()

    print("💥 NUKE OPERATION")
    print("⚠️  WARNING: This will permanently delete ALL pipeline data!")
    print()

    show_deletion_preview()

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Nothing will be deleted")
        nuke_directories(dry_run=True)
        return

    print("\n⚠️  FINAL WARNING:")
    print("This action is IRREVERSIBLE and will delete:")
    print("• All videos in data/videos/")
    print("• All extracted frames")
    print("• All annotations and exports")
    print("• All trained models")
    print("• All test results")
    print()

    if not args.force:
        print("Type 'DELETE EVERYTHING' to confirm:")
        confirmation = input("> ").strip()

        if confirmation != "DELETE EVERYTHING":
            print("❌ Operation cancelled")
            return

    print("\n🗑️  Starting deletion...")
    deleted_dirs = nuke_directories()

    print(f"\n💥 NUKE COMPLETE!")
    print(f"✅ Deleted {len(deleted_dirs)} directories")
    print("🚀 Ready for fresh start!")

if __name__ == "__main__":
    main()