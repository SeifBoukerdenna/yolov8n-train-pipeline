from label_studio_sdk import Client
import yaml
import json
from pathlib import Path
import shutil
from datetime import datetime

class LabelStudioSync:
    def __init__(self, config):
        self.ls = Client(
            url=config['labelstudio']['url'],
            api_key=config['labelstudio']['api_key']
        )
        self.project_id = config['labelstudio']['project_id']
        self.project = self.ls.get_project(self.project_id)

    def auto_export_annotations(self, output_dir):
        """Automatically export annotations in YOLO format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = output_dir / f"export_{timestamp}"

        # Export annotations
        export_result = self.project.export_tasks(
            export_type="YOLO",
            download_all_tasks=False,  # Only completed tasks
        )

        # Process export
        temp_zip = export_dir / "export.zip"
        temp_zip.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_zip, 'wb') as f:
            f.write(export_result)

        # Extract
        shutil.unpack_archive(temp_zip, export_dir)
        temp_zip.unlink()

        # Organize files
        images_dir = export_dir / "images"
        labels_dir = export_dir / "labels"

        # Move files to correct structure
        for img_file in export_dir.glob("*.jpg"):
            img_file.rename(images_dir / img_file.name)
        for txt_file in export_dir.glob("*.txt"):
            if txt_file.name != "classes.txt":
                txt_file.rename(labels_dir / txt_file.name)

        print(f"✓ Exported {len(list(labels_dir.glob('*.txt')))} annotations")
        return export_dir

    def get_labeling_progress(self):
        """Get current labeling progress"""
        tasks = self.project.get_tasks()
        total = len(tasks)
        completed = sum(1 for t in tasks if t['is_labeled'])
        return {
            'total': total,
            'completed': completed,
            'percentage': (completed / total * 100) if total > 0 else 0
        }

    def import_new_frames(self, frames_dir, gcs_prefix):
        """Import new frames to Label Studio"""
        frames_dir = Path(frames_dir)
        tasks = []

        for img_path in frames_dir.glob("*.png"):
            # Create GCS URL
            gcs_url = f"gs://{gcs_prefix}/{img_path.name}"

            task = {
                "data": {
                    "image": gcs_url
                }
            }
            tasks.append(task)

        if tasks:
            self.project.import_tasks(tasks)
            print(f"✓ Imported {len(tasks)} new frames to Label Studio")

if __name__ == "__main__":
    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    ls_sync = LabelStudioSync(config)

    # Show progress
    progress = ls_sync.get_labeling_progress()
    print(f"Labeling Progress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)")

    # Export if enough labeled
    if progress['percentage'] > 80:
        ls_sync.auto_export_annotations("data/annotations")
