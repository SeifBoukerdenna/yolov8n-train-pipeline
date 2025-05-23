#!/usr/bin/env python3
# scripts/4_export_annotations.py

import yaml
import shutil
from pathlib import Path
from label_studio_sdk import Client
from datetime import datetime

def export_annotations(ls_config, output_dir):
    """Export annotations from Label Studio"""
    ls = Client(url=ls_config['url'], api_key=ls_config['api_key'])
    project = ls.get_project(ls_config['project_id'])

    # Check progress
    tasks = project.get_tasks()
    completed = sum(1 for t in tasks if t.get('is_labeled', False))
    total = len(tasks)

    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    if completed == 0:
        print("❌ No labeled tasks found")
        return None

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(output_dir) / f"export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export using API endpoint directly
    import requests

    headers = {"Authorization": f"Token {ls_config['api_key']}"}
    export_url = f"{ls_config['url']}/api/projects/{ls_config['project_id']}/export"

    params = {
        "exportType": "YOLO",
        "download_all_tasks": "false"
    }

    response = requests.get(export_url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"❌ Export failed: {response.text}")
        return None

    export_result = response.content

    # Save and extract
    temp_zip = export_dir / "export.zip"
    with open(temp_zip, 'wb') as f:
        f.write(export_result)

    shutil.unpack_archive(temp_zip, export_dir)
    temp_zip.unlink()

    # Organize files
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Move files
    for img_file in export_dir.glob("*.jpg"):
        img_file.rename(images_dir / img_file.name)
    for txt_file in export_dir.glob("*.txt"):
        if txt_file.name != "classes.txt":
            txt_file.rename(labels_dir / txt_file.name)

    # Download images from tasks
    print("Downloading images...")
    import requests

    # Create mapping of task IDs to image names
    task_image_mapping = {}
    for task in tasks:
        if task.get('is_labeled', False):
            task_id = task['id']
            image_url = task['data']['image']
            original_name = image_url.split('/')[-1].split('?')[0]  # Remove query params
            task_image_mapping[task_id] = original_name

    # Download images and rename labels to match
    downloaded_images = []
    for task in tasks:
        if task.get('is_labeled', False):
            task_id = task['id']
            image_url = task['data']['image']
            original_name = task_image_mapping[task_id]
            base_name = original_name.replace('.png', '')

            try:
                # Download image
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    with open(images_dir / original_name, 'wb') as f:
                        f.write(img_response.content)
                    downloaded_images.append(base_name)
            except Exception as e:
                print(f"Failed to download {original_name}: {e}")

    # Rename label files to match image names
    for txt_file in labels_dir.glob("*.txt"):
        # Find corresponding image
        for base_name in downloaded_images:
            if base_name in str(txt_file):
                new_name = f"{base_name}.txt"
                txt_file.rename(labels_dir / new_name)
                break

    print(f"✅ Exported {len(list(labels_dir.glob('*.txt')))} annotations to {export_dir}")
    print(f"✅ Downloaded {len(list(images_dir.glob('*.png')))} images")
    return export_dir

def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    export_dir = export_annotations(config['labelstudio'], "data/annotations")

    if export_dir:
        print(f"\n📁 Files ready at: {export_dir}")

if __name__ == "__main__":
    main()