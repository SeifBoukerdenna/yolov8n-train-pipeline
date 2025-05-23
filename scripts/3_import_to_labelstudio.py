# scripts/3_import_to_labelstudio.py

import yaml
from pathlib import Path
from label_studio_sdk import Client
from google.cloud import storage
from datetime import datetime, timedelta

def create_signed_urls(bucket_name, prefix):
    """Create signed URLs for all images in bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    urls = []

    for blob in blobs:
        if blob.name.endswith('.png'):
            # Use public URL instead of signed URL
            url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
            urls.append(url)

    return urls

def import_to_labelstudio(urls, ls_config):
    """Import URLs to Label Studio"""
    ls = Client(url=ls_config['url'], api_key=ls_config['api_key'])
    project = ls.get_project(ls_config['project_id'])

    # Clear existing tasks
    existing_tasks = project.get_tasks()
    if existing_tasks:
        print(f"Deleting {len(existing_tasks)} existing tasks...")
        for task in existing_tasks:
            project.delete_task(task['id'])

    # Create new tasks
    tasks = [{"data": {"image": url}} for url in urls]

    if tasks:
        print(f"Importing {len(tasks)} images...")
        project.import_tasks(tasks)
        print(f"✅ Imported {len(tasks)} images to Label Studio")
    else:
        print("❌ No images found")

def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Creating signed URLs...")
    urls = create_signed_urls(
        config['gcs']['bucket'],
        config['gcs']['prefix']
    )

    print(f"Found {len(urls)} images")

    import_to_labelstudio(urls, config['labelstudio'])

    print(f"\n🎨 Go label at: {config['labelstudio']['url']}")

if __name__ == "__main__":
    main()