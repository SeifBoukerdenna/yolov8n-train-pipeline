import os
from pathlib import Path
from datetime import datetime
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import time

console = Console()

class PipelineStatus:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        try:
            with open("configs/pipeline.yaml") as f:
                return yaml.safe_load(f)
        except:
            return {}

    def get_video_status(self):
        """Check video processing status"""
        video_dir = Path("data/videos")
        processed_dir = Path("data/processed")

        videos = list(video_dir.glob("*.mp4"))
        processed = set()

        for frame_dir in processed_dir.iterdir():
            if frame_dir.is_dir():
                processed.add(frame_dir.name)

        return {
            'total': len(videos),
            'processed': len(processed),
            'pending': [v.name for v in videos if v.stem not in processed]
        }

    def get_frame_status(self):
        """Check frame extraction status"""
        processed_dir = Path("data/processed")
        total_frames = sum(len(list(d.glob("*.png"))) for d in processed_dir.iterdir() if d.is_dir())

        # Check GCS sync status
        sync_state_file = Path(".gcs_sync_state.json")
        synced_frames = 0
        if sync_state_file.exists():
            with open(sync_state_file) as f:
                sync_state = json.load(f)
                synced_frames = len(sync_state)

        return {
            'extracted': total_frames,
            'synced_to_gcs': synced_frames,
            'pending_sync': total_frames - synced_frames
        }

    def get_labeling_status(self):
        """Check Label Studio progress"""
        try:
            from label_studio_sdk import Client
            ls = Client(
                url=self.config['labelstudio']['url'],
                api_key=self.config['labelstudio']['api_key']
            )
            project = ls.get_project(self.config['labelstudio']['project_id'])
            tasks = project.get_tasks()

            total = len(tasks)
            completed = sum(1 for t in tasks if t.get('is_labeled', False))
            skipped = sum(1 for t in tasks if t.get('was_cancelled', False))

            return {
                'total_tasks': total,
                'completed': completed,
                'skipped': skipped,
                'remaining': total - completed - skipped,
                'percentage': (completed / total * 100) if total > 0 else 0
            }
        except:
            return {
                'total_tasks': 0,
                'completed': 0,
                'skipped': 0,
                'remaining': 0,
                'percentage': 0,
                'error': 'Cannot connect to Label Studio'
            }

    def get_training_status(self):
        """Check model training status"""
        versions_file = Path("models/versions/versions.json")

        if not versions_file.exists():
            return {'trained_models': 0, 'latest': None}

        with open(versions_file) as f:
            versions = json.load(f)

        if not versions:
            return {'trained_models': 0, 'latest': None}

        latest = versions[-1]
        return {
            'trained_models': len(versions),
            'latest': latest['version'],
            'latest_date': latest['timestamp'][:10],
            'latest_map': latest.get('metrics', {}).get('mAP50', 0),
            'dataset_size': latest.get('dataset_stats', {}).get('total_annotations', 0)
        }

    def create_status_display(self):
        """Create beautiful status display"""
        # Get all status info
        video_status = self.get_video_status()
        frame_status = self.get_frame_status()
        label_status = self.get_labeling_status()
        train_status = self.get_training_status()

        # Create main table
        table = Table(title="🚀 YOLOv8n Pipeline Status",
                      title_style="bold magenta",
                      border_style="cyan")

        table.add_column("Stage", style="cyan", width=20)
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        # 1. Video Processing
        video_details = f"{video_status['processed']}/{video_status['total']} processed"
        if video_status['pending']:
            video_details += f"\nPending: {', '.join(video_status['pending'][:3])}"
            if len(video_status['pending']) > 3:
                video_details += f" +{len(video_status['pending'])-3} more"

        table.add_row(
            "📹 Videos",
            "✅ Ready" if video_status['processed'] > 0 else "⏳ Waiting",
            video_details
        )

        # 2. Frame Extraction
        table.add_row(
            "🖼️  Frames",
            "✅ Extracted" if frame_status['extracted'] > 0 else "⏳ Pending",
            f"{frame_status['extracted']} frames extracted\n" +
            f"{frame_status['synced_to_gcs']} synced to GCS"
        )

        # 3. Labeling Progress
        if 'error' in label_status:
            label_status_text = "❌ Error"
            label_details = label_status['error']
        else:
            if label_status['percentage'] < 20:
                label_status_text = "🆕 Just Started"
            elif label_status['percentage'] < 80:
                label_status_text = "🎨 In Progress"
            elif label_status['percentage'] < 100:
                label_status_text = "📈 Almost Done!"
            else:
                label_status_text = "✅ Complete!"

            label_details = (
                f"{label_status['completed']}/{label_status['total_tasks']} labeled "
                f"({label_status['percentage']:.1f}%)\n"
                f"{label_status['remaining']} remaining"
            )

        table.add_row("🏷️  Labeling", label_status_text, label_details)

        # 4. Training
        if train_status['trained_models'] == 0:
            train_status_text = "⏳ Not Started"
            train_details = "Need labeled data first"
        else:
            train_status_text = f"✅ {train_status['latest']}"
            train_details = (
                f"Models trained: {train_status['trained_models']}\n"
                f"Latest mAP@50: {train_status['latest_map']:.3f}\n"
                f"Dataset size: {train_status['dataset_size']} annotations"
            )

        table.add_row("🤖 Training", train_status_text, train_details)

        # Create progress bar for labeling
        if label_status['total_tasks'] > 0 and 'error' not in label_status:
            progress = Progress(
                TextColumn("[bold blue]Labeling Progress"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            task = progress.add_task("", total=100, completed=label_status['percentage'])
        else:
            progress = None

        # Next action recommendation
        if video_status['total'] == 0:
            next_action = Panel(
                "📹 Add videos to data/videos/ to get started!",
                title="Next Action",
                border_style="red"
            )
        elif video_status['processed'] < video_status['total']:
            next_action = Panel(
                "🔄 Run: ./run_pipeline.sh to process videos",
                title="Next Action",
                border_style="yellow"
            )
        elif label_status['total_tasks'] == 0:
            next_action = Panel(
                "⏳ Waiting for frames to sync to Label Studio...",
                title="Next Action",
                border_style="yellow"
            )
        elif label_status['percentage'] < 80:
            next_action = Panel(
                f"🎨 Continue labeling in Label Studio!\n"
                f"   {label_status['remaining']} frames left\n"
                f"   Open: {self.config.get('labelstudio', {}).get('url', 'http://localhost:8080')}",
                title="Next Action",
                border_style="green"
            )
        elif train_status['trained_models'] == 0:
            next_action = Panel(
                "🚀 Ready to train! Run: ./run_pipeline.sh train",
                title="Next Action",
                border_style="green"
            )
        else:
            next_action = Panel(
                "✅ Pipeline complete! Add more videos to improve model\n"
                f"   Current mAP@50: {train_status['latest_map']:.3f}",
                title="Status",
                border_style="green"
            )

        return table, progress, next_action

    def display_once(self):
        """Display status once"""
        console.clear()
        table, progress, next_action = self.create_status_display()

        console.print("\n")
        console.print(table)

        if progress:
            console.print("\n")
            console.print(progress)

        console.print("\n")
        console.print(next_action)
        console.print("\n")

        # Show timestamp
        console.print(f"[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    def watch(self, interval=5):
        """Watch mode - updates every interval seconds"""
        console.print("[bold green]Starting pipeline monitor...[/bold green]")
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        try:
            while True:
                self.display_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[bold red]Monitoring stopped[/bold red]")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8n Pipeline Status Monitor")
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch mode - updates every 5 seconds')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='Update interval in seconds (default: 5)')

    args = parser.parse_args()

    status = PipelineStatus()

    if args.watch:
        status.watch(args.interval)
    else:
        status.display_once()

if __name__ == "__main__":
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("Installing required package: rich")
        import subprocess
        subprocess.check_call(["pip", "install", "rich"])
        print("Package installed! Please run the script again.")
        exit(0)

    main()