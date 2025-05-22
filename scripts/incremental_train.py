import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil
import json
from datetime import datetime
import hashlib
import subprocess

class IncrementalTrainer:
    def __init__(self, config_path="configs/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.versions_dir = Path("models/versions")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.version_info_file = self.versions_dir / "versions.json"
        self.load_version_info()

    def load_version_info(self):
        """Load version history"""
        if self.version_info_file.exists():
            with open(self.version_info_file) as f:
                self.versions = json.load(f)
        else:
            self.versions = []

    def save_version_info(self):
        """Save version history"""
        with open(self.version_info_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def get_dataset_hash(self, dataset_dir):
        """Generate hash of dataset for change detection"""
        dataset_dir = Path(dataset_dir)
        hash_md5 = hashlib.md5()

        # Hash all label files
        label_files = sorted(dataset_dir.glob("labels/**/*.txt"))
        for label_file in label_files:
            hash_md5.update(label_file.read_bytes())

        return hash_md5.hexdigest()[:8]

    def get_latest_checkpoint(self):
        """Get the best checkpoint from previous training"""
        if not self.versions:
            return None

        latest = self.versions[-1]
        checkpoint_path = Path(latest['checkpoint_path'])

        if checkpoint_path.exists():
            return checkpoint_path
        return None

    def determine_training_strategy(self, dataset_dir):
        """Determine optimal training strategy"""
        dataset_hash = self.get_dataset_hash(dataset_dir)

        if not self.versions:
            print("🆕 First training - starting from pretrained YOLOv8n")
            return {
                'strategy': 'new',
                'checkpoint': 'yolov8n.pt',
                'epochs': self.config['training'].get('initial_epochs', 50)
            }

        # Check if dataset changed
        last_version = self.versions[-1]
        if last_version['dataset_hash'] == dataset_hash:
            print("ℹ️  Dataset unchanged - no training needed")
            return {'strategy': 'skip'}

        # Calculate dataset changes
        old_stats = last_version.get('dataset_stats', {})
        new_stats = self.get_dataset_stats(dataset_dir)

        # Determine incremental vs full retrain
        total_old = old_stats.get('total_annotations', 0)
        total_new = new_stats['total_annotations']
        change_ratio = abs(total_new - total_old) / max(total_old, 1)

        if change_ratio > 0.5:  # More than 50% change
            print(f"🔄 Major dataset change ({change_ratio:.1%}) - full retrain recommended")
            return {
                'strategy': 'retrain',
                'checkpoint': 'yolov8n.pt',
                'epochs': self.config['training'].get('initial_epochs', 50)
            }
        else:
            print(f"📈 Minor dataset change ({change_ratio:.1%}) - incremental training")
            return {
                'strategy': 'incremental',
                'checkpoint': self.get_latest_checkpoint(),
                'epochs': self.config['training'].get('incremental_epochs', 20)
            }

    def get_dataset_stats(self, dataset_dir):
        """Get dataset statistics"""
        dataset_dir = Path(dataset_dir)

        stats = {
            'total_images': len(list((dataset_dir / 'images/train').glob('*'))),
            'total_annotations': 0,
            'class_distribution': {}
        }

        # Count annotations
        for label_file in (dataset_dir / 'labels/train').glob('*.txt'):
            lines = label_file.read_text().strip().split('\n')
            stats['total_annotations'] += len(lines)

            for line in lines:
                if line:
                    class_id = int(line.split()[0])
                    stats['class_distribution'][class_id] = \
                        stats['class_distribution'].get(class_id, 0) + 1

        return stats

    def train_model(self, strategy_info, dataset_dir):
        """Execute training with given strategy"""
        if strategy_info['strategy'] == 'skip':
            print("✅ Using existing model (dataset unchanged)")
            return self.versions[-1]['version']

        # Create new version
        version_num = len(self.versions) + 1
        version_name = f"v{version_num}"
        version_dir = self.versions_dir / version_name
        version_dir.mkdir(exist_ok=True)

        print(f"\n🚀 Training {version_name}...")
        print(f"   Strategy: {strategy_info['strategy']}")
        print(f"   Checkpoint: {strategy_info['checkpoint']}")
        print(f"   Epochs: {strategy_info['epochs']}")

        # Load model
        model = YOLO(str(strategy_info['checkpoint']))

        # Train
        results = model.train(
            data="configs/data.yaml",
            epochs=strategy_info['epochs'],
            imgsz=self.config['training'].get('img_size', 640),
            batch=self.config['training'].get('batch_size', 16),
            project=str(version_dir),
            name="train",
            exist_ok=True,
            patience=self.config['training'].get('early_stopping_patience', 10),
            save=True,
            save_period=5,  # Save checkpoint every 5 epochs
            plots=True,
            amp=True,  # Automatic mixed precision for faster training
            cache=True,  # Cache images for faster training
        )

        # Copy best weights
        best_weights = version_dir / "train/weights/best.pt"
        final_weights = version_dir / "best.pt"
        shutil.copy(best_weights, final_weights)

        # Save version info
        version_info = {
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_info['strategy'],
            'parent_version': self.versions[-1]['version'] if self.versions else None,
            'checkpoint_path': str(final_weights),
            'dataset_hash': self.get_dataset_hash(dataset_dir),
            'dataset_stats': self.get_dataset_stats(dataset_dir),
            'training_args': {
                'epochs': strategy_info['epochs'],
                'batch_size': self.config['training'].get('batch_size', 16),
                'img_size': self.config['training'].get('img_size', 640),
            },
            'metrics': self.extract_metrics(version_dir / "train")
        }

        self.versions.append(version_info)
        self.save_version_info()

        # Create symlink to latest
        latest_link = self.versions_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.name)

        print(f"\n✅ Training complete! Model saved as {version_name}")
        self.print_metrics_summary(version_info['metrics'])

        return version_name

    def extract_metrics(self, train_dir):
        """Extract training metrics from results"""
        results_csv = train_dir / "results.csv"
        metrics = {}

        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)

            # Get best epoch metrics
            best_idx = df['fitness'].idxmax()
            metrics = {
                'best_epoch': int(best_idx) + 1,
                'mAP50': float(df.loc[best_idx, 'metrics/mAP50']),
                'mAP50-95': float(df.loc[best_idx, 'metrics/mAP50-95']),
                'precision': float(df.loc[best_idx, 'metrics/precision']),
                'recall': float(df.loc[best_idx, 'metrics/recall']),
                'fitness': float(df.loc[best_idx, 'fitness']),
            }

        return metrics

    def print_metrics_summary(self, metrics):
        """Print metrics summary"""
        if not metrics:
            return

        print("\n📊 Best Model Metrics:")
        print(f"   Best Epoch: {metrics.get('best_epoch', 'N/A')}")
        print(f"   mAP@50: {metrics.get('mAP50', 0):.3f}")
        print(f"   mAP@50-95: {metrics.get('mAP50-95', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")

    def compare_versions(self, version1=None, version2=None):
        """Compare two model versions"""
        if version1 is None:
            version1 = self.versions[-2]['version'] if len(self.versions) >= 2 else None
        if version2 is None:
            version2 = self.versions[-1]['version'] if self.versions else None

        if not version1 or not version2:
            print("❌ Need at least 2 versions to compare")
            return

        v1_info = next(v for v in self.versions if v['version'] == version1)
        v2_info = next(v for v in self.versions if v['version'] == version2)

        print(f"\n📊 Comparing {version1} vs {version2}")
        print("="*50)

        # Compare metrics
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
            v1_val = v1_info['metrics'].get(metric, 0)
            v2_val = v2_info['metrics'].get(metric, 0)
            diff = v2_val - v1_val
            diff_pct = (diff / v1_val * 100) if v1_val > 0 else 0

            symbol = "↗️" if diff > 0 else "↘️" if diff < 0 else "→"
            print(f"{metric:12s}: {v1_val:.3f} → {v2_val:.3f} {symbol} ({diff_pct:+.1f}%)")

        # Compare dataset
        print(f"\nDataset changes:")
        v1_total = v1_info['dataset_stats']['total_annotations']
        v2_total = v2_info['dataset_stats']['total_annotations']
        print(f"Annotations: {v1_total} → {v2_total} ({v2_total-v1_total:+d})")

    def export_for_deployment(self, version=None, format='onnx'):
        """Export model for deployment"""
        if version is None:
            version = self.versions[-1]['version']

        version_info = next(v for v in self.versions if v['version'] == version)
        model_path = Path(version_info['checkpoint_path'])

        print(f"\n📦 Exporting {version} to {format.upper()}...")

        model = YOLO(str(model_path))
        export_path = model.export(format=format, imgsz=640, half=True)

        # Move to organized location
        export_dir = self.versions_dir / version / "exports"
        export_dir.mkdir(exist_ok=True)

        final_path = export_dir / f"model.{format}"
        shutil.move(export_path, final_path)

        print(f"✅ Exported to: {final_path}")
        return final_path

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data',
                        help='Dataset directory')
    parser.add_argument('--compare', action='store_true',
                        help='Compare last two versions')
    parser.add_argument('--export', choices=['onnx', 'tflite', 'coreml'],
                        help='Export model format')
    parser.add_argument('--version', help='Specific version to use')

    args = parser.parse_args()

    trainer = IncrementalTrainer()

    if args.compare:
        trainer.compare_versions()
    elif args.export:
        trainer.export_for_deployment(args.version, args.export)
    else:
        # Determine training strategy
        strategy = trainer.determine_training_strategy(args.dataset)

        # Train if needed
        if strategy['strategy'] != 'skip':
            trainer.train_model(strategy, args.dataset)

            # Auto-compare if we have previous version
            if len(trainer.versions) >= 2:
                trainer.compare_versions()

if __name__ == "__main__":
    main()