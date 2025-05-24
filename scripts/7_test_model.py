# scripts/7_test_model.py

import yaml
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_images(source_path):
    """Load test images from directory"""
    source_path = Path(source_path)

    if source_path.is_file():
        return [source_path]

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(source_path.glob(ext))
        images.extend(source_path.glob(ext.upper()))

    return sorted(images)

def run_inference(model, image_paths, conf_threshold=0.25):
    """Run inference on test images"""
    results = []

    for img_path in image_paths:
        # Run prediction
        pred = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )[0]

        # Extract results
        boxes = pred.boxes
        if boxes is not None:
            detections = []
            for i in range(len(boxes)):
                det = {
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                    'confidence': float(boxes.conf[i].cpu()),
                    'class': int(boxes.cls[i].cpu()),
                    'class_name': model.names[int(boxes.cls[i].cpu())]
                }
                detections.append(det)
        else:
            detections = []

        results.append({
            'image': str(img_path),
            'detections': detections,
            'detection_count': len(detections)
        })

    return results

def analyze_results(results, classes):
    """Analyze detection results"""
    total_images = len(results)
    images_with_detections = sum(1 for r in results if r['detection_count'] > 0)
    total_detections = sum(r['detection_count'] for r in results)

    # Class distribution
    class_counts = {cls: 0 for cls in classes}
    confidence_scores = []

    for result in results:
        for det in result['detections']:
            class_counts[det['class_name']] += 1
            confidence_scores.append(det['confidence'])

    analysis = {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'detection_rate': images_with_detections / total_images if total_images > 0 else 0,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'class_distribution': class_counts,
        'confidence_stats': {
            'mean': np.mean(confidence_scores) if confidence_scores else 0,
            'std': np.std(confidence_scores) if confidence_scores else 0,
            'min': np.min(confidence_scores) if confidence_scores else 0,
            'max': np.max(confidence_scores) if confidence_scores else 0
        }
    }

    return analysis

def create_test_report(results, analysis, output_dir, model_path):
    """Create detailed test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"test_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    with open(report_dir / "raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary report
    report_text = f"""# Model Test Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {model_path}

## Summary Statistics
- Total test images: {analysis['total_images']}
- Images with detections: {analysis['images_with_detections']}
- Detection rate: {analysis['detection_rate']:.2%}
- Total detections: {analysis['total_detections']}
- Average detections per image: {analysis['avg_detections_per_image']:.2f}

## Confidence Statistics
- Mean confidence: {analysis['confidence_stats']['mean']:.3f}
- Std deviation: {analysis['confidence_stats']['std']:.3f}
- Min confidence: {analysis['confidence_stats']['min']:.3f}
- Max confidence: {analysis['confidence_stats']['max']:.3f}

## Class Distribution
"""

    for class_name, count in analysis['class_distribution'].items():
        percentage = (count / analysis['total_detections'] * 100) if analysis['total_detections'] > 0 else 0
        report_text += f"- {class_name}: {count} ({percentage:.1f}%)\n"

    # Low confidence detections
    low_conf_threshold = 0.5
    low_conf_detections = []
    for result in results:
        for det in result['detections']:
            if det['confidence'] < low_conf_threshold:
                low_conf_detections.append({
                    'image': Path(result['image']).name,
                    'class': det['class_name'],
                    'confidence': det['confidence']
                })

    if low_conf_detections:
        report_text += f"\n## Low Confidence Detections (< {low_conf_threshold})\n"
        for det in sorted(low_conf_detections, key=lambda x: x['confidence']):
            report_text += f"- {det['image']}: {det['class']} ({det['confidence']:.3f})\n"

    # Images without detections
    no_detections = [r for r in results if r['detection_count'] == 0]
    if no_detections:
        report_text += f"\n## Images Without Detections ({len(no_detections)})\n"
        for result in no_detections:
            report_text += f"- {Path(result['image']).name}\n"

    with open(report_dir / "report.md", 'w') as f:
        f.write(report_text)

    return report_dir

def create_visualizations(results, analysis, report_dir):
    """Create visualization plots"""

    # Confidence distribution
    confidence_scores = []
    for result in results:
        for det in result['detections']:
            confidence_scores.append(det['confidence'])

    if confidence_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Detection Confidence Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(report_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Class distribution
    class_counts = analysis['class_distribution']
    if any(count > 0 for count in class_counts.values()):
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.bar(classes, counts)
        plt.xlabel('Class')
        plt.ylabel('Detection Count')
        plt.title('Class Distribution in Test Set')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Detection rate by image
    detection_counts = [r['detection_count'] for r in results]
    plt.figure(figsize=(12, 6))
    plt.plot(detection_counts, marker='o', alpha=0.7)
    plt.xlabel('Image Index')
    plt.ylabel('Detection Count')
    plt.title('Detections per Image')
    plt.grid(True, alpha=0.3)
    plt.savefig(report_dir / "detections_per_image.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_annotated_images(model, image_paths, output_dir, conf_threshold=0.25, max_images=50):
    """Save annotated images with detections"""
    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)

    # Limit number of images to avoid too many files
    selected_images = image_paths[:max_images] if len(image_paths) > max_images else image_paths

    for img_path in selected_images:
        # Run prediction with save=True
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            save=True,
            project=str(annotated_dir),
            name='',
            exist_ok=True
        )

    print(f"📸 Saved {len(selected_images)} annotated images to {annotated_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model on new images')
    parser.add_argument('--source', required=True,
                       help='Path to test images directory or single image')
    parser.add_argument('--model', default='models/best.pt',
                       help='Path to model file')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--output', default='test_results',
                       help='Output directory for results')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images with detections')
    parser.add_argument('--max-images', type=int, default=300,
                       help='Maximum images to save (for annotated output)')

    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Load model
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Load config for class names
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Load test images
    print(f"📁 Loading test images from: {args.source}")
    image_paths = load_test_images(args.source)

    if not image_paths:
        print(f"❌ No images found in: {args.source}")
        return

    print(f"🖼️  Found {len(image_paths)} test images")

    # Run inference
    print(f"🔍 Running inference (conf={args.conf})...")
    results = run_inference(model, image_paths, args.conf)

    # Analyze results
    analysis = analyze_results(results, config['classes'])

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    print(f"📊 Generating test report...")
    report_dir = create_test_report(results, analysis, output_dir, model_path)

    # Create visualizations
    create_visualizations(results, analysis, report_dir)

    # Save annotated images if requested
    if args.save_images:
        save_annotated_images(model, image_paths, report_dir, args.conf, args.max_images)

    # Print summary
    print(f"\n📈 Test Results Summary:")
    print(f"   Total images: {analysis['total_images']}")
    print(f"   Detection rate: {analysis['detection_rate']:.2%}")
    print(f"   Average confidence: {analysis['confidence_stats']['mean']:.3f}")
    print(f"   Total detections: {analysis['total_detections']}")

    print(f"\n📁 Full report saved to: {report_dir}")
    print(f"📋 View report: {report_dir / 'report.md'}")

if __name__ == "__main__":
    main()