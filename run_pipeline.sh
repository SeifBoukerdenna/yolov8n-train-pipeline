#!/usr/bin/env bash
# Convenience script to run pipeline

case "${1:-help}" in
    full)
        echo "🚀 Running full pipeline..."
        python scripts/pipeline.py "${@:2}"
        ;;
    continue)
        echo "⏭️ Continuing pipeline after labeling..."
        python scripts/pipeline.py --continue "${@:2}"
        ;;
    sanitize)
        echo "🧹 Sanitizing dataset..."
        python scripts/sanitize.py "${@:2}"
        ;;
    split)
        echo "📊 Splitting dataset into train/val..."
        python scripts/5_split_dataset.py "${@:2}"
        ;;
    train)
        echo "🏋️ Running training only..."
        python scripts/6_train_model.py train "${@:2}"
        ;;
    test)
        echo "🧪 Testing model..."
        python scripts/7_test_model.py "${@:2}"
        ;;
    validate)
        echo "🔍 Validating model..."
        python scripts/6_train_model.py validate "${@:2}"
        ;;
    detect)
        echo "🔎 Running detection..."
        python scripts/6_train_model.py detect "${@:2}"
        ;;
    quick)
        echo "⚡ Quick command..."
        python quick_commands.py "${@:2}"
        ;;
    *)
        echo "🚀 YOLOv8 Training Pipeline"
        echo ""
        echo "Usage: ./run_pipeline.sh <command> [options]"
        echo ""
        echo "Main Commands:"
        echo "  full      - Run complete pipeline (extract→upload→import)"
        echo "  continue  - Continue after labeling (export→split→train)"
        echo "  test      - Test trained model on new images"
        echo ""
        echo "Individual Steps:"
        echo "  sanitize  - Clean empty label pairs"
        echo "  split     - Split dataset into train/val"
        echo "  train     - Train model"
        echo "  validate  - Validate model performance"
        echo "  detect    - Run detection on images"
        echo ""
        echo "Quick Commands:"
        echo "  quick extract --skip 10 --random"
        echo "  quick test data/new_images"
        echo "  quick status"
        echo ""
        echo "Examples:"
        echo "  ./run_pipeline.sh full --frame-skip 15 --randomize"
        echo "  ./run_pipeline.sh continue --train-ratio 0.9"
        echo "  ./run_pipeline.sh test --source data/test_images --save-images"
        echo "  ./run_pipeline.sh quick status"
        ;;
esac