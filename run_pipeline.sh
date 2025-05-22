#!/usr/bin/env bash
# Convenience script to run pipeline

case "${1:-full}" in
    watch)
        echo "👀 Starting continuous watch mode..."
        python scripts/pipeline.py --mode continuous --watch
        ;;
    train)
        echo "🏋️ Running training only..."
        python scripts/incremental_train.py
        ;;
    validate)
        echo "🔍 Validating annotations..."
        python scripts/validate_annotations.py
        ;;
    export)
        echo "📦 Exporting model..."
        python scripts/incremental_train.py --export "${2:-onnx}"
        ;;
    compare)
        echo "📊 Comparing model versions..."
        python scripts/incremental_train.py --compare
        ;;
    *)
        echo "🚀 Running full pipeline..."
        python scripts/pipeline.py --mode full
        ;;
esac
