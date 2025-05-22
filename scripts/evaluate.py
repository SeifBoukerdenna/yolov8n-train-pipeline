# scripts/evaluate.py

import os
from ultralytics import YOLO

def main():
    weights = os.getenv("WEIGHTS_PATH", "models/exp1/weights/best.pt")
    data_yaml = os.getenv("DATA_YAML", "configs/data.yaml")

    model = YOLO(weights)
    results = model.val(data=data_yaml)
    print(results)

if __name__ == "__main__":
    main()
