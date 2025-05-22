# scripts/train.py

import os
from ultralytics import YOLO

def main():
    # read hyperparams from env or defaults
    data_yaml = os.getenv("DATA_YAML", "configs/data.yaml")
    model_cfg = os.getenv("MODEL_CFG", "configs/model.yaml")
    epochs    = int(os.getenv("EPOCHS", 50))
    imgsz     = int(os.getenv("IMG_SIZE", 640))
    batch     = int(os.getenv("BATCH_SIZE", 16))
    project   = os.getenv("PROJECT_DIR", "models")
    run_name  = os.getenv("RUN_NAME", "exp1")

    model = YOLO(model_cfg)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name
    )

if __name__ == "__main__":
    main()
