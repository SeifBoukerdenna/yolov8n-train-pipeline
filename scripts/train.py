#!/usr/bin/env python3
import yaml
from ultralytics import YOLO
from pathlib import Path

# 1) load train.yaml
cfg_file = Path(__file__).parent.parent / 'configs' / 'train.yaml'
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

# 2) extract params
data      = cfg['data']
model_cfg = cfg['cfg']
epochs    = cfg['epochs']
imgsz     = cfg['imgsz']
batch     = cfg['batch_size']
project   = cfg['project']
run_name  = cfg['run_name']
resume    = cfg.get('resume', False)

# 3) launch training
model = YOLO(model_cfg)
model.train(
    data=data,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    project=project,
    name=run_name,
    resume=resume
)
