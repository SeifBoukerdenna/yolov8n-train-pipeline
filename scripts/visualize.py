from ultralytics import YOLO
from pathlib import Path

# 1) Resolve paths relative to repo root
ROOT    = Path(__file__).parent.parent
WEIGHTS = ROOT / 'models' / 'exp13' / 'weights' / 'best.pt'
SOURCE  = ROOT / 'data'   / 'images' / 'val'

# 2) Load your trained model
model = YOLO(str(WEIGHTS))

# 3) Run prediction, explicitly setting project/name so it writes under pipeline/runs/
results = model.predict(
    source=str(SOURCE),                # folder (or file/glob) to run on
    conf=0.25,                         # confidence threshold
    save=True,                         # write images with boxes
    save_txt=False,                    # disable writing .txt preds
    project=str(ROOT / 'runs'),        # base output dir
    name='detect_predict'              # subfolder under runs/
)

# 4) Print out where the visuals were saved
save_dir = results[0].save_dir      # Path to the actual save directory
print(f"Annotated images saved to: {save_dir}")
