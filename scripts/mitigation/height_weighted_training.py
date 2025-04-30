#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
import yaml
from ultralytics import YOLO

# --- CONFIGURATION ---
TRAIN_IMG_DIR = Path("dataset/bdd100k/images/100k/train")
TRAIN_LABEL_DIR = Path("dataset/bdd100k/labels/train") 
VAL_IMG_DIR = Path("dataset/bdd100k/images/100k/val")

# Where to write the weighted image list and YAML
WEIGHTED_TXT = Path("dataset/bdd100k/height_weighted_train.txt")
WEIGHTED_YAML = Path("dataset/bdd100k/bdd100k_height_weighted.yaml")

# Threshold in pixels below which a box is “short”
SHORT_THRESHOLD_PX = 100

# --- STEP 1: Partition images by presence of a short bbox ---
all_imgs = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
short_imgs, tall_imgs = [], []

for img_path in all_imgs:
    lbl_path = TRAIN_LABEL_DIR / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        continue
    img_h = Image.open(img_path).height
    with open(lbl_path) as f:
        lines = [l.split() for l in f.read().splitlines()]
    # YOLO label format: class x_center y_center width height
    has_short = any(
        (float(parts[4]) * img_h) < SHORT_THRESHOLD_PX
        for parts in lines
        if len(parts) == 5
    )
    (short_imgs if has_short else tall_imgs).append(img_path)

print(
    f"Found {len(short_imgs)} short-pedestrian images, {len(tall_imgs)} tall-only images."
)

# --- STEP 2: Write weighted train list (short twice, tall once) ---
os.makedirs(WEIGHTED_TXT.parent, exist_ok=True)
with open(WEIGHTED_TXT, "w") as f:
    for p in short_imgs:
        abs_path = str(p.resolve())  # ← absolute
        f.write(abs_path + "\n")
        f.write(abs_path + "\n")  # duplicate
    for p in tall_imgs:
        f.write(str(p.resolve()) + "\n")  # absolute
print(f"✅ Weighted image list saved to {WEIGHTED_TXT}")

# --- STEP 3: Write a new data YAML pointing to this list ---
data = {
    "train": str(WEIGHTED_TXT.resolve()),  # absolute
    "val": str(VAL_IMG_DIR.resolve()),  # absolute
    "nc": 1,
    "names": ["pedestrian"],
}
with open(WEIGHTED_YAML, "w") as f:
    yaml.dump(data, f)
print(f"✅ Height-weighted YAML saved to {WEIGHTED_YAML}")

# --- STEP 4: Launch YOLOv8 training on the weighted dataset ---
model = YOLO("yolov8n.pt")  # or your finetuned checkpoint

model.train(
    data=str(WEIGHTED_YAML),
    epochs=5,
    imgsz=320,
    batch=8,
    classes=[0],
    project="runs/detect/mitigation",
    name="height_weighted_train",
)

print("✅ Height-weighted training complete.")
