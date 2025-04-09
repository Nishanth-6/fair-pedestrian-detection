# scripts/audit.py
import os
import random
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
short_missed = 0
short_total = 0

# Analyze 500 random images
image_dir = "dataset/bdd100k/images/100k/val/"
all_images = os.listdir(image_dir)
sampled_images = random.sample(all_images, 500)  # Same subset as detect.py

for img in sampled_images:
    results = model(f"{image_dir}/{img}")
    for box in results[0].boxes:
        h = box.xywh[0][3].item()
        if h < 100:
            short_total += 1
            if box.conf.item() < 0.5:
                short_missed += 1

print(f"Audited {len(sampled_images)} images. Short Pedestrian FNR: {short_missed/short_total:.2f}")