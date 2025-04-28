# scripts/audit.py
import os
import random
from ultralytics import YOLO

image_dir = "dataset/bdd100k/images/100k/val"
sampled_images = random.sample(os.listdir(image_dir), 500)

model = YOLO("yolov8n.pt")  # or path to your trained weights

short_total = 0
short_detected = 0
tall_total = 0
tall_detected = 0

for img in sampled_images:
    results = model(os.path.join(image_dir, img))
    for box in results[0].boxes:
        height = box.xywh[0][3].item()
        conf = box.conf.item()
        if height < 140:
            short_total += 1
            if conf >= 0.5:
                short_detected += 1
        elif height >= 150:
            tall_total += 1
            if conf >= 0.5:
                tall_detected += 1

hdr = (tall_detected / tall_total) / (short_detected / short_total) if short_detected > 0 and tall_detected > 0 else 0

print(f"\n📊 Audit Results on 500 Images:")
print(f"  Short pedestrians detected: {short_detected}/{short_total}")
print(f"  Tall pedestrians detected: {tall_detected}/{tall_total}")
print(f"  HDR (Height Disparity Ratio): {hdr:.2f}")
