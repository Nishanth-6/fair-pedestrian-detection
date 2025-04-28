# scripts/export_detections.py

import os
import pandas as pd
import csv
import random
from ultralytics import YOLO

# Sample 500 images from the validation set
image_dir = "dataset/bdd100k/images/100k/val"
sampled_images = random.sample(os.listdir(image_dir), 500)

# Load the YOLOv8 model (pretrained or your own weights)
model = YOLO("yolov8n.pt")  # Replace with trained weights if available

# Run detection
results = model.predict(
    source=[os.path.join(image_dir, img) for img in sampled_images],
    save=False,
    conf=0.25
)

# Output CSV path
output_csv = "runs/detect/predict/predictions.csv"
os.makedirs("runs/detect/predict", exist_ok=True)

# Write detections to CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "height", "confidence", "class_name"])

    for img, result in zip(sampled_images, results):
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls = int(box.cls.item())  # Class ID
            label = model.names[cls]   # Convert class ID to label name
            height = xyxy[3] - xyxy[1]  # ymax - ymin

            writer.writerow([
                img,
                round(xyxy[0], 2),  # xmin
                round(xyxy[1], 2),  # ymin
                round(xyxy[2], 2),  # xmax
                round(xyxy[3], 2),  # ymax
                round(height, 2),
                round(conf, 3),
                label
            ])

print(f"✅ Detection results saved to {output_csv}")
