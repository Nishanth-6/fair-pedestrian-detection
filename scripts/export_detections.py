# scripts/export_detections.py
import os
import csv
import random
from ultralytics import YOLO

image_dir = "dataset/bdd100k/images/100k/val"
sampled_images = random.sample(os.listdir(image_dir), 500)

model = YOLO("yolov8n.pt")  # or trained weights

results = model.predict(
    source=[os.path.join(image_dir, img) for img in sampled_images],
    save=False,
    conf=0.25
)

output_csv = "runs/detect/predict/predictions.csv"
os.makedirs("runs/detect/predict", exist_ok=True)

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "height", "confidence"])

    for img, result in zip(sampled_images, results):
        for box in result.boxes:
            h = box.xywh[0][3].item()
            conf = box.conf.item()
            writer.writerow([img, round(h, 2), round(conf, 3)])

print(f"✅ Detection results saved to {output_csv}")
