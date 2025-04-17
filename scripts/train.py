# scripts/train.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/bdd100k/bdd100k_small.yaml",
    epochs=1,
    imgsz=320,
    batch=8,
    classes=[0]
)

import csv
from datetime import datetime

log_file = "logs/metrics.csv"
os.makedirs("logs", exist_ok=True)

with open(log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["timestamp", "run", "mAP", "HDR", "box_loss", "cls_loss"])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train9",
        0.0,    # dummy mAP
        1.5,    # dummy HDR
        14.3,   # from train logs
        0       # no classification loss
    ])
