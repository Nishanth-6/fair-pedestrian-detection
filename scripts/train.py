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
