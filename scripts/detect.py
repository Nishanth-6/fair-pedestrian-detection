# scripts/detect.py
from ultralytics import YOLO
import os
import random

# Load model
model = YOLO("yolov8n.pt")

# Get list of images and sample 500
image_dir = "dataset/bdd100k/images/100k/val/"
all_images = os.listdir(image_dir)
sampled_images = random.sample(all_images, 500)  # Adjust number here

# Detect on sampled images
results = model.predict(source=[os.path.join(image_dir, img) for img in sampled_images], save=True, conf=0.5)

print(f"Detected {len(sampled_images)} images. Results saved to runs/detect/predict/")