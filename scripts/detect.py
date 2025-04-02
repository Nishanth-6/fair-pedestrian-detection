from ultralytics import YOLO

# Use a direct URL to bypass local image issues
model = YOLO("yolov8n.pt")
results = model.predict(source="https://ultralytics.com/images/bus.jpg")

# Save results
results[0].save("result.jpg")
print("Detection saved to result.jpg!")