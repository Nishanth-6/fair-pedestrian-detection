# from ultralytics import YOLO
# from pathlib import Path

# model = YOLO("runs/detect/mitigation/height_weighted_train/weights/best.pt")

# results = model.predict(
#     source="dataset/bdd100k/BDD100k.v4i.yolov8/valid/images",
#     save=True,
#     save_txt=True,
#     save_conf=True,
#     conf=0.25
# )

# print("✅ Predictions saved to:", results[0].save_dir)

#!/usr/bin/env python3
#!/usr/bin/env python3
from pathlib import Path
from ultralytics import YOLO

DATA_ROOT   = Path("dataset/bdd100k/BDD100k.v4i.yolov8")
VAL_IMG_DIR = DATA_ROOT / "valid/images"
WEIGHTS     = Path("runs/detect/mitigation/height_weighted_train/weights/best.pt")
OUT         = "runs/detect/predict6"          # or predict7, …
PERSON_ID = 0                            # check data.yaml !

model = YOLO(str(WEIGHTS))

results = model.predict(
        source=str(VAL_IMG_DIR),
        save=True,
        save_txt=True,
        conf=0.25,
        classes=[PERSON_ID],      # now 0
        project="runs/detect",
        name="predict63")         # new folder
print(f"✅ Predictions saved to: {results[0].save_dir}")
