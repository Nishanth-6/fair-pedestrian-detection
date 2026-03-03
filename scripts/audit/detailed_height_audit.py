# scripts/audit/detailed_height_audit.py

import json
import os
import pandas as pd

# 1) Load your enriched predictions (must exist)
pred_path = "runs/detect/audit/enriched_predictions.csv"
pred_df   = pd.read_csv(pred_path)

# 2) Load BDD100K val annotations
ann_path = "dataset/bdd100k/labels/bdd100k_labels_images_val.json"
with open(ann_path) as f:
    val_anns = json.load(f)

# 3) Build a DataFrame of every ground-truth pedestrian with its height
records = []
for img_entry in val_anns:
    img_name = img_entry["name"]
    for lbl in img_entry["labels"]:
        if lbl["category"] == "person":
            y1 = lbl["box2d"]["y1"]
            y2 = lbl["box2d"]["y2"]
            records.append({
                "image": img_name,
                "height": y2 - y1
            })
anno_df = pd.DataFrame(records)
anno_df["height_group"] = anno_df["height"].apply(lambda h: "short" if h < 100 else "tall")

# 4) Aggregate counts per group
gt_counts  = anno_df   .height_group.value_counts().rename("gt_count")
det_counts = pred_df   .height_group.value_counts().rename("det_count")

perf_df = pd.concat([gt_counts, det_counts], axis=1).fillna(0)
perf_df["detection_rate"] = perf_df["det_count"] / perf_df["gt_count"]
perf_df["miss_rate"]      = 1 - perf_df["detection_rate"]

# 5) Save to CSV
out_dir = "runs/detect/audit"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "detection_performance_by_height_group.csv")
perf_df.to_csv(out_path)

# 6) Summary printout
print("✅ Detailed performance by height group:\n")
print(perf_df)
print(f"\nSaved to {out_path}")
