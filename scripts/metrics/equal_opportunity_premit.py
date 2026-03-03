import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou

# === Configuration ===
GT_DIR   = Path("dataset/bdd100k/BDD100k.v4i.yolov8/valid/labels")
PRED_DIR = Path("runs/detect/predict_nomit/labels")  # ← UPDATE if different
SHORT_THRESHOLD = 100   # px
IOU_THRESH = 0.5
CLASS_ID = 0            # pedestrian

hits_short = hits_tall = 0
tot_short  = tot_tall  = 0

for gt_file in GT_DIR.glob("*.txt"):
    image_file = Path("dataset/bdd100k/BDD100k.v4i.yolov8/valid/images") / gt_file.name.replace(".txt", ".jpg")
    if not image_file.exists():
        continue
    img = Image.open(image_file)
    H, W = img.height, img.width

    gt_boxes = []
    for line in gt_file.read_text().splitlines():
        parts = list(map(float, line.split()))
        if parts[0] != CLASS_ID:
            continue
        x1 = (parts[1] - parts[3]/2) * W
        y1 = (parts[2] - parts[4]/2) * H
        x2 = (parts[1] + parts[3]/2) * W
        y2 = (parts[2] + parts[4]/2) * H
        gt_boxes.append([x1, y1, x2, y2])
        if parts[4] * H < SHORT_THRESHOLD:
            tot_short += 1
        else:
            tot_tall += 1

    if not gt_boxes:
        continue
    gt_boxes = torch.tensor(gt_boxes)

    pred_file = PRED_DIR / gt_file.name
    if not pred_file.exists():
        continue
    pred_boxes = []
    for line in pred_file.read_text().splitlines():
        parts = list(map(float, line.split()))
        if len(parts) < 6 or parts[0] != CLASS_ID:
            continue
        x1 = (parts[1] - parts[3]/2) * W
        y1 = (parts[2] - parts[4]/2) * H
        x2 = (parts[1] + parts[3]/2) * W
        y2 = (parts[2] + parts[4]/2) * H
        pred_boxes.append([x1, y1, x2, y2])
    if not pred_boxes:
        continue
    pred_boxes = torch.tensor(pred_boxes)

    ious = box_iou(pred_boxes, gt_boxes).numpy()
    for p_idx, g_idx in zip(*np.where(ious >= IOU_THRESH)):
        h_gt = gt_boxes[g_idx][3] - gt_boxes[g_idx][1]
        if h_gt < SHORT_THRESHOLD:
            hits_short += 1
        else:
            hits_tall += 1
        ious[:, g_idx] = 0
        ious[p_idx, :] = 0

# === Final Results ===
tpr_short = hits_short / tot_short if tot_short else 0
tpr_tall  = hits_tall  / tot_tall if tot_tall else 0
eod       = abs(tpr_short - tpr_tall)

print(f"✅ [PRE-MITIGATION] TPR Short: {tpr_short:.3f}  |  TPR Tall: {tpr_tall:.3f}")
print(f"📊 [PRE-MITIGATION] Equal Opportunity Difference = {eod:.3f}")
