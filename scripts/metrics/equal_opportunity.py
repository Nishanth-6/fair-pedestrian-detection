"""
Compute Equal-Opportunity Difference between short and tall pedestrians.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch  
from ultralytics.utils.metrics import box_iou  # Ultralytics helper

# ------------------------------------------------------------------
GT_DIR   = Path("dataset/bdd100k/BDD100k.v4i.yolov8/valid/labels")
PRED_DIR = Path("runs/detect/predict63/labels")        # <— update if needed
CLASS_ID = 0            # pedestrian
SHORT_THRESHOLD = 100   # px
IOU_THRESH = 0.5
# ------------------------------------------------------------------

hits_short = hits_tall = 0
tot_short  = tot_tall  = 0

for gt_file in GT_DIR.glob("*.txt"):
    IMG_DIR  = GT_DIR.parent / "images"          # …/valid/images
    img_path = IMG_DIR / gt_file.with_suffix(".jpg").name
    img      = Image.open(img_path)
    H, W = img.height, img.width

    # ---- read GT as xyxy (absolute px) ---------------------------
    with open(gt_file) as f:
        gt_boxes = []
        for l in f.read().splitlines():
            cls, xc, yc, w, h = map(float, l.split()[:5])
            if cls != CLASS_ID:
                continue
            x1 = (xc - w/2) * W
            y1 = (yc - h/2) * H
            x2 = (xc + w/2) * W
            y2 = (yc + h/2) * H
            gt_boxes.append([x1, y1, x2, y2])

            # (tot_short if h*H < SHORT_THRESHOLD else tot_tall)  += 1
            if h * H < SHORT_THRESHOLD:
                  tot_short += 1
            else:
                tot_tall  += 1
    if not gt_boxes:
        continue
    gt_boxes = np.array(gt_boxes)

    # ---- read predictions ---------------------------------------
    pred_file = PRED_DIR / gt_file.name
    if not pred_file.exists():
        continue
    with open(pred_file) as f:
        pred_boxes = []
        for l in f.read().splitlines():
            parts = list(map(float, l.split()))
            if len(parts) < 5:       # guard
                continue
            cls, xc, yc, w, h = parts[:5]
            if cls != CLASS_ID:
                continue
            x1 = (xc - w/2) * W
            y1 = (yc - h/2) * H
            x2 = (xc + w/2) * W
            y2 = (yc + h/2) * H
            pred_boxes.append([x1, y1, x2, y2])
    if not pred_boxes:
        continue
    pred_boxes = np.array(pred_boxes)

    # ---- IoU matching (greedy) -----------------------------------
    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
    for p_idx, g_idx in zip(*np.where(ious >= IOU_THRESH)):
        h_gt = (gt_boxes[g_idx][3] - gt_boxes[g_idx][1])  # height in px
        if h_gt < SHORT_THRESHOLD:
            hits_short += 1
        else:
            hits_tall  += 1
        ious[:, g_idx] = 0        # mark GT as matched
        ious[p_idx, :] = 0        # mark pred as used

# ------------------ results ---------------------------------------
tpr_short = hits_short / tot_short if tot_short else 0
tpr_tall  = hits_tall  / tot_tall  if tot_tall  else 0
eod       = abs(tpr_short - tpr_tall)

print(f"✅ TPR Short: {tpr_short:.3f}  |  TPR Tall: {tpr_tall:.3f}")
print(f"📊 Equal Opportunity Difference = {eod:.3f}")
