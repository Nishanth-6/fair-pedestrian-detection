#!/usr/bin/env python3
"""
Binary search to find optimal weight multiplier for short pedestrians.
Objective: Minimize Equal Opportunity Difference (EOD).
"""

import os
from pathlib import Path
import subprocess
import shutil

# --- Settings -----------------------------------------------------
START = 1.0
END = 4.0
MAX_TRIALS = 4
EOD_THRESHOLD = 0.01
TRAIN_SCRIPT = "scripts/mitigation/height_weighted_training.py"
PRED_SCRIPT = "scripts/mitigation/run_predictions.py"
EVAL_SCRIPT = "scripts/metrics/equal_opportunity.py"
TXT_FILE = Path("dataset/bdd100k/height_weighted_train.txt")
WEIGHTED_YAML = Path("dataset/bdd100k/bdd100k_height_weighted.yaml")
# ------------------------------------------------------------------

def write_weighted_txt(short_imgs, tall_imgs, weight):
    with open(TXT_FILE, "w") as f:
        for p in short_imgs:
            abs_path = str(p.resolve())
            for _ in range(int(weight)):
                f.write(abs_path + "\n")
        for p in tall_imgs:
            f.write(str(p.resolve()) + "\n")

def read_split():
    short, tall = [], []
    seen = set()  # ← FIX ADDED HERE
    for line in TXT_FILE.read_text().splitlines():
        path = Path(line)
        if path.name not in seen:
            seen.add(path.name)
            short.append(path)
        else:
            tall.append(path)
    return short, tall


def get_eod():
    result = subprocess.run(
        ["python3", EVAL_SCRIPT],
        stdout=subprocess.PIPE,
        text=True
    )
    for line in result.stdout.splitlines():
        if "Equal Opportunity Difference" in line:
            return float(line.split("=")[-1].strip())
    return float("inf")

def train_and_eval(weight):
    print(f"\n🔁 Trying weight = {weight:.2f}")
    short_imgs, tall_imgs = read_split()
    write_weighted_txt(short_imgs, tall_imgs, weight)

    subprocess.run(["python3", TRAIN_SCRIPT])
    subprocess.run(["python3", PRED_SCRIPT])

    return get_eod()

# --- Binary Search Logic ------------------------------------------
low, high = START, END
best_eod = float("inf")
best_w = None

for _ in range(MAX_TRIALS):
    mid = (low + high) / 2
    eod = train_and_eval(mid)
    print(f"🧪 Weight {mid:.2f} → EOD = {eod:.3f}")

    if eod < best_eod:
        best_eod = eod
        best_w = mid

    if eod < EOD_THRESHOLD:
        break

    if eod > 0:  # short TPR too low
        low = mid
    else:        # tall TPR too low
        high = mid

print(f"\n✅ Optimal weight = {best_w:.2f}, with EOD = {best_eod:.3f}")
