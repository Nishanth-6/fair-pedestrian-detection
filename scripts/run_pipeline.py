# scripts/run_pipeline.py
import os

print("▶️ Starting Training...")
os.system("python scripts/train.py")

print("▶️ Running Detection...")
os.system("python scripts/detect.py")

print("▶️ Auditing Fairness...")
os.system("python scripts/audit.py")

print("✅ Pipeline completed.")
