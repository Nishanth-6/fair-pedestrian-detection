# scripts/audit_height_bias.py

import pandas as pd
import os

# Load detection results
df = pd.read_csv("runs/detect/predict/predictions.csv")

# Only keep persons (your export already includes class_name = person only if you're filtering earlier)
# So this line is needed only if class_name exists
# If it does NOT exist (as in your latest CSV), just use the height column
# Remove this line if class_name column is absent
# df = df[df['class_name'] == 'person'].copy()

# Group by height threshold
df['height_group'] = df['height'].apply(lambda h: 'short' if h < 100 else 'tall')

# Aggregate average confidence
group_stats = df.groupby('height_group')['confidence'].agg(['mean', 'count'])

# Display and save
print("Group-wise average confidence:\n", group_stats)

# Create audit folder if not exists
os.makedirs("runs/detect/audit", exist_ok=True)
group_stats.to_csv("runs/detect/audit/height_confidence_summary.csv", index=True)
