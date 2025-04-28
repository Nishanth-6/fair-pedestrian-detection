# scripts/enrich_predictions.py

import pandas as pd
import os

# Input file
input_csv = "runs/detect/predict/predictions.csv"

# Output file
output_csv = "runs/detect/audit/enriched_predictions.csv"

# Load basic predictions
df = pd.read_csv(input_csv)

# Add height group label (short or tall)
def assign_height_group(h):
    return 'short' if h < 100 else 'tall'

df['height_group'] = df['height'].apply(assign_height_group)

# Save enriched dataset
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"✅ Enriched predictions saved to {output_csv}")
