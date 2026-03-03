# scripts/plotting/height_bias_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_confidence_by_height_group(csv_path, save_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    for group in df['height_group'].unique():
        subset = df[df['height_group'] == group]
        plt.hist(subset['confidence'], bins=20, alpha=0.5, label=group)

    plt.title('Confidence Distribution by Height Group')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confidence_distribution_by_height_group.png'))
    plt.close()
