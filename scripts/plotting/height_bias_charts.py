import matplotlib.pyplot as plt

comparison_df.plot(kind='bar', figsize=(7, 5), rot=0)
plt.title("Detection Count vs Expected Label Count (Short vs Tall)")
plt.ylabel("Count")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("runs/detect/audit/detection_label_comparison.png")
plt.show()
