# scripts/plotting/plot_eod.py
import matplotlib.pyplot as plt
vals = {"Short": 0.147, "Tall": 0.901}
plt.bar(vals.keys(), vals.values())
plt.ylim(0, 1)
plt.ylabel("True-Positive Rate")
plt.title("Detection rate by height group")
plt.tight_layout()
plt.savefig("runs/detect/metrics/eod_bar.png", dpi=200)
