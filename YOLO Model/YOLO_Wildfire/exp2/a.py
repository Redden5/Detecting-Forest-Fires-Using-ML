import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
# Point this at your results.csv (absolute or relative).
# If you leave it as None, it will try "results.csv" in the current working dir.
RESULTS_CSV = "C:\\Users\\redde\\VS Code Projects\\Summer Research\\YOLO Model\\YOLO_Wildfire\\exp2\\results.csv"
# ==============

# ---- style to match RF/XGB ----
FONT_LABEL = 14
FONT_TITLE = 16
FONT_LEGEND = 12
FIGSIZE = (6, 4)

# Resolve paths
if RESULTS_CSV is None:
    results_path = Path("results.csv").resolve()
else:
    results_path = Path(RESULTS_CSV).expanduser().resolve()

if not results_path.exists():
    raise FileNotFoundError(f"results.csv not found at: {results_path}")

# Save next to results.csv, under styled_plots/
OUT_DIR = results_path.parent / "styled_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading:  {results_path}")
print(f"Saving to: {OUT_DIR}")

# --- load results.csv ---
df = pd.read_csv(results_path)

# Try common Ultralytics column names (covers metrics/precision(B) style too)
def pick_col(df, keys):
    for k in keys:
        matches = [c for c in df.columns if k in c]
        if matches:
            return matches[0]
    return None

p_col = pick_col(df, ["metrics/precision", "precision"])
r_col = pick_col(df, ["metrics/recall", "recall"])
if p_col is None or r_col is None:
    # Show headers to help adjust quickly
    print("Columns in results.csv:\n", list(df.columns))
    raise ValueError("Could not find precision/recall columns in results.csv.")

prec = pd.to_numeric(df[p_col], errors="coerce").fillna(0).to_numpy()
rec  = pd.to_numeric(df[r_col], errors="coerce").fillna(0).to_numpy()
epochs = np.arange(len(df))

# F1 per epoch (protect against 0/0)
f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

def _plot_epoch_metric(y, name, filename):
    plt.figure(figsize=FIGSIZE)
    plt.plot(epochs, y, linewidth=2, label=name)
    plt.xlabel("Epoch", fontsize=FONT_LABEL)
    plt.ylabel(name, fontsize=FONT_LABEL)
    plt.ylim(0, 1)
    plt.title(f"YOLO — {name} per Epoch", fontsize=FONT_TITLE)
    # legend inside plot
    plt.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    outpath = OUT_DIR / filename
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved:   {outpath}")

# Save three PNGs
_plot_epoch_metric(prec, "Precision", "yolo_precision_per_epoch.png")
_plot_epoch_metric(rec,  "Recall",    "yolo_recall_per_epoch.png")
_plot_epoch_metric(f1,   "F1-score",  "yolo_f1_per_epoch.png")
