"""
visualize_results.py
Drop next to your project root (same level as the 'runs' folder and epoch_log.csv).
Usage:     py -3.12 visualize_results.py
Outputs:
  - epoch_metrics.jpg       (all per-epoch training curves)
  - dataset_analysis.jpg    (split distribution + per-class final metrics)
"""

import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── AUTO-DETECT paths ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# epoch_log.csv lives next to the script
CSV  = os.path.join(BASE, "epoch_log.csv")

# find latest train* run (for any future use / consistency with make_grid.py)
train_dirs = sorted(glob.glob(os.path.join(BASE, "runs", "detect", "train*")))
if train_dirs:
    RUN_DIR = train_dirs[-1]
    print(f"  Using run folder : {RUN_DIR}")
else:
    RUN_DIR = None
    print("  Warning: no train* folder found under runs/detect — continuing without it.")

OUT1 = os.path.join(BASE, "epoch_metrics.jpg")
OUT2 = os.path.join(BASE, "dataset_analysis.jpg")
print(f"  CSV              : {CSV}")
print(f"  Output 1         : {OUT1}")
print(f"  Output 2         : {OUT2}\n")
# ─────────────────────────────────────────────────────────────────────────────

DPI    = 300
BG     = "#0A0E1A"
PANEL  = "#111827"
CYAN   = "#22D3EE"
GREEN  = "#34D399"
YELLOW = "#FBBF24"
RED    = "#F87171"
PURPLE = "#A78BFA"
ORANGE = "#FB923C"
WHITE  = "#F1F5F9"
MUTED  = "#64748B"
GRID   = "#1E293B"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   WHITE,
    "axes.titlecolor":   WHITE,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        WHITE,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID,
})

# ── Load CSV ──────────────────────────────────────────────────────────────────
if not os.path.exists(CSV):
    raise FileNotFoundError(
        f"epoch_log.csv not found at {CSV}\n"
        "Make sure the script sits in the same folder as epoch_log.csv."
    )

epochs, precision, recall, map50, map5095, gpu_util, vram, epoch_s = ([] for _ in range(8))

with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        e = int(row["epoch"])
        if e == 0:
            continue
        epochs.append(e)
        precision.append(float(row["precision"]))
        recall.append(float(row["recall"]))
        map50.append(float(row["mAP50"]))
        map5095.append(float(row["mAP50-95"]))
        gpu_util.append(float(row["gpu_util_%"]))
        vram.append(float(row["vram_used_gb"]))
        epoch_s.append(float(row["epoch_s"]))

best_epoch = epochs[map50.index(max(map50))]
print(f"  Epochs loaded: {len(epochs)}  |  Best mAP50 epoch: {best_epoch}")

# ── Figure 1: per-epoch training curves ───────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(22, 13), dpi=DPI)
fig1.patch.set_facecolor(BG)
fig1.suptitle("Training Metrics per Epoch  -  YOLOv8l Campus Safety Detection",
              fontsize=17, fontweight="bold", color=WHITE, y=0.98)

def vline(ax, epoch, label):
    ax.axvline(x=epoch, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(epoch + 0.4, ax.get_ylim()[1] * 0.97, label, color=YELLOW, fontsize=8.5, va="top")

def style(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both", alpha=0.5)
    ax.set_xlim(1, max(epochs))
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top","right"]].set_visible(False)
    vline(ax, best_epoch, f"best\nep{best_epoch}")

ax = axes[0, 0]
ax.plot(epochs, map50,   color=CYAN,  linewidth=2.2, label="mAP@0.5")
ax.plot(epochs, map5095, color=GREEN, linewidth=2.2, label="mAP@0.5:0.95", linestyle="--")
ax.fill_between(epochs, map50,   alpha=0.12, color=CYAN)
ax.fill_between(epochs, map5095, alpha=0.12, color=GREEN)
ax.legend()
style(ax, "mAP (Validation)", "mAP Score", (0, 0.65))

ax = axes[0, 1]
ax.plot(epochs, precision, color=PURPLE, linewidth=2.2, label="Precision")
ax.plot(epochs, recall,    color=ORANGE, linewidth=2.2, label="Recall", linestyle="--")
ax.fill_between(epochs, precision, alpha=0.12, color=PURPLE)
ax.fill_between(epochs, recall,    alpha=0.12, color=ORANGE)
ax.legend()
style(ax, "Precision & Recall", "Score", (0, 1.0))

f1 = [2*p*r/(p+r+1e-9) for p, r in zip(precision, recall)]
ax = axes[0, 2]
ax.plot(epochs, f1, color=RED, linewidth=2.2, label="F1 Score")
ax.fill_between(epochs, f1, alpha=0.15, color=RED)
best_f1    = max(f1)
best_f1_ep = epochs[f1.index(best_f1)]
ax.scatter([best_f1_ep], [best_f1], color=YELLOW, s=60, zorder=5)
ax.annotate(f"  peak {best_f1:.3f}", xy=(best_f1_ep, best_f1), color=YELLOW, fontsize=9)
ax.legend()
style(ax, "F1 Score (Computed)", "F1", (0, 1.0))

ax = axes[1, 0]
ax.bar(epochs, gpu_util, color=CYAN, alpha=0.7, width=0.7)
ax.axhline(np.mean(gpu_util), color=YELLOW, linewidth=1.5, linestyle="--",
           label=f"mean {np.mean(gpu_util):.1f}%")
ax.legend()
style(ax, "GPU Utilisation (%)", "GPU %", (0, 20))

ax = axes[1, 1]
ax.plot(epochs, vram, color=GREEN, linewidth=2.0, label="VRAM Used (GB)")
ax.fill_between(epochs, vram, alpha=0.15, color=GREEN)
ax.axhline(8.6, color=RED, linewidth=1.2, linestyle=":", label="Total VRAM 8.6 GB")
ax.legend()
style(ax, "VRAM Usage (GB)", "GB", (0, 10))

ax = axes[1, 2]
ax.plot(epochs, epoch_s, color=ORANGE, linewidth=1.8, label="Epoch time (s)")
ax.fill_between(epochs, epoch_s, alpha=0.12, color=ORANGE)
ax.axhline(np.mean(epoch_s), color=YELLOW, linewidth=1.3, linestyle="--",
           label=f"mean {np.mean(epoch_s):.1f}s")
ax.legend()
style(ax, "Epoch Duration (seconds)", "Seconds")

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(OUT1, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print(f"  Saved -> {OUT1}")

# ── Figure 2: dataset analysis & per-class metrics ────────────────────────────
CLASSES = [
    "Fire alarm", "Left Exit", "Left/Right Exit", "Right Exit",
    "Straight Exit", "blocked emerg. exit", "boxes blocking exit",
    "emerg. exit door", "exit block violation", "exit-sign",
    "head", "helmet", "objects blocked exit", "person", "sign",
]

TRAIN  = [304, 112, 157, 239, 122,  15,  10,  10,   2,  49, 365, 1242,  14,  16, 156]
VALID  = [ 78,  38,  50,  58,  35,   4,   3,   1,   1,  10, 108,  403,   3,   4,  36]
TEST   = [ 42,  14,  21,  33,  23,   1,   0,   3,   2,   4,  70,  223,   1,   3,  20]

PREC   = [0.494, 0.325, 0.320, 0.296, 0.466, 1.000, 0.368, 0.000, 0.000, 0.824,
          0.585, 0.679, 0.388, 0.000, 0.866]
REC    = [0.923, 0.868, 0.920, 0.897, 0.550, 0.000, 0.333, 0.000, 0.000, 1.000,
          0.898, 0.926, 0.667, 0.000, 1.000]
AP50   = [0.849, 0.455, 0.381, 0.400, 0.396, 0.202, 0.353, 0.012, 0.018, 0.995,
          0.823, 0.939, 0.266, 0.000, 0.995]

y    = np.arange(len(CLASSES))
fig2 = plt.figure(figsize=(26, 20), dpi=DPI, facecolor=BG)
fig2.suptitle("Dataset Analysis & Per-Class Final Results  -  YOLOv8l Campus Safety",
              fontsize=17, fontweight="bold", color=WHITE, y=0.99)

ax1   = fig2.add_subplot(3, 2, (1, 2))
bar_h = 0.6
ax1.barh(y, TRAIN, bar_h, label="Train", color=CYAN,   alpha=0.85)
ax1.barh(y, VALID,  bar_h, left=TRAIN,   label="Valid", color=GREEN,  alpha=0.85)
ax1.barh(y, TEST,   bar_h, left=[t+v for t, v in zip(TRAIN, VALID)],
         label="Test", color=ORANGE, alpha=0.85)
ax1.set_yticks(y)
ax1.set_yticklabels(CLASSES, fontsize=10)
ax1.set_xlabel("Number of Instances")
ax1.set_title("Class Distribution Across Train / Valid / Test Splits", fontweight="bold", pad=10)
ax1.grid(True, axis="x", alpha=0.4)
ax1.spines[["top","right"]].set_visible(False)
ax1.legend(loc="lower right", fontsize=11)
for i, (t, v, te) in enumerate(zip(TRAIN, VALID, TEST)):
    total = t + v + te
    ax1.text(total + 5, i, str(total), va="center", color=MUTED, fontsize=8.5)
split_text = ("Total: 2,159 images\nTrain: 1,386  (64.2%)\n"
              "Valid:    396  (18.3%)\nTest:     377  (17.5%)")
ax1.text(0.98, 0.05, split_text, transform=ax1.transAxes, fontsize=10,
         va="bottom", ha="right", color=WHITE,
         bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=GRID))

ax2       = fig2.add_subplot(3, 2, 3)
colors_ap = [GREEN if v >= 0.8 else YELLOW if v >= 0.4 else RED for v in AP50]
ax2.barh(y, AP50, bar_h, color=colors_ap, alpha=0.85)
ax2.set_yticks(y);  ax2.set_yticklabels(CLASSES, fontsize=9)
ax2.set_xlabel("AP @ IoU 0.5")
ax2.set_title("Per-Class AP@0.5  (Best Model)", fontweight="bold", pad=8)
ax2.set_xlim(0, 1.1)
ax2.axvline(0.5, color=MUTED, linewidth=1, linestyle=":")
ax2.grid(True, axis="x", alpha=0.4)
ax2.spines[["top","right"]].set_visible(False)
for i, v in enumerate(AP50):
    ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5,
             color=WHITE if v > 0.05 else MUTED)
ax2.legend(handles=[
    mpatches.Patch(color=GREEN,  label=">= 0.8  Strong"),
    mpatches.Patch(color=YELLOW, label="0.4-0.8  Mid"),
    mpatches.Patch(color=RED,    label="< 0.4  Weak"),
], fontsize=9, loc="lower right")

ax3       = fig2.add_subplot(3, 2, 4)
colors_p  = [GREEN if v >= 0.7 else YELLOW if v >= 0.4 else RED for v in PREC]
ax3.barh(y, PREC, bar_h, color=colors_p, alpha=0.85)
ax3.set_yticks(y);  ax3.set_yticklabels(CLASSES, fontsize=9)
ax3.set_xlabel("Precision")
ax3.set_title("Per-Class Precision  (Best Model)", fontweight="bold", pad=8)
ax3.set_xlim(0, 1.1)
ax3.axvline(0.5, color=MUTED, linewidth=1, linestyle=":")
ax3.grid(True, axis="x", alpha=0.4)
ax3.spines[["top","right"]].set_visible(False)
for i, v in enumerate(PREC):
    ax3.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=WHITE)

ax4       = fig2.add_subplot(3, 2, 5)
colors_r  = [GREEN if v >= 0.7 else YELLOW if v >= 0.4 else RED for v in REC]
ax4.barh(y, REC, bar_h, color=colors_r, alpha=0.85)
ax4.set_yticks(y);  ax4.set_yticklabels(CLASSES, fontsize=9)
ax4.set_xlabel("Recall")
ax4.set_title("Per-Class Recall  (Best Model)", fontweight="bold", pad=8)
ax4.set_xlim(0, 1.1)
ax4.axvline(0.5, color=MUTED, linewidth=1, linestyle=":")
ax4.grid(True, axis="x", alpha=0.4)
ax4.spines[["top","right"]].set_visible(False)
for i, v in enumerate(REC):
    ax4.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=WHITE)

ax5 = fig2.add_subplot(3, 2, 6)
ax5.set_facecolor(PANEL)
ax5.axis("off")
summary_data = [
    ["Metric",               "Value"],
    ["Overall mAP@0.5",      "0.472"],
    ["Overall mAP@0.5:0.95", "0.300"],
    ["Overall Precision",    "0.441"],
    ["Overall Recall",       "0.599"],
    ["Best Epoch",           str(best_epoch)],
    ["Total Epochs Run",     str(max(epochs))],
    ["Train Images",         "1,386"],
    ["Valid Images",         "396"],
    ["Test Images",          "377"],
    ["Total Classes",        "15"],
    ["Batch Size",           "16"],
    ["Image Size",           "640x640"],
    ["Optimizer",            "AdamW"],
    ["GPU",                  "RTX 3070 Ti 8GB"],
]
tbl = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.65)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if r % 2 == 0 else "#0F172A")
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=WHITE if r > 0 else CYAN,
                        fontweight="bold" if r == 0 else "normal")
ax5.set_title("Training Configuration & Summary", fontweight="bold",
              color=WHITE, fontsize=13, pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig2.savefig(OUT2, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"  Saved -> {OUT2}")
print("\nDone.")
