"""
figS_cv_boxplots.py
===================
SI Figure — per-fold CV stability box plots.

Saves:
  figS_cv_boxplots_cls.pdf  — AUPRC across 5 folds, top 9 clf combos
  figS_cv_boxplots_reg.pdf  — R² across 5 folds, top 9 reg combos

Usage
-----
  from figS_cv_boxplots import plot_cv_boxplots_cls, plot_cv_boxplots_reg
  plot_cv_boxplots_cls()
  plot_cv_boxplots_reg()

Requires: cv_results_folds.csv, cv_results_folds_reg.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

GNN_COLOR = "#2C7BB6"
CLS_COLOR = "#D7191C"
GNN_MODELS = {"GCN", "GINE", "GATV2"}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

DPI    = 300
OUTDIR = "."
TOP_N  = 9   # number of combinations to show


def _model_color(name):
    """Blue for GNN, red for classical."""
    model = name.split(" (")[0].upper()
    return GNN_COLOR if model in GNN_MODELS else CLS_COLOR


def _boxplot(ax, data_dict, metric, ylabel, title, top_n=TOP_N):
    """
    Draw horizontal box plots sorted by median metric descending.

    data_dict: {name: [fold_values]}
    """
    # Sort by median, take top N
    medians = {k: np.median(v) for k, v in data_dict.items()}
    sorted_names = sorted(medians, key=medians.get, reverse=True)[:top_n]
    sorted_names = sorted_names[::-1]   # flip for bottom-to-top display

    data   = [data_dict[n] for n in sorted_names]
    colors = [_model_color(n) for n in sorted_names]

    bp = ax.boxplot(data, vert=False, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=1),
                    capprops=dict(lw=1),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_yticks(range(1, len(sorted_names) + 1))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", lw=0.5, alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(facecolor=GNN_COLOR, alpha=0.75, label="GNN"),
        mpatches.Patch(facecolor=CLS_COLOR, alpha=0.75, label="Classical"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=8,
              loc="lower right")


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — AUPRC across folds
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_boxplots_cls(csv_path="cv_results_folds.csv"):
    df = pd.read_csv(csv_path)

    # Build dict: name → list of per-fold AUPRC values
    data_dict = {}
    for name, grp in df.groupby("name"):
        data_dict[name] = grp["auprc"].tolist()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _boxplot(ax, data_dict, "auprc",
             ylabel="AUPRC (5-fold CV)",
             title=f"Classification CV Stability — Top {TOP_N} Combinations")
    fig.tight_layout()
    path = f"{OUTDIR}/figS_cv_boxplots_cls.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION — R² across folds
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_boxplots_reg(csv_path="cv_results_folds_reg.csv"):
    df = pd.read_csv(csv_path)

    data_dict = {}
    for name, grp in df.groupby("name"):
        data_dict[name] = grp["r2"].tolist()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _boxplot(ax, data_dict, "r2",
             ylabel="R² (5-fold CV)",
             title=f"Regression CV Stability — Top {TOP_N} Combinations")
    fig.tight_layout()
    path = f"{OUTDIR}/figS_cv_boxplots_reg.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")