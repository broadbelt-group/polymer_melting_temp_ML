import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    average_precision_score,
)


# COLORS 

COLOR_GINE   =  "#BD2D87"   # pink
COLOR_XGB    = "#ff7f0e"   # orange
COLOR_RANDOM = "#AAAAAA"   # grey for random baseline

# Stereo class colors
STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#ABD9E9",
    "atactic":      "#FDAE61",
    "stereo-irregular": "#D7191C",
    "achiral":      "#1A9641",
}

# Architecture colors
ARCH_COLORS = {
    "alternating": "#7B2D8B",
    "block":       "#E66101",
    "random":      "#4DAC26",
}

# ── Figure settings ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       14,
    "axes.linewidth":  1.2,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype":    42,   
    "svg.fonttype":    "none",
})

FIG_W  = 3.5   
FIG_H  = 3.2
DPI    = 300
OUTDIR = "Figures"   



# LOAD DATA


with open("test_predictions-full.json") as f:
    preds = json.load(f)

with open("test_results_subgroups-full.json") as f:
    subgroups = json.load(f)

with open("test_results-full.csv") as f:
    metrics_df = pd.read_csv(f).set_index("name")

# Model keys in the JSON
MODELS = {
    "GINE (PBSG)":          ("gine_pbsg",   COLOR_GINE, "GINE (PBSG)"),
    "XGB (FP+pooled poly)": ("xgb_fp_pooled_poly",    COLOR_XGB,  "XGB (FP+pooled poly)"),
}


def _get(model_name, key):
    mk = MODELS[model_name][0]
    return np.array(preds[mk][key])


# ROC CURVES (one per model)

def plot_roc(model_name, color, label, suffix):
    labels = _get(model_name, "labels")
    probs  = _get(model_name, "probs")
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLOR_RANDOM, lw=1,
            linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {label}", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    path = f"{OUTDIR}/fig3a_roc_{suffix}.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# PR CURVES (one per model)

def plot_pr(model_name, color, label, suffix):
    labels = _get(model_name, "labels")
    probs  = _get(model_name, "probs")
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap           = average_precision_score(labels, probs)
    baseline     = labels.mean()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(rec, prec, color=color, lw=2,
            label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color=COLOR_RANDOM, lw=1, linestyle="--",
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {label}", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    path = f"{OUTDIR}/fig3b_pr_{suffix}.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# CONFUSION MATRICES (one per model)

from matplotlib.colors import LinearSegmentedColormap

def make_cmap(base_color):
    return LinearSegmentedColormap.from_list(
        "", ["#ffffff", base_color]
    )

def plot_confusion(model_name, color, label, suffix):
    labels_arr = _get(model_name, "labels")
    preds_arr  = _get(model_name, "preds")
    cm = confusion_matrix(labels_arr, preds_arr)

    simple_color = "#ff7f0e"   # orange
    gnn_color    = "#BD2D87"

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    if model_name=="GINE (PBSG)":
        im = ax.imshow(cm, cmap=make_cmap(gnn_color), aspect="auto")
    else:
        im = ax.imshow(cm, cmap=make_cmap(simple_color), aspect="auto")
    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Tm", "Has Tm"])
    ax.set_yticklabels(["No Tm", "Has Tm"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if model_name=="XGB (FP+pooled poly)":
        ax.set_title("XGB (PolyFP+pooled)", fontsize=16, fontweight="bold")
    else: 
        ax.set_title(f"{label}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = f"{OUTDIR}/fig3c_cm_{suffix}.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# SUBGROUP F1 BY STEREO CLASS

def plot_subgroup_stereo():
    stereo_order = ["isotactic", "syndiotactic", "atactic",
                    "stereo-irregular", "achiral"]
    model_keys   = [("gine_pbsg", "GINE (PBSG)", COLOR_GINE),
                    ("xgb_fp_pooled_poly",  "XGB (poly)",  COLOR_XGB)]

    x     = np.arange(len(stereo_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(FIG_W * 1.8, FIG_H))

    for i, (mk, label, color) in enumerate(model_keys):
        f1s = []
        ns  = []
        for grp in stereo_order:
            entry = subgroups[mk].get("stereo_class", {}).get(grp, {})
            f1s.append(entry.get("f1", 0.0))
            ns.append(entry.get("n", 0))

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", lw=0.5)

        # Annotate n
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in stereo_order],
                       rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("F1 by Stereo Class", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = f"{OUTDIR}/fig3d_subgroup_stereo.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# SUBGROUP F1 BY COPOLYMER ARCHITECTURE

def plot_subgroup_arch():
    arch_order = ["alternating", "block", "random"]
    model_keys = [("gine_pbsg", "GINE (PBSG)", COLOR_GINE),
                  ("xgb_fp_pooled_poly",  "XGB (poly)",  COLOR_XGB)]

    x     = np.arange(len(arch_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(FIG_W * 1.4, FIG_H))

    for i, (mk, label, color) in enumerate(model_keys):
        f1s = []
        ns  = []
        for grp in arch_order:
            entry = subgroups[mk].get("copolymer_type", {}).get(grp, {})
            f1s.append(entry.get("f1", 0.0))
            ns.append(entry.get("n", 0))

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", lw=0.5)

        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in arch_order])
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("F1 by Copolymer Architecture", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = f"{OUTDIR}/fig3e_subgroup_arch.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# RUN ALL

if __name__ == "__main__":
    for model_name, (mk, color, label) in MODELS.items():
        print(model_name)
        suffix = mk
        plot_roc(model_name, color, label, suffix)
        plot_pr(model_name, color, label, suffix)
        plot_confusion(model_name, color, label, suffix)

    plot_subgroup_stereo()
    plot_subgroup_arch()
    print("\nAll Figure 3 panels saved.")