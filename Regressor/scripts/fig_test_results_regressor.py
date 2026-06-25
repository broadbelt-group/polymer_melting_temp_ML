
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# COLORS

MODEL_COLORS = {
    "gatv2_pbsg":   "#BD2D87",
    "xgb_fp_pooled_ru":    "#2ca02c",
    "xgb_fp_ru":   "#98df8a",
    "xgb_fp_pooled_poly": "#ff7f0e",
}
MODEL_LABELS = {
    "gatv2_pbsg": "GATv2 (PBSG)",
    "xgb_fp_pooled_ru":    "XGB (RepFP+pooled)",
    "xgb_fp_ru":    "XGB (RepFP)",
    "xgb_fp_pooled_poly":  "XGB (PolyFP+pooled)",
}

REP_COLORS_REG = {
    "PBSG":          "#C5407E",   
    "SMILES+global": "#1F77B4",  
    "SMILES":        "#AEC6E8",  
}
MODEL_ORDER = ["gatv2_pbsg", "xgb_fp_pooled_ru", "xgb_fp_pooled_poly"]

STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#74ADD1",
    "atactic":      "#FDAE61",
    "stereo-irregular": "#D7191C",
    "achiral":      "#1A9641",
}
STEREO_ORDER = ["isotactic", "syndiotactic", "atactic", "stereo-irregular", "achiral"]

ARCH_COLORS = {
    "alternating": "#7B2D8B",
    "block":       "#E66101",
    "random":      "#4DAC26",
}
ARCH_ORDER = ["alternating", "block", "random"]

COLOR_TRAIN = "#2C7BB6"
COLOR_TEST  = "#D7191C"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          14,
    "axes.linewidth":     1.2,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

DPI    = 300
OUTDIR = "Figures"


# LOAD DATA

def _load_data():
    with open("test_predictions_reg_full.json") as f:
        preds = json.load(f)
    with open("test_results_reg_subgroups_full.json") as f:
        subgroups = json.load(f)
    return preds, subgroups


# PANEL A — Predicted vs Actual Tm (3 subplots)

def plot_pred_vs_actual(preds, df_reg, test_idx):
    test_idx = np.array(test_idx)
    stereo   = df_reg.iloc[test_idx]["stereo_class"].fillna("unknown").values

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

    for ax, mk in zip(axes, MODEL_ORDER):
        labels_arr = np.array(preds[mk]["labels"])
        preds_arr  = np.array(preds[mk]["preds"])
        color      = MODEL_COLORS[mk]
        print(mk)
        print(color)

        for s in STEREO_ORDER:
            mask = stereo == s
            if mask.sum() == 0:
                continue
            ax.scatter(labels_arr[mask], preds_arr[mask],
                       c=color, s=25, alpha=1.0,
                       edgecolors="none", zorder=3,
                       label=s.capitalize())

        # Perfect prediction line
        lims = [min(labels_arr.min(), preds_arr.min()) - 5,
                max(labels_arr.max(), preds_arr.max()) + 5]
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.5, zorder=2)

        # Metrics annotation
        r2  = 1 - np.sum((labels_arr - preds_arr)**2) / \
                  np.sum((labels_arr - labels_arr.mean())**2)
        mae = np.mean(np.abs(labels_arr - preds_arr))
        ax.text(0.05, 0.95,
                f"R² = {r2:.2f}\nMAE = {mae:.1f}°C",
                transform=ax.transAxes, fontsize=12,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="grey", alpha=0.8))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual T$_m$ (°C)", fontsize=14)
        ax.set_title(MODEL_LABELS[mk], fontsize=14, fontweight="bold",
                     color='black')

    axes[0].set_ylabel("Predicted T$_m$ (°C)", fontsize=14)

    # Shared stereo legend on last panel
    fig.tight_layout()
    path = f"{OUTDIR}/fig_pred_vs_actual.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# PANEL B — Residuals by Stereo Class (box plot, one panel per model)

def plot_residuals_stereo(preds, df_reg, test_idx):
    test_idx = np.array(test_idx)
    stereo   = df_reg.iloc[test_idx]["stereo_class"].fillna("unknown").values

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

    for ax, mk in zip(axes, MODEL_ORDER):
        labels_arr = np.array(preds[mk]["labels"])
        preds_arr  = np.array(preds[mk]["preds"])
        residuals  = preds_arr - labels_arr

        data   = []
        labels = []
        colors = []
        for s in STEREO_ORDER:
            mask = stereo == s
            if mask.sum() == 0:
                continue
            data.append(residuals[mask])
            labels.append(f"{s.capitalize()}\n(n={mask.sum()})")
            colors.append(STEREO_COLORS[s])

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=1.5),
                        whiskerprops=dict(lw=1),
                        capprops=dict(lw=1),
                        flierprops=dict(marker="o", markersize=3,
                                        alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(MODEL_LABELS[mk], fontsize=10, fontweight="bold",
                     color=MODEL_COLORS[mk])

    axes[0].set_ylabel("Residual (Predicted − Actual) °C", fontsize=10)
    fig.suptitle("Residuals by Stereo Class", fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = f"{OUTDIR}/fig_residuals_stereo.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# PANEL C — Subgroup MAE by Stereo Class

def plot_subgroup_mae_stereo(subgroups):
    x     = np.arange(len(STEREO_ORDER))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for i, mk in enumerate(MODEL_ORDER):
        maes = []
        ns   = []
        for s in STEREO_ORDER:
            entry = subgroups[mk].get("stereo_class", {}).get(s, {})
            maes.append(entry.get("mae", 0.0))
            ns.append(entry.get("n", 0))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, maes, width,
                      color=MODEL_COLORS[mk], alpha=1.0,
                      edgecolor="white", lw=0.5,
                      label=MODEL_LABELS[mk])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STEREO_ORDER],
                       rotation=0, ha="center")
    ax.set_ylabel("MAE (°C)", fontsize=14)
    ax.grid(False)

    fig.tight_layout()
    path = f"{OUTDIR}/fig_subgroup_mae_stereo.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# PANEL D — Subgroup MAE by Copolymer Architecture

def plot_subgroup_mae_arch(subgroups):
    x     = np.arange(len(ARCH_ORDER))
    width = 0.28
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for i, mk in enumerate(MODEL_ORDER):
        maes = []
        ns   = []
        for a in ARCH_ORDER:
            entry = subgroups[mk].get("copolymer_type", {}).get(a, {})
            maes.append(entry.get("mae", 0.0))
            ns.append(entry.get("n", 0))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, maes, width,
                      color=MODEL_COLORS[mk], alpha=1.0,
                      edgecolor="white", lw=0.5,
                      label=MODEL_LABELS[mk])

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ARCH_ORDER])
    ax.set_ylabel("MAE (°C)", fontsize=14)
    ax.grid(False)

    fig.tight_layout()
    path = f"{OUTDIR}/fig_subgroup_mae_arch.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# PANEL E — Tm Distribution: Train vs Test

def plot_tm_distribution(df_reg, trainval_idx, test_idx):
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    tm_train = df_reg.iloc[trainval_idx]["Tm"].values
    tm_test  = df_reg.iloc[test_idx]["Tm"].values

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Histogram
    bins = np.linspace(
        min(tm_train.min(), tm_test.min()) - 5,
        max(tm_train.max(), tm_test.max()) + 5,
        30
    )
    ax.hist(tm_train, bins=bins, color=COLOR_TRAIN, alpha=0.45,
            density=True, label=f"Train+Val (n={len(tm_train)})",
            zorder=2)
    ax.hist(tm_test,  bins=bins, color=COLOR_TEST,  alpha=0.45,
            density=True, label=f"Test (n={len(tm_test)})",
            zorder=2)

    # KDE overlay
    for tm, color in [(tm_train, COLOR_TRAIN), (tm_test, COLOR_TEST)]:
        kde  = gaussian_kde(tm, bw_method=0.3)
        x_   = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x_, kde(x_), color=color, lw=2, zorder=3)

    # Summary stats annotation
    ax.text(0.97, 0.95,
            f"Train: {tm_train.mean():.1f} ± {tm_train.std():.1f}°C\n"
            f"Test:  {tm_test.mean():.1f} ± {tm_test.std():.1f}°C",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="grey", alpha=1.0))

    ax.set_xlabel("T$_m$ (°C)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("T$_m$ Distribution: Train vs Test",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = f"{OUTDIR}/fig_tm_distribution.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# RUN ALL

def run_all_fig(df_reg, trainval_idx, test_idx):
    preds, subgroups = _load_data()
    plot_pred_vs_actual(preds, df_reg, test_idx)
    plot_residuals_stereo(preds, df_reg, test_idx)
    plot_subgroup_mae_stereo(subgroups)
    plot_subgroup_mae_arch(subgroups)
    plot_tm_distribution(df_reg, trainval_idx, test_idx)
    print("\nAll Figure panels saved.")