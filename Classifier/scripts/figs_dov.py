"""
figS_dov.py
===========
SI Figure — Domain of Validity analysis.

Panels:
  figS_dov_cls.pdf  — classification DoV
    Panel A: DoV score distribution (train LOO vs test), threshold marked
    Panel B: AUPRC by reliability tier (best model = XGB FP+pooled poly)

  figS_dov_reg.pdf  — regression DoV
    Panel A: DoV score distribution (train LOO vs test), threshold marked
    Panel B: MAE by reliability tier (best model = XGB FP+pooled RU)

Also:
  fig2c_gnn_efficiency_reg.pdf  — GNN R² vs training time (regression)

Usage
-----
  from figS_dov import plot_dov_cls, plot_dov_reg, plot_reg_efficiency
  
  # Classification (in classifier notebook):
  plot_dov_cls(dov, result_df, test_results)
  
  # Regression (in regression notebook):
  plot_dov_reg(dov_reg, result_df_reg, test_results_reg)
  
  # Regression GNN efficiency (in regression notebook):
  plot_reg_efficiency()   # requires cv_results_reg.csv

Requires:
  test_predictions.json         (classification)
  test_predictions_reg.json     (regression)
  cv_results_reg.csv            (regression efficiency)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import average_precision_score

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

COLOR_TRAIN    = "#2C7BB6"
COLOR_TEST     = "#D7191C"
COLOR_THRESH   = "#333333"

TIER_COLORS = {
    "Very low": "#D7191C",
    "Low":      "#FDAE61",
    "Moderate": "#ABD9E9",
    "High":     "#2C7BB6",
}
TIER_ORDER = ["Very low", "Low", "Moderate", "High"]

REP_COLORS_REG = {
    "PBSG":          "#2C7BB6",
    "SMILES+global": "#D7191C",
    "SMILES":        "#1A9641",
}
MODEL_MARKERS_REG = {
    "GCN":   "o",
    "GINE":  "s",
    "GATV2": "^",
}
REP_ORDER_REG   = ["PBSG", "SMILES+global", "SMILES"]
MODEL_ORDER_REG = ["GCN", "GINE", "GATV2"]

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

DPI    = 300
OUTDIR = "Figures"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: DoV distribution panel
# ══════════════════════════════════════════════════════════════════════════════

def _plot_dov_distribution(ax, dov, result_df):
    """Panel A: train LOO scores vs test scores, threshold marked."""
    train_scores = dov.loo_scores_
    test_scores  = result_df["dov_score"].values
    threshold    = dov.threshold_

    bins = np.linspace(
        min(train_scores.min(), test_scores.min()) - 0.01,
        max(train_scores.max(), test_scores.max()) + 0.01,
        35,
    )

    ax.hist(train_scores, bins=bins, color=COLOR_TRAIN, alpha=0.45,
            density=True, label=f"Train LOO (n={len(train_scores)})",
            zorder=2)
    ax.hist(test_scores, bins=bins, color=COLOR_TEST, alpha=0.55,
            density=True, label=f"Test (n={len(test_scores)})",
            zorder=2)

    ax.axvline(threshold, color=COLOR_THRESH, lw=2, linestyle="--",
               zorder=4,
               label=f"Threshold (p5={threshold:.3f})")

    # Shade OOD region
    ax.axvspan(bins[0], threshold, alpha=0.08, color=COLOR_TEST,
               zorder=1, label="OOD region")

    n_ood = (test_scores < threshold).sum()
    ax.text(threshold - 0.005, ax.get_ylim()[1] * 0.85,
            f"OOD\nn={n_ood}",
            ha="right", va="top", fontsize=8.5,
            color=COLOR_TEST, fontweight="bold")

    ax.set_xlabel("DoV Score (mean Tanimoto, k=5 NN)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Domain of Validity Distribution", fontsize=10,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION DoV
# ══════════════════════════════════════════════════════════════════════════════

def plot_dov_cls(dov, result_df, test_results,
                 best_model_key="xgb_poly"):
    """
    Parameters
    ----------
    dov           : fitted DomainOfValidityV2 instance
    result_df     : output of dov.score_df() on test set
    test_results  : dict from run_all_test (has probs/preds/labels per model)
    best_model_key: key in test_results for the model to show in Panel B
    """
    # Load predictions
    labels = np.array(test_results[best_model_key]["labels"])
    probs  = np.array(test_results[best_model_key]["probs"])

    tiers = result_df["reliability"].astype(str).values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Panel A ───────────────────────────────────────────────────────────────
    _plot_dov_distribution(axes[0], dov, result_df)

    # ── Panel B: AUPRC per reliability tier ───────────────────────────────────
    ax = axes[1]
    tier_auprcs = []
    tier_ns     = []
    tier_colors = []

    for tier in TIER_ORDER:
        mask = tiers == tier
        n    = mask.sum()
        tier_ns.append(n)
        if n < 2 or len(np.unique(labels[mask])) < 2:
            tier_auprcs.append(np.nan)
        else:
            tier_auprcs.append(
                average_precision_score(labels[mask], probs[mask])
            )
        tier_colors.append(TIER_COLORS[tier])

    x = np.arange(len(TIER_ORDER))
    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in tier_auprcs],
                  color=tier_colors, alpha=0.88,
                  edgecolor="white", lw=0.5)

    # n labels and NaN markers
    for i, (bar, n, val) in enumerate(zip(bars, tier_ns, tier_auprcs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={n}" if not np.isnan(val) else f"n={n}\n(insuf.)",
                ha="center", va="bottom", fontsize=8, color="dimgrey")

    # Overall AUPRC reference line
    overall = average_precision_score(labels, probs)
    ax.axhline(overall, color="grey", lw=1.2, linestyle="--",
               alpha=0.7, label=f"Overall AUPRC = {overall:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER, fontsize=9.5)
    ax.set_xlabel("Reliability Tier", fontsize=10)
    ax.set_ylabel("AUPRC", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Classifier Performance by DoV Tier\n"
                 f"({best_model_key.replace('_', ' ').upper()})",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", lw=0.5, alpha=0.3)

    fig.suptitle("Domain of Validity — Classification",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = f"{OUTDIR}/figS_dov_cls.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION DoV
# ══════════════════════════════════════════════════════════════════════════════

def plot_dov_reg(dov_reg, result_df_reg, test_results_reg,
                 best_model_key="xgb_ru"):
    """
    Parameters
    ----------
    dov_reg         : fitted DomainOfValidityV2 for regression
    result_df_reg   : output of dov_reg.score_df() on regression test set
    test_results_reg: dict from run_all_test_reg
    best_model_key  : key in test_results_reg for Panel B
    """
    labels = np.array(test_results_reg[best_model_key]["labels"])
    preds  = np.array(test_results_reg[best_model_key]["preds"])
    tiers  = result_df_reg["reliability"].astype(str).values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Panel A ───────────────────────────────────────────────────────────────
    _plot_dov_distribution(axes[0], dov_reg, result_df_reg)

    # ── Panel B: MAE per reliability tier ─────────────────────────────────────
    ax = axes[1]
    tier_maes   = []
    tier_ns     = []
    tier_colors = []

    for tier in TIER_ORDER:
        mask = tiers == tier
        n    = mask.sum()
        tier_ns.append(n)
        if n == 0:
            tier_maes.append(np.nan)
        else:
            tier_maes.append(
                float(np.mean(np.abs(labels[mask] - preds[mask])))
            )
        tier_colors.append(TIER_COLORS[tier])

    x = np.arange(len(TIER_ORDER))
    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in tier_maes],
                  color=tier_colors, alpha=0.88,
                  edgecolor="white", lw=0.5)

    for bar, n, val in zip(bars, tier_ns, tier_maes):
        label = f"n={n}" if not np.isnan(val) else f"n={n}\n(none)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                label, ha="center", va="bottom",
                fontsize=8, color="dimgrey")

    # Overall MAE reference line
    overall_mae = float(np.mean(np.abs(labels - preds)))
    ax.axhline(overall_mae, color="grey", lw=1.2, linestyle="--",
               alpha=0.7, label=f"Overall MAE = {overall_mae:.1f}°C")

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER, fontsize=9.5)
    ax.set_xlabel("Reliability Tier", fontsize=10)
    ax.set_ylabel("MAE (°C)", fontsize=10)
    ax.set_title("Regressor Performance by DoV Tier\n"
                 f"({best_model_key.replace('_', ' ').upper()})",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", lw=0.5, alpha=0.3)

    fig.suptitle("Domain of Validity — Regression",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = f"{OUTDIR}/figS_dov_reg.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION GNN EFFICIENCY SCATTER (R² vs training time)
# ══════════════════════════════════════════════════════════════════════════════

def plot_reg_efficiency(csv_path="cv_results_reg.csv"):
    """
    R² vs training time scatter for GNN graph rep combos only.
    Mirrors the classification efficiency scatter.
    """
    df = pd.read_csv(csv_path)
    df["model"] = df["name"].str.extract(r"^(.+?)\s*\(")[0].str.strip().str.upper()
    df["rep"]   = df["name"].str.extract(r"\((.+)\)")[0].str.strip()

    graph_reps = {"PBSG", "SMILES+global", "SMILES"}
    df_gnn = df[
        df["model"].isin(MODEL_ORDER_REG) &
        df["rep"].isin(graph_reps)
    ].copy()

    df_gnn["time_min"] = df_gnn["time_s"] / 60
    df_gnn = df_gnn[df_gnn["time_min"] < 50]  # exclude extreme outliers

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for _, row in df_gnn.iterrows():
        model  = row["model"]
        rep    = row["rep"]
        color  = REP_COLORS_REG.get(rep, "#999999")
        marker = MODEL_MARKERS_REG.get(model, "o")

        ax.scatter(row["time_min"], row["r2"],
                   c=color, marker=marker, s=100,
                   alpha=0.92, edgecolors="white", linewidths=0.8,
                   zorder=4)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Training Time (minutes, 5-fold CV)", fontsize=10)
    ax.set_ylabel("Mean R² (5-fold CV)", fontsize=10)
    ax.set_title("GNN Performance vs Training Time\n(Regression)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", lw=0.5, alpha=0.3)

    # ── Legends ───────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    rep_patches = [
        mpatches.Patch(facecolor=REP_COLORS_REG[r], alpha=0.92, label=r)
        for r in REP_ORDER_REG
    ]
    model_handles = [
        mlines.Line2D([], [], color="grey",
                      marker=MODEL_MARKERS_REG[m],
                      markersize=8, linestyle="None", label=m)
        for m in MODEL_ORDER_REG
    ]
    leg1 = ax.legend(handles=rep_patches, frameon=False,
                     fontsize=8.5, loc="lower right",
                     title="Graph rep", title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, frameon=False,
              fontsize=8.5, loc="upper left",
              title="Model", title_fontsize=8.5)

    fig.tight_layout()
    path = f"{OUTDIR}/fig2c_gnn_efficiency_reg.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")