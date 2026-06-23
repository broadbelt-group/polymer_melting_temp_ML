"""
fig2b_cv_barchart.py
====================
Figure 2 Panel B — bar chart of top model × representation combinations,
sorted by mean AUPRC with ± std error bars.

Saved as:
  fig2b_cv_barchart.pdf

Usage
-----
  python fig2b_cv_barchart.py
  # or in notebook:
  %run fig2b_cv_barchart.py

Requires: cv_results.csv (the 27-combo untuned CV results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# COLORS — change to match your paper palette
# ══════════════════════════════════════════════════════════════════════════════

# Bar colors by model family
COLOR_GNN_GRAPH  = "#2C7BB6"   # GNN with graph rep (PBSG/SMILES)
COLOR_GNN_VECTOR = "#74ADD1"   # GNN with vector rep (if any in top N)
COLOR_XGB        = "#D7191C"   # XGB
COLOR_RF         = "#E87E1A"   # RF
COLOR_LR         = "#F9A653"   # LR/RR

MODEL_COLORS = {
    "GCN":   COLOR_GNN_GRAPH,
    "GINE":  COLOR_GNN_GRAPH,
    "GATV2": COLOR_GNN_GRAPH,
    "LR":    COLOR_LR,
    "RF":    COLOR_RF,
    "XGB":   COLOR_XGB,
}

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
TOP_N  = 10   # number of combinations to show — adjust as needed


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("cv_results_full.csv")

# Parse model and rep
df["model"] = df["name"].str.extract(r"^(.+?)\s*\(")[0].str.strip().str.upper()
df["rep"]   = df["name"].str.extract(r"\((.+)\)")[0].str.strip()

# Sort by AUPRC descending, take top N
df_top = df.sort_values("auprc", ascending=False).head(TOP_N).reset_index(drop=True)

# Short display names for x-axis
df_top["short_name"] = df_top["name"].str.replace(
    "FP+pooled", "FP+pool", regex=False
).str.replace("SMILES+global", "SMILES+gl", regex=False)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

fig_w = max(6.0, TOP_N * 0.75)
fig_h = 4.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

x      = np.arange(len(df_top))
colors = [MODEL_COLORS.get(m, "#999999") for m in df_top["model"]]

bars = ax.bar(
    x, df_top["auprc"],
    color=colors, alpha=0.88,
    edgecolor="white", linewidth=0.5,
    zorder=3,
)

# Error bars
ax.errorbar(
    x, df_top["auprc"],
    yerr=df_top["auprc_std"],
    fmt="none", color="black",
    capsize=4, capthick=1.2, elinewidth=1.2,
    zorder=4,
)

# Value labels on top of bars
for bar, val, std in zip(bars, df_top["auprc"], df_top["auprc_std"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.004,
        f"{val:.3f}",
        ha="center", va="bottom",
        fontsize=7.5, color="black",
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(
    df_top["short_name"],
    rotation=35, ha="right", fontsize=8.5,
)
ax.set_ylabel("Mean AUPRC (5-fold CV)", fontsize=10)
ax.set_title(f"Top {TOP_N} Model × Representation Combinations",
             fontsize=11, fontweight="bold")

# Y axis range — zoom in to show differences
ymin = max(0.0, df_top["auprc"].min() - df_top["auprc_std"].max() - 0.03)
ymax = df_top["auprc"].max() + df_top["auprc_std"].max() + 0.025
ax.set_ylim(ymin, ymax)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="y", lw=0.6, alpha=0.4, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=COLOR_GNN_GRAPH, alpha=0.88, label="GNN (graph rep)"),
    mpatches.Patch(facecolor=COLOR_XGB,       alpha=0.88, label="XGB"),
    mpatches.Patch(facecolor=COLOR_RF,        alpha=0.88, label="RF"),
    mpatches.Patch(facecolor=COLOR_LR,        alpha=0.88, label="LR / Ridge"),
]
# Only show legend items that actually appear
present_models = set(df_top["model"])
legend_items = [
    p for p in legend_items
    if any(k in p.get_label() for k in present_models)
    or ("GNN" in p.get_label() and any(
        m in present_models for m in ["GCN","GINE","GATV2"]))
]
ax.legend(handles=legend_items, frameon=False, fontsize=8,
          loc="lower right")

fig.tight_layout()
path = f"{OUTDIR}/fig2b_cv_barchart.pdf"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")