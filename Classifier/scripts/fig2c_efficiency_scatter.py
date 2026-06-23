"""
fig2c_gnn_efficiency.py
=======================
Figure 2 Panel C — GNN efficiency scatter.
x = training time (minutes), y = mean AUPRC
9 points: 3 models × 3 graph representations

Color = representation (PBSG/SMILES+global/SMILES) — clearly distinct
Marker = model (GCN/GINE/GATv2) — shape encodes model
Labels = short, manually offset to avoid overlap
x-axis = linear minutes

Saved as: fig2c_gnn_efficiency.pdf

Requires: cv_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ══════════════════════════════════════════════════════════════════════════════
# COLORS & MARKERS — rep gets color, model gets shape
# ══════════════════════════════════════════════════════════════════════════════

REP_COLORS = {
    "PBSG":          "#C5407E",   
    "SMILES+global": "#1F77B4",  
    "SMILES":        "#AEC6E8",  
}

MODEL_MARKERS = {
    "GCN":   "o",
    "GINE":  "s",
    "GATV2": "^",
}

MODEL_SIZE = 200

REP_ORDER   = ["PBSG", "SMILES+global", "SMILES"]
MODEL_ORDER = ["GCN", "GINE", "GATV2"]

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          14,
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
# LOAD & PARSE
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("cv_results_full.csv")
df["model"] = df["name"].str.extract(r"^(.+?)\s*\(")[0].str.strip().str.upper()
df["rep"]   = df["name"].str.extract(r"\((.+)\)")[0].str.strip()

graph_reps = {"PBSG", "SMILES+global", "SMILES"}
df_gnn = df[
    df["model"].isin(MODEL_ORDER) &
    df["rep"].isin(graph_reps)
].copy()
df_gnn["time_min"] = df_gnn["time_s"] / 60


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for _, row in df_gnn.iterrows():
    model  = row["model"]
    rep    = row["rep"]
    x      = row["time_min"]
    y      = row["auprc"]
    color  = REP_COLORS[rep]
    marker = MODEL_MARKERS[model]

    ax.scatter(x, y, c=color, marker=marker, s=MODEL_SIZE,
               alpha=0.92, edgecolors="white", linewidths=0.8,
               zorder=4)


# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Training Time (minutes, 5-fold CV)", fontsize=14)
ax.set_ylabel("Mean AUPRC (5-fold CV)", fontsize=14)
#ax.set_title("GNN Performance vs Training Time", fontsize=11, fontweight="bold")

# Linear x-axis with clean ticks
ax.set_xlim(left=0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

# Zoom y-axis
ymin = df_gnn["auprc"].min() - df_gnn["auprc_std"].max() - 0.012
ymax = df_gnn["auprc"].max() + df_gnn["auprc_std"].max() + 0.015
ax.set_ylim(ymin, ymax)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(False)


# ── Legends ───────────────────────────────────────────────────────────────────
# Color = representation
rep_patches = [
    mpatches.Patch(facecolor=REP_COLORS[r], alpha=0.92, label=r)
    for r in REP_ORDER
]

# Shape = model
model_handles = [
    mlines.Line2D([], [], color="grey", marker=MODEL_MARKERS[m],
                  markersize=8, linestyle="None", label=m)
    for m in MODEL_ORDER
]

"""leg1 = ax.legend(handles=rep_patches, frameon=False,
                 fontsize=8.5, loc="lower right",
                 title="Graph rep", title_fontsize=8.5)"""
#ax.add_artist(leg1)
ax.legend(handles=model_handles, frameon=False,
          fontsize=8.5, loc="lower right", ncols=3
          ) #title="Model", title_fontsize=8.5

fig.tight_layout()
path = f"{OUTDIR}/fig2c_gnn_efficiency.pdf"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")