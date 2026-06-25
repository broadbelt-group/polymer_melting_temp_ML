import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# COLORS

CMAP_GNN = "Purples"      # heatmap colormap for GNN panel
CMAP_CLS = "Purples"     # heatmap colormap for classical panel

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       14,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
})

DPI    = 300
OUTDIR = "Figures"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


# LOAD & PARSE

df = pd.read_csv("cv_results_full.csv")
df["model"] = df["name"].str.extract(r"^(.+?)\s*\(")[0].str.strip().str.upper()
df["rep"]   = df["name"].str.extract(r"\((.+)\)")[0].str.strip()

# ── Column orders ──────────────────────────────────────────────────────────────
GNN_COLS = [
    # Graph reps
    "PBSG", "SMILES+global", "SMILES",
    # Vector reps: pooled paired with non-pooled
    "FP+pooled RU",   "FP RU",
    "FP+pooled poly", "FP poly",
    "FP SMILES+gl",   "FP SMILES",
]

CLS_COLS = [
    "FP+pooled RU",   "FP RU",
    "FP+pooled poly", "FP poly",
    "FP SMILES+gl",   "FP SMILES",
]

REP_DISPLAY = {
    "PBSG":           "PBSG",
    "SMILES+global":  "SMILES+global",
    "SMILES":         "SMILES",
    "FP+pooled RU":   "RepFP+pooled",
    "FP RU":          "RepFP",
    "FP+pooled poly": "PolyFP+pooled",
    "FP poly":        "PolyFP",
    "FP SMILES+gl":   "Smi+global",
    "FP SMILES":      "Smi",
}

GNN_ROWS = ["GCN", "GINE", "GATV2"]
CLS_ROWS = ["LR", "RF", "XGB"]


# SHARED PLOT FUNCTION

def _plot_heatmap(pivot, pivot_std, row_order, col_order,
                  cmap, title, outpath,
                  n_graph_reps=0,    # draw vertical divider after this many cols
                  vmin=None, vmax=None):
    vals = pivot.values.astype(float)

    if vmin is None:
        vmin = np.nanpercentile(vals, 5)
    if vmax is None:
        vmax = np.nanpercentile(vals, 95)

    n_rows, n_cols = vals.shape
    cell_h = 0.58
    cell_w = 0.92
    fig_h  = n_rows * cell_h + 1.6
    fig_w  = n_cols * cell_w + 2.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i in range(n_rows):
        for j in range(n_cols):
            val = vals[i, j]
            std = pivot_std.values[i, j]
            if np.isnan(val):
                continue
            norm_val  = (val - vmin) / max(vmax - vmin, 1e-6)
            text_color = "white" if norm_val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}\n±{std:.3f}",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color,
                    fontweight="bold" if norm_val > 0.65 else "normal")

    # ── Tick labels ───────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    display_labels = [REP_DISPLAY.get(c, c) for c in col_order]
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_order, fontsize=10, fontweight="bold")

    ax.set_xlabel("Representation", fontsize=10, labelpad=8)
    ax.set_ylabel("Model", fontsize=10, labelpad=8)

    # ── Divider: graph reps vs vector reps ────────────────────────────────────
    if n_graph_reps > 0 and n_graph_reps < n_cols:
        ax.axvline(n_graph_reps - 0.5, color="white", lw=3)
        ax.axvline(n_graph_reps - 0.5, color="grey",  lw=1,
                   linestyle="--", alpha=0.7)

    # ── Divider: pooled vs non-pooled pairs (every 2 vector rep cols) ─────────
    # First vector rep column index
    vec_start = n_graph_reps
    for pair_end in range(vec_start + 2, n_cols, 2):
        ax.axvline(pair_end - 0.5, color="white", lw=1.5)
        ax.axvline(pair_end - 0.5, color="lightgrey", lw=0.7,
                   linestyle=":", alpha=0.8)

    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# GNN HEATMAP

def plot_gnn_heatmap():
    df_gnn = df[df["model"].isin(GNN_ROWS)]
    pivot  = df_gnn.pivot(index="model", columns="rep", values="auprc")
    pivot_std = df_gnn.pivot(index="model", columns="rep", values="auprc_std")

    # Filter to available rows/cols
    rows = [r for r in GNN_ROWS if r in pivot.index]
    cols = [c for c in GNN_COLS if c in pivot.columns]
    pivot     = pivot.loc[rows, cols]
    pivot_std = pivot_std.loc[rows, cols]

    # Count graph rep columns for divider
    n_graph = sum(1 for c in cols if c in ["PBSG", "SMILES+global", "SMILES"])

    _plot_heatmap(
        pivot, pivot_std, rows, cols,
        cmap        = CMAP_GNN,
        title       = "GNN Models — CV AUPRC",
        outpath     = f"{OUTDIR}/fig2a_cv_heatmap_gnn.pdf",
        n_graph_reps= n_graph,
    )


# CLASSICAL HEATMAP

def plot_classical_heatmap():
    df_cls = df[df["model"].isin(CLS_ROWS)]
    pivot  = df_cls.pivot(index="model", columns="rep", values="auprc")
    pivot_std = df_cls.pivot(index="model", columns="rep", values="auprc_std")

    rows = [r for r in CLS_ROWS if r in pivot.index]
    cols = [c for c in CLS_COLS if c in pivot.columns]
    pivot     = pivot.loc[rows, cols]
    pivot_std = pivot_std.loc[rows, cols]

    _plot_heatmap(
        pivot, pivot_std, rows, cols,
        cmap        = CMAP_CLS,
        title       = "Classical Models — CV AUPRC",
        outpath     = f"{OUTDIR}/fig2a_cv_heatmap_classical.pdf",
        n_graph_reps= 0,   # no graph reps for classical
    )


# RUN

if __name__ == "__main__":
    plot_gnn_heatmap()
    plot_classical_heatmap()
    print("\nBoth heatmaps saved.")