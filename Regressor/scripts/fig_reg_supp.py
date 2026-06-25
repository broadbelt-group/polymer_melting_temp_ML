
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# COLORS

COLOR_TRAIN = "#2C7BB6"
COLOR_TEST  = "#D7191C"

STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#74ADD1",
    "atactic":      "#FDAE61",
    "stereo-irregular": "#D7191C",
    "achiral":      "#1A9641",
}
STEREO_ORDER = ["isotactic", "syndiotactic", "atactic", "stereo-irregular", "achiral"]

GNN_ROWS = ["GCN", "GINE", "GATV2"]
CLS_ROWS = ["RR", "RF", "XGB"]

GNN_COLS = ["PBSG", "SMILES+global", "SMILES"]
CLS_COLS = [
    "FP+pooled RU",   "FP RU",
    "FP+pooled poly", "FP poly",
    "FP SMILES+gl",   "FP SMILES",
]

REP_DISPLAY = {
    "PBSG":          "PBSG",
    "SMILES+global": "SMILES+global",
    "SMILES":        "SMILES",
    "FP+pooled RU":  "FP+pool RU",
    "FP RU":         "FP RU",
    "FP+pooled poly":"FP+pool poly",
    "FP poly":       "FP poly",
    "FP SMILES+gl":  "FP SMILES+gl",
    "FP SMILES":     "FP SMILES",
}

CMAP = "Purples"

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


# REGRESSION SPLIT VALIDATION

def plot_reg_split_validation(df, trainval_idx, test_idx):
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    df_tv   = df.iloc[trainval_idx]
    df_test = df.iloc[test_idx]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8),
                          gridspec_kw={"width_ratios": [1.4, 1.8, 1.8]})

    # ── LEFT: Tm distribution ─────────────────────────────────────────────────
    ax = axes[0]
    tm_train = df_tv["Tm"].values
    tm_test  = df_test["Tm"].values

    bins = np.linspace(
        min(tm_train.min(), tm_test.min()) - 5,
        max(tm_train.max(), tm_test.max()) + 5,
        25,
    )

    ax.hist(tm_train, bins=bins, color=COLOR_TRAIN, alpha=0.35,
            density=True, zorder=2)
    ax.hist(tm_test,  bins=bins, color=COLOR_TEST,  alpha=0.35,
            density=True, zorder=2)

    for tm, color, label, n in [
        (tm_train, COLOR_TRAIN, "Train+Val", len(tm_train)),
        (tm_test,  COLOR_TEST,  "Test",      len(tm_test)),
    ]:
        kde = gaussian_kde(tm, bw_method=0.3)
        x_  = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x_, kde(x_), color=color, lw=0, zorder=3,
                label=f"{label} (n={n})\n"
                      f"{tm.mean():.1f}±{tm.std():.1f}°C")

    ax.set_xlabel("T$_m$ (°C)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("T$_m$ Distribution", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    # ── MIDDLE: Stereo class bar ─────────────────────────────────────────────
    ax = axes[1]
    n_stereo = len(STEREO_ORDER)
    bar_w    = 0.35
    x        = np.arange(n_stereo)

    splits      = [df_tv, df_test]
    split_names = [f"Train+Val\n(n={len(df_tv)})",
                   f"Test\n(n={len(df_test)})"]
    split_cols  = [COLOR_TRAIN, COLOR_TEST]

    for i, (sub, label, color) in enumerate(zip(splits, split_names, split_cols)):
        pcts = [100 * (sub["stereo_class"] == s).sum() / len(sub)
                for s in STEREO_ORDER]
        offset = (i - 0.5) * bar_w
        bars = ax.bar(x + offset, pcts, bar_w,
                      color=color, alpha=0.85,
                      edgecolor="white", lw=0.5,
                      label=label)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, color="dimgrey")
                

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STEREO_ORDER],
                       rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Stereo Class Distribution", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(False)

    # ── RIGHT: copolymer architecture ──────────────────────────────────────
    ax = axes[2]
    df_copy = df.copy()
    df_copy["copolymer_type"] = df_copy["copolymer_type"].fillna("homopolymer")
    df_tv_c   = df_copy.iloc[trainval_idx]
    df_test_c = df_copy.iloc[test_idx]
    ARCH_ORDER = ["homopolymer", "random", "alternating", "block"]
    n_arch = len(ARCH_ORDER)
    x      = np.arange(n_arch)
    bar_w  = 0.35

    for i, (sub, label, color) in enumerate(zip(
            [df_tv_c, df_test_c],
            [f"Train+Val", f"Test"],
            [COLOR_TRAIN, COLOR_TEST])):
        pcts   = [100 * (sub["copolymer_type"] == a).sum() / len(sub)
                  for a in ARCH_ORDER]
        offset = (i - 0.5) * bar_w
        bars   = ax.bar(x + offset, pcts, bar_w, color=color, alpha=0.85,
                        edgecolor="white", lw=0.5, label=label)
        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ARCH_ORDER],
                       rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Copolymer Architecture", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(False)

    fig.suptitle("Regression Train/Test Split Validation",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = f"{OUTDIR}/fig_reg_split_validation.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# REGRESSION CV HEATMAPS

def _heatmap(pivot, pivot_std, row_order, col_order,
             title, outpath, n_graph_reps=0):
    vals = pivot.values.astype(float)
    vmin = np.nanpercentile(vals, 10)
    vmax = np.nanpercentile(vals, 90)

    n_rows, n_cols = vals.shape
    cell_h = 0.58
    cell_w = 0.92
    fig_h  = n_rows * cell_h + 1.6
    fig_w  = n_cols * cell_w + 2.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = vals[i, j]
            std = pivot_std.values[i, j]
            if np.isnan(val):
                continue
            norm_val   = (val - vmin) / max(vmax - vmin, 1e-6)
            text_color = "white" if norm_val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}\n±{std:.3f}",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color,
                    fontweight="bold" if norm_val > 0.65 else "normal")

    display_cols = [REP_DISPLAY.get(c, c) for c in col_order]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(display_cols, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_order, fontsize=10, fontweight="bold")
    ax.set_xlabel("Representation", fontsize=10, labelpad=8)
    ax.set_ylabel("Model", fontsize=10, labelpad=8)

    if n_graph_reps > 0 and n_graph_reps < n_cols:
        ax.axvline(n_graph_reps - 0.5, color="white", lw=3)
        ax.axvline(n_graph_reps - 0.5, color="grey", lw=1,
                   linestyle="--", alpha=0.7)

    # Pair dividers for vector reps
    vec_start = n_graph_reps
    for pair_end in range(vec_start + 2, n_cols, 2):
        ax.axvline(pair_end - 0.5, color="white", lw=1.5)
        ax.axvline(pair_end - 0.5, color="lightgrey", lw=0.7,
                   linestyle=":", alpha=0.8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_reg_cv_heatmaps(csv_path="cv_results_reg.csv"):
    df = pd.read_csv(csv_path)
    df["model"] = df["name"].str.extract(r"^(.+?)\s*\(")[0].str.strip().str.upper()
    df["rep"]   = df["name"].str.extract(r"\((.+)\)")[0].str.strip()

    # ── GNN heatmap ───────────────────────────────────────────────────────────
    df_gnn = df[df["model"].isin(GNN_ROWS)]
    if len(df_gnn) > 0:
        pivot     = df_gnn.pivot(index="model", columns="rep", values="r2")
        pivot_std = df_gnn.pivot(index="model", columns="rep", values="r2_std")
        rows = [r for r in GNN_ROWS if r in pivot.index]
        cols = [c for c in GNN_COLS if c in pivot.columns]
        if rows and cols:
            n_graph = sum(1 for c in cols
                          if c in {"PBSG", "SMILES+global", "SMILES"})
            _heatmap(pivot.loc[rows, cols], pivot_std.loc[rows, cols],
                     rows, cols,
                     title="GNN Models — CV R² (Regression)",
                     outpath=f"{OUTDIR}/fig2a_cv_heatmap_gnn_reg.pdf",
                     n_graph_reps=n_graph)

    # ── Classical heatmap ─────────────────────────────────────────────────────
    df_cls = df[df["model"].isin(CLS_ROWS)]
    if len(df_cls) > 0:
        pivot     = df_cls.pivot(index="model", columns="rep", values="r2")
        pivot_std = df_cls.pivot(index="model", columns="rep", values="r2_std")
        rows = [r for r in CLS_ROWS if r in pivot.index]
        cols = [c for c in CLS_COLS if c in pivot.columns]
        if rows and cols:
            _heatmap(pivot.loc[rows, cols], pivot_std.loc[rows, cols],
                     rows, cols,
                     title="Classical Models — CV R² (Regression)",
                     outpath=f"{OUTDIR}/fig2a_cv_heatmap_cls_reg.pdf",
                     n_graph_reps=0)