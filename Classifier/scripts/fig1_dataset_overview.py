"""
fig1_dataset_overview.py
========================
Figure 1 panels for dataset overview.

Panels saved as separate PDFs for PowerPoint assembly:
  fig1_note_panelA.txt        — reminder to make schematic in ChemDraw/PPT
  fig1b_dataset_composition.pdf — stereo class distribution colored by has_Tm
  fig1c_label_distribution.pdf  — has_Tm in train vs test
  fig1d_tsne.pdf                — t-SNE of X_fp_RU colored by train/test

Usage
-----
  # In your notebook, make sure these are loaded:
  #   df_cls        : full classification dataframe (643 rows)
  #   X_fp_RU       : repeat unit Morgan FP array (643, n_bits)
  #   trainval_idx  : list of trainval indices
  #   test_idx      : list of test indices

  %run fig1_dataset_overview.py

Requires: df_cls, X_fp_RU, trainval_idx, test_idx in notebook namespace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

from rdkit import Chem
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# COLORS — change to match your paper palette
# ══════════════════════════════════════════════════════════════════════════════

COLOR_HAS_TM    = "#2C7BB6"   # blue  — has Tm
COLOR_NO_TM     = "#D7191C"   # red   — no Tm
COLOR_TRAINVAL  = "#2C7BB6"   # blue  — train/val points in t-SNE
COLOR_TEST      = "#D7191C"   # red   — test points in t-SNE

STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#ABD9E9",
    "atactic":      "#FDAE61",
    "heterotactic": "#D7191C",
    "achiral":      "#1A9641",
    "unknown":      "#AAAAAA",
}

STEREO_ORDER = ["isotactic", "syndiotactic", "atactic",
                "heterotactic", "achiral"]

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
# PANEL A — placeholder note
# ══════════════════════════════════════════════════════════════════════════════

with open(f"{OUTDIR}/fig1_note_panelA.txt", "w") as f:
    f.write(
        "Figure 1 Panel A — Schematic\n"
        "=============================\n"
        "Make in ChemDraw or PowerPoint.\n\n"
        "Suggested content:\n"
        "  - Show the same polymer repeat unit in isotactic vs syndiotactic form\n"
        "  - Arrow: repeat unit SMILES → PBSG graph with stereo/regio annotations\n"
        "  - Highlight that stereo class is encoded as a graph-level feature\n"
        "  - Optional: show has_Tm label as a binary output node\n"
    )
print("Saved: fig1_note_panelA.txt")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — dataset stereo class composition colored by has_Tm
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset_composition(df):
    stereo_col = "stereo_class"
    label_col  = "has_Tm"

    # Count per stereo class, split by has_Tm
    rows = []
    for stereo in STEREO_ORDER:
        sub = df[df[stereo_col] == stereo]
        rows.append({
            "stereo":   stereo,
            "has_Tm":   sub[label_col].sum(),
            "no_Tm":    (~sub[label_col]).sum(),
            "total":    len(sub),
        })
    comp = pd.DataFrame(rows)

    x     = np.arange(len(STEREO_ORDER))
    width = 0.55
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    bars_has = ax.bar(x, comp["has_Tm"], width,
                      color=COLOR_HAS_TM, alpha=0.88,
                      label="Has T$_m$", edgecolor="white", lw=0.5)
    bars_no  = ax.bar(x, comp["no_Tm"], width,
                      bottom=comp["has_Tm"],
                      color=COLOR_NO_TM, alpha=0.88,
                      label="No T$_m$", edgecolor="white", lw=0.5)

    # Total n label above each bar
    for i, row in comp.iterrows():
        ax.text(i, row["total"] + 2, f"n={row['total']}",
                ha="center", va="bottom", fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STEREO_ORDER],
                       rotation=20, ha="right")
    ax.set_ylabel("Number of polymers")
    ax.set_title("Dataset Composition by Stereo Class",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, comp["total"].max() * 1.18)

    fig.tight_layout()
    path = f"{OUTDIR}/fig1b_dataset_composition.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — has_Tm label distribution: train vs test
# ══════════════════════════════════════════════════════════════════════════════

def plot_label_distribution(df, trainval_idx, test_idx):
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    df_train = df.iloc[trainval_idx]
    df_test  = df.iloc[test_idx]

    splits = {
        "Train+Val\n(n={})".format(len(df_train)): df_train,
        "Test\n(n={})".format(len(df_test)):       df_test,
    }

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    x     = np.arange(len(splits))
    width = 0.45

    for i, (label, sub) in enumerate(splits.items()):
        has_pct = 100 * sub["has_Tm"].mean()
        no_pct  = 100 - has_pct
        ax.bar(i, has_pct, width, color=COLOR_HAS_TM, alpha=0.88,
               edgecolor="white", lw=0.5)
        ax.bar(i, no_pct, width, bottom=has_pct,
               color=COLOR_NO_TM, alpha=0.88,
               edgecolor="white", lw=0.5)
        # Percentage labels
        ax.text(i, has_pct / 2, f"{has_pct:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(i, has_pct + no_pct / 2, f"{no_pct:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(splits.keys()), fontsize=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)
    ax.set_title("T$_m$ Label Distribution",
                 fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(facecolor=COLOR_HAS_TM, alpha=0.88, label="Has T$_m$"),
        mpatches.Patch(facecolor=COLOR_NO_TM,  alpha=0.88, label="No T$_m$"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=9,
              loc="upper right")

    fig.tight_layout()
    path = f"{OUTDIR}/fig1c_label_distribution.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — t-SNE colored by train/test split
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne(X, df, trainval_idx, test_idx,
              perplexity=30, random_state=42):
    """
    Run t-SNE on X_fp_RU and color by train/test.

    Parameters
    ----------
    X            : numpy array (n_samples, n_features) — X_fp_RU
    df           : full dataframe
    trainval_idx : list/array of trainval indices
    test_idx     : list/array of test indices
    """
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    print("  Running t-SNE (this takes ~1-2 minutes)...")

    # Standardize before t-SNE
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=perplexity,
            random_state=random_state, max_iter=1000,
            learning_rate="auto", init="pca")
    coords = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Plot trainval first (background), test on top
    ax.scatter(
        coords[trainval_idx, 0], coords[trainval_idx, 1],
        c=COLOR_TRAINVAL, s=18, alpha=0.55,
        edgecolors="none", zorder=2,
        label=f"Train+Val (n={len(trainval_idx)})",
    )
    ax.scatter(
        coords[test_idx, 0], coords[test_idx, 1],
        c=COLOR_TEST, s=18, alpha=0.85,
        edgecolors="white", linewidths=0.4, zorder=3,
        label=f"Test (n={len(test_idx)})",
    )

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("Chemical Space Coverage\n(Morgan FP, repeat unit)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9, loc="best",
              markerscale=1.5)

    # Remove all spines for t-SNE — standard practice
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = f"{OUTDIR}/fig1d_tsne.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return coords   # return coords in case you want to replot with
                    # different coloring without rerunning t-SNE


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL — expects these in notebook namespace:
#   df_cls, X_fp_RU, trainval_idx, test_idx
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # These must be loaded in your notebook before running
    try:
        _df   = df_cls
        _X    = X_fp_RU
        _tv   = trainval_idx
        _test = test_idx
    except NameError as e:
        print(f"Missing variable: {e}")
        print("Make sure df_cls, X_fp_RU, trainval_idx, test_idx are loaded.")
        raise

    plot_dataset_composition(_df)
    plot_label_distribution(_df, _tv, _test)
    tsne_coords = plot_tsne(_X, _df, _tv, _test)

    # Save t-SNE coords for replotting later without rerunning
    import numpy as np
    np.save(f"{OUTDIR}/tsne_coords_cls.npy", tsne_coords)
    print("Saved: tsne_coords_cls.npy  (reload to replot without rerunning t-SNE)")

    print("\nAll Figure 1 panels saved.")
    print("Panel A: make the schematic in ChemDraw/PowerPoint — see fig1_note_panelA.txt")