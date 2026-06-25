"""
fig1d_tsne_replot.py
====================
Replot t-SNE with different colorings without rerunning t-SNE.
Loads saved coords from tsne_coords_cls.npy.

Saves:
  fig1d_tsne_split.pdf        — colored by train/test
  fig1d_tsne_stereo.pdf       — colored by stereo class
  fig1d_tsne_polymerclass.pdf — colored by polymer class (SMARTS-derived)
  fig1d_tsne_hasTm.pdf        — colored by has_Tm label

Usage
-----
  # In notebook:
  from fig1d_tsne_replot import replot_all_tsne
  replot_all_tsne(df_cls, trainval_idx, test_idx)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════════════
# POLYMER CLASS SMARTS (same as splitting script)
# ══════════════════════════════════════════════════════════════════════════════

_POLYMER_CLASS_SMARTS = [
    ("polyamide",     ["[NX3H][CX3](=O)", "[NX3]([CX3](=O))[CX3](=O)"]),
    ("polyester",     ["[OX2][CX3](=O)[!N]", "[CX3](=O)[OX2][CX3](=O)"]),
    ("polycarbonate", ["[OX2][CX3](=O)[OX2]"]),
    ("polyurethane",  ["[NX3H][CX3](=O)[OX2]", "[OX2][CX3](=O)[NX3H]"]),
    ("polyether",     ["[CX4][OX2][CX4]", "[cX3][OX2][CX4]", "[CX4][OX2][cX3]"]),
    ("polyolefin",    ["[CX4H2][CX4H2]", "[CX4H2][CX4H]([*])",
                       "[CX3H]=[CX3H2]", "[CX3H]=[CX3H]"]),
    ("polysiloxane",  ["[Si][OX2][Si]"]),
]
_FALLBACK_CLASS = "other"


def _get_polymer_class(smiles, patterns):
    units = [s.strip() for s in smiles.split(".") if s.strip()]
    unit_classes = []
    for unit in units:
        mol = Chem.MolFromSmiles(unit)
        if mol is None:
            unit_classes.append(_FALLBACK_CLASS)
            continue
        assigned = _FALLBACK_CLASS
        for cls, pats in patterns:
            if any(mol.HasSubstructMatch(p) for p in pats if p is not None):
                assigned = cls
                break
        unit_classes.append(assigned)
    if not unit_classes:
        return _FALLBACK_CLASS
    counts = Counter(unit_classes)
    top_n = counts.most_common(1)[0][1]
    top   = [c for c, n in counts.items() if n == top_n]
    return top[0] if len(top) == 1 else "mixed"


def assign_polymer_classes(df):
    patterns = []
    for cls, smarts_list in _POLYMER_CLASS_SMARTS:
        pats = [Chem.MolFromSmarts(s) for s in smarts_list]
        patterns.append((cls, pats))
    return df["repeat_units"].fillna("").apply(
        lambda s: _get_polymer_class(s, patterns)
    )


# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#ABD9E9",
    "atactic":      "#FDAE61",
    "stereo-irregular": "#D7191C",
    "achiral":      "#1A9641",
    "unknown":      "#AAAAAA",
}

POLYMER_CLASS_COLORS = {
    "polyolefin":    "#2C7BB6",
    "polyester":     "#D7191C",
    "polyether":     "#1A9641",
    "polyamide":     "#FF7F00",
    "polycarbonate": "#984EA3",
    "polyurethane":  "#A65628",
    "polysiloxane":  "#F781BF",
    "mixed":         "#999999",
    "other":         "#DDDDDD",
}

HASTM_COLORS = {
    True:  "#2C7BB6",
    False: "#D7191C",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "pdf.fonttype":   42,
    "svg.fonttype":   "none",
})

DPI    = 300
OUTDIR = "Figures"


# ══════════════════════════════════════════════════════════════════════════════
# CORE PLOT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _tsne_scatter(coords, colors, labels_map, title, outpath,
                  point_size=18, alpha=0.75, order=None):
    """
    Generic t-SNE scatter.

    Parameters
    ----------
    coords     : (n, 2) array of t-SNE coordinates
    colors     : list of color strings, one per sample
    labels_map : dict of {value: color} for legend
    title      : plot title
    outpath    : save path
    order      : list of keys in labels_map for legend order
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=point_size, alpha=alpha,
               edgecolors="none", zorder=2)

    # Legend
    order = order or list(labels_map.keys())
    patches = [
        mpatches.Patch(facecolor=labels_map[k], alpha=0.9,
                       label=str(k).capitalize())
        for k in order if k in labels_map
    ]
    ax.legend(handles=patches, frameon=False, fontsize=8.5,
              loc="best", markerscale=1.5)

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# FOUR COLORING OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne_split(coords, df, trainval_idx, test_idx):
    """Color by train/test split."""
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)
    colors = ["#2C7BB6"] * len(df)
    for i in test_idx:
        colors[i] = "#D7191C"

    labels_map = {
        f"Train+Val (n={len(trainval_idx)})": "#2C7BB6",
        f"Test (n={len(test_idx)})":          "#D7191C",
    }
    _tsne_scatter(coords, colors, labels_map,
                  "Chemical Space — Train/Test Split",
                  f"{OUTDIR}/fig1d_tsne_split.pdf")


def plot_tsne_stereo(coords, df):
    """Color by stereo class."""
    stereo = df["stereo_class"].fillna("unknown").values
    colors = [STEREO_COLORS.get(s, "#AAAAAA") for s in stereo]

    order = ["isotactic", "syndiotactic", "atactic", "stereo-irregular", "achiral"]
    _tsne_scatter(coords, colors, STEREO_COLORS,
                  "Chemical Space — Stereo Class",
                  f"{OUTDIR}/fig1d_tsne_stereo.pdf",
                  order=order)


def plot_tsne_polymer_class(coords, df):
    """Color by SMARTS-derived polymer class."""
    print("  Assigning polymer classes from SMARTS...")
    poly_cls = assign_polymer_classes(df).values
    colors   = [POLYMER_CLASS_COLORS.get(c, "#DDDDDD") for c in poly_cls]

    # Only include classes that appear in the data
    present = sorted(set(poly_cls),
                     key=lambda x: -list(poly_cls).count(x))
    _tsne_scatter(coords, colors, POLYMER_CLASS_COLORS,
                  "Chemical Space — Polymer Class",
                  f"{OUTDIR}/fig1d_tsne_polymerclass.pdf",
                  order=present)


def plot_tsne_hasTm(coords, df):
    """Color by has_Tm label."""
    labels = df["has_Tm"].values
    colors = [HASTM_COLORS[bool(l)] for l in labels]
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos

    labels_map = {
        f"Has T$_m$ (n={n_pos})":  "#2C7BB6",
        f"No T$_m$ (n={n_neg})":   "#D7191C",
    }
    _tsne_scatter(coords, colors, labels_map,
                  "Chemical Space — T$_m$ Label",
                  f"{OUTDIR}/fig1d_tsne_hasTm.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════

def replot_all_tsne(df, trainval_idx, test_idx,
                    coords_path="tsne_coords_cls.npy"):
    """
    Replot t-SNE with all four colorings.

    Parameters
    ----------
    df           : df_cls (643 rows)
    trainval_idx : list/array of trainval indices
    test_idx     : list/array of test indices
    coords_path  : path to saved tsne_coords_cls.npy
    """
    coords = np.load(coords_path)
    print(f"Loaded t-SNE coords: {coords.shape}")

    plot_tsne_split(coords, df, trainval_idx, test_idx)
    plot_tsne_stereo(coords, df)
    plot_tsne_polymer_class(coords, df)
    plot_tsne_hasTm(coords, df)

    print("\nAll t-SNE panels saved.")