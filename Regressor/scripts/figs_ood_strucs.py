"""
figS_ood_structures.py
======================
SI Figure — OOD polymer structures drawn with RDKit,
annotated with DoV scores and reliability tiers.

Also includes regression t-SNE colored by Tm value.

Saves:
  figS_ood_structures.pdf  — grid of OOD polymer structures
  fig1d_tsne_reg_tm.pdf    — regression t-SNE colored by Tm

Usage
-----
  from figS_ood_structures import plot_ood_structures, plot_tsne_reg_tm

  # Classification OOD structures (in classifier notebook):
  plot_ood_structures(result_df, df_cls, test_idx)

  # Regression t-SNE (in regression notebook):
  # First run t-SNE:
  tsne_coords_reg = plot_tsne_reg_tm(X_fp_RU_reg, df_reg)
  # Or reload saved coords:
  plot_tsne_reg_tm(coords=np.load("tsne_coords_reg.npy"), df_reg=df_reg)

Requires:
  rdkit, sklearn, matplotlib
  result_df       : dov.score_df() output (classification)
  df_cls          : classification dataframe
  test_idx        : test indices
  X_fp_RU_reg     : repeat unit FP array for regression dataset
  df_reg          : regression dataframe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

TIER_COLORS = {
    "Very low": "#D7191C",
    "Low":      "#FDAE61",
    "Moderate": "#ABD9E9",
    "High":     "#2C7BB6",
}

TM_CMAP = "plasma"   # colormap for Tm values in t-SNE

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "pdf.fonttype":   42,
    "svg.fonttype":   "none",
})

DPI    = 300
OUTDIR = "."


# ══════════════════════════════════════════════════════════════════════════════
# OOD POLYMER STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

def _smiles_to_img(smiles, size=(300, 200)):
    """Draw a SMILES string to a PIL image."""
    # Use first repeat unit for copolymers
    first_unit = smiles.split(".")[0].strip()
    # Remove polymer end group markers [*]
    clean = first_unit.replace("[*]", "*")
    mol   = Chem.MolFromSmiles(clean)
    if mol is None:
        # Try original
        mol = Chem.MolFromSmiles(first_unit)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Convert SVG to PIL image via cairosvg if available, else use PNG drawer
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode())
        return Image.open(BytesIO(png_data))
    except ImportError:
        # Fallback: use PNG drawer
        drawer2 = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer2.drawOptions().addStereoAnnotation = True
        drawer2.DrawMolecule(mol)
        drawer2.FinishDrawing()
        png_data = drawer2.GetDrawingText()
        return Image.open(BytesIO(png_data))


def plot_ood_structures(result_df, df_cls, test_idx,
                        outpath=None):
    """
    Draw structures of OOD test polymers with DoV scores.

    Parameters
    ----------
    result_df : dov.score_df() output — must have in_domain_chem,
                dov_score, reliability columns
    df_cls    : full classification dataframe
    test_idx  : list/array of test indices
    """
    test_idx = np.array(test_idx)

    # Get OOD rows
    ood_mask = ~result_df["in_domain_chem"].values
    ood_df   = result_df[ood_mask].reset_index(drop=True)
    ood_idx  = test_idx[ood_mask]

    # Get polymer info from df_cls
    ood_info = df_cls.iloc[ood_idx][
        ["repeat_units", "stereo_class", "has_Tm"]
    ].reset_index(drop=True)
    ood_info["dov_score"]   = ood_df["dov_score"].values
    ood_info["reliability"] = ood_df["reliability"].astype(str).values

    n_ood = len(ood_info)
    n_cols = min(3, n_ood)
    n_rows = (n_ood + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.8, n_rows * 3.2))
    if n_ood == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for idx, row in ood_info.iterrows():
        ri   = idx // n_cols
        ci   = idx % n_cols
        ax   = axes[ri][ci]
        smi  = row["repeat_units"]
        tier = row["reliability"]
        color = TIER_COLORS.get(tier, "#999999")

        img = _smiles_to_img(smi)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Could not\ndraw structure",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="grey")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        stereo = str(row["stereo_class"]).capitalize()
        has_tm = "Has T$_m$" if row["has_Tm"] else "No T$_m$"
        title  = (f"{stereo} | {has_tm}\n"
                  f"DoV={row['dov_score']:.3f} | {tier}")
        ax.set_title(title, fontsize=8, color=color, fontweight="bold",
                     pad=4)

    # Hide empty subplots
    for idx in range(n_ood, n_rows * n_cols):
        ri = idx // n_cols
        ci = idx % n_cols
        axes[ri][ci].set_visible(False)

    # Legend for tier colors
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, label=t)
        for t, c in TIER_COLORS.items()
        if t in ood_info["reliability"].values
    ]
    fig.legend(handles=patches, frameon=False, fontsize=8.5,
               loc="lower center", ncol=len(patches),
               title="Reliability tier", title_fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Out-of-Domain Test Polymers (n={n_ood})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = outpath or f"{OUTDIR}/figS_ood_structures.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION t-SNE COLORED BY Tm
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne_reg_tm(X_fp=None, df_reg=None,
                     coords=None,
                     perplexity=30, random_state=42,
                     coords_save_path="tsne_coords_reg.npy"):
    """
    t-SNE of regression dataset colored by Tm value (continuous colormap).

    Parameters
    ----------
    X_fp       : numpy array (n_samples, n_features) — repeat unit FP
                 Required if coords is None.
    df_reg     : regression dataframe with "Tm" column
    coords     : pre-computed t-SNE coords (n, 2) — skip recomputing if provided
    coords_save_path : where to save coords for reuse

    Returns
    -------
    coords : (n, 2) t-SNE coordinates
    """
    if coords is None:
        if X_fp is None:
            raise ValueError("Either X_fp or coords must be provided.")
        print("  Running t-SNE on regression dataset (~1-2 min)...")
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_fp)
        tsne    = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state, max_iter=1000,
                       learning_rate="auto", init="pca")
        coords  = tsne.fit_transform(X_sc)
        np.save(coords_save_path, coords)
        print(f"  Saved coords → {coords_save_path}")

    tm     = df_reg["Tm"].values
    stereo = df_reg["stereo_class"].fillna("unknown").values

    # ── Plot 1: colored by Tm ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    norm = mcolors.Normalize(vmin=tm.min(), vmax=tm.max())
    cmap = cm.get_cmap(TM_CMAP)
    sc   = ax.scatter(coords[:, 0], coords[:, 1],
                      c=tm, cmap=TM_CMAP, norm=norm,
                      s=22, alpha=0.85,
                      edgecolors="none", zorder=2)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("T$_m$ (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("Chemical Space Colored by T$_m$\n"
                 "(Morgan FP, repeat unit)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path1 = f"{OUTDIR}/fig1d_tsne_reg_tm.pdf"
    fig.savefig(path1, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Plot 2: colored by stereo class ──────────────────────────────────────
    STEREO_COLORS = {
        "isotactic":    "#2C7BB6",
        "syndiotactic": "#74ADD1",
        "atactic":      "#FDAE61",
        "heterotactic": "#D7191C",
        "achiral":      "#1A9641",
        "unknown":      "#AAAAAA",
    }
    STEREO_ORDER = ["isotactic", "syndiotactic", "atactic",
                    "heterotactic", "achiral"]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    for s in STEREO_ORDER:
        mask = stereo == s
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=STEREO_COLORS[s], s=20, alpha=0.80,
                   edgecolors="none", zorder=2,
                   label=s.capitalize())

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("Chemical Space — Stereo Class\n(Regression dataset)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [plt.scatter([], [], c=STEREO_COLORS[s], s=30,
                           label=s.capitalize())
               for s in STEREO_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="best")

    fig.tight_layout()
    path2 = f"{OUTDIR}/fig1d_tsne_reg_stereo.pdf"
    fig.savefig(path2, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    return coords