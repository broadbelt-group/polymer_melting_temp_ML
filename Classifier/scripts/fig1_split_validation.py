"""
fig1c_split_validation.py
=========================
Figure 1 Panel C — Split validation showing stratification worked.

Two subplots side by side:
  Left:  has_Tm % in Train+Val vs Test
  Right: Stereo class % in Train+Val vs Test

Saved as: fig1c_split_validation.pdf

Usage
-----
  from fig1c_split_validation import plot_split_validation
  plot_split_validation(df_cls, trainval_idx, test_idx)

Requires: df_cls, trainval_idx, test_idx in notebook namespace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

COLOR_TRAIN = "#2C7BB6"   # blue — train+val
COLOR_TEST  = "#D7191C"   # red  — test

COLOR_HAS_TM = "#2C7BB6"
COLOR_NO_TM  = "#D7191C"

STEREO_COLORS = {
    "isotactic":    "#2C7BB6",
    "syndiotactic": "#74ADD1",
    "atactic":      "#FDAE61",
    "stereo-irregular": "#D7191C",
    "achiral":      "#1A9641",
}
STEREO_ORDER = ["isotactic", "syndiotactic", "atactic", "stereo-irregular", "achiral"]

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


def plot_split_validation(df, trainval_idx, test_idx):
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    df = df.copy()
    df["copolymer_type"] = df["copolymer_type"].fillna("homopolymer")

    df_tv   = df.iloc[trainval_idx]
    df_test = df.iloc[test_idx]

    ARCH_ORDER = ["homopolymer", "random", "alternating", "block"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8),
                              gridspec_kw={"width_ratios": [1, 1.8, 1.8]})

    # ── LEFT: has_Tm stacked bar ───────────────────────────────────────────
    ax = axes[0]
    width = 0.45
    x     = np.arange(2)
    splits      = [df_tv, df_test]
    split_ns    = [len(df_tv), len(df_test)]

    for i, (sub, n) in enumerate(zip(splits, split_ns)):
        has_pct = 100 * sub["has_Tm"].mean()
        no_pct  = 100 - has_pct
        ax.bar(i, has_pct, width, color=COLOR_HAS_TM, alpha=0.88,
               edgecolor="white", lw=0.5)
        ax.bar(i, no_pct, width, bottom=has_pct,
               color=COLOR_NO_TM, alpha=0.88,
               edgecolor="white", lw=0.5)
        ax.text(i, has_pct / 2, f"{has_pct:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(i, has_pct + no_pct / 2, f"{no_pct:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s}\n(n={n})" for s, n in
         zip(["Train+Val", "Test"], split_ns)], fontsize=9.5)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("T$_m$ Label", fontsize=10, fontweight="bold")
    legend_patches = [
        mpatches.Patch(facecolor=COLOR_HAS_TM, alpha=0.88, label="Has T$_m$"),
        mpatches.Patch(facecolor=COLOR_NO_TM,  alpha=0.88, label="No T$_m$"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=8,
              loc="upper right")

    # ── MIDDLE: stereo class ───────────────────────────────────────────────
    ax = axes[1]
    n_stereo = len(STEREO_ORDER)
    bar_w    = 0.35
    x        = np.arange(n_stereo)

    for i, (sub, label, color) in enumerate(zip(
            [df_tv, df_test],
            [f"Train+Val (n={len(df_tv)})", f"Test (n={len(df_test)})"],
            [COLOR_TRAIN, COLOR_TEST])):
        pcts   = [100 * (sub["stereo_class"] == s).sum() / len(sub)
                  for s in STEREO_ORDER]
        offset = (i - 0.5) * bar_w
        bars   = ax.bar(x + offset, pcts, bar_w, color=color, alpha=0.85,
                        edgecolor="white", lw=0.5, label=label)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STEREO_ORDER],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Stereo Class", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(False)

    # ── RIGHT: copolymer architecture ──────────────────────────────────────
    ax = axes[2]
    n_arch = len(ARCH_ORDER)
    x      = np.arange(n_arch)

    for i, (sub, label, color) in enumerate(zip(
            [df_tv, df_test],
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
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Copolymer Architecture", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(False)

    fig.suptitle("Train/Test Split Validation",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = f"{OUTDIR}/fig1c_split_validation-2.pdf"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")