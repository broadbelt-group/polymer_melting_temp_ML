"""
fig_shap_importance.py
======================
XGBoost gain-based feature importance for best models.
No SHAP required — uses XGBoost's built-in gain importance.

Usage
-----
  from fig_shap_importance import plot_shap_importance

  # Classification
  plot_shap_importance(
      model   = results["xgb_poly"]["clf"],
      rep     = "poly",
      task    = "cls",
      outpath = "figures/fig_importance_cls.pdf",
  )

  # Regression
  import joblib
  reg_xgb_ru = joblib.load("saved_models_top/reg_xgb_pooled_ru.pkl")
  plot_shap_importance(
      model   = reg_xgb_ru,
      rep     = "ru",
      task    = "reg",
      outpath = "figures/fig_importance_reg.pdf",
  )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
})

DPI = 300

# ── Feature name definitions ──────────────────────────────────────────────────

FP_NAMES = [f"morgan_fp_{i}" for i in range(1024)]

DESC_NAMES = [
    "MolWt", "LogP", "TPSA", "RotBonds", "Rings",
    "HBD", "HBA", "Heteroatoms", "NHOH",
    "AliphaticRings", "AmideBonds", "AromaticRings",
    "backbone_atoms", "sidechain_atoms", "avg_branch_length",
    "flexibility", "side_bulkiness_frac", "kier_flex",
]
ARCH_NAMES = ["arch_alternating", "arch_block", "arch_random", "arch_homopolymer"]
STEREO_NAMES = [
    "stereo_isotactic", "stereo_syndiotactic", "stereo_heterotactic",
    "stereo_atactic", "stereo_achiral", "stereo_unknown",
]
SEQ_NAMES  = ["Pm", "Px", "asym"]
EDGE_NAMES = ["edge_stereo"]

ALL_NAMES = FP_NAMES + DESC_NAMES + ARCH_NAMES + STEREO_NAMES + SEQ_NAMES + EDGE_NAMES

GROUPS = [
    ("Morgan FP\n(1024)",   FP_NAMES),
    ("Descriptors\n(18)",   DESC_NAMES),
    ("Architecture\n(4)",   ARCH_NAMES),
    ("Stereo class\n(6)",   STEREO_NAMES),
    ("Pm/Px/asym\n(3)",     SEQ_NAMES),
    ("Edge stereo\n(1)",    EDGE_NAMES),
]

GROUP_COLORS = ["#AEC6E8", "#2CA02C", "#FF7F0E", "#C5407E", "#9467BD", "#8C564B"]


def _clean_name(name: str) -> str:
    name = name.replace("stereo_", "Stereo: ")
    name = name.replace("arch_",   "Arch: ")
    name = name.replace("morgan_fp_", "FP bit ")
    name = name.replace("_", " ")
    return name.capitalize()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_importance(
    model,
    X_test  = None,
    rep:     str = "poly",
    task:    str = "cls",
    outpath: str = "fig_importance.pdf",
    n_top:   int = 20,
):
    # ── 1. Compute gain importance ────────────────────────────────────────────
    print("  Computing XGBoost feature importance (normalized gain)...")
    score_dict = model.get_booster().get_score(importance_type="gain")

    n_feat    = model.n_features_in_
    mean_gain = np.zeros(n_feat)
    for fname, val in score_dict.items():
        idx = int(fname[1:])
        mean_gain[idx] = val

    total = mean_gain.sum()
    if total > 0:
        mean_gain = mean_gain / total

    if n_feat == len(ALL_NAMES):
        imp = pd.Series(mean_gain, index=ALL_NAMES)
    else:
        print(f"  Warning: {n_feat} features vs {len(ALL_NAMES)} names — using generic names")
        imp = pd.Series(mean_gain, index=[f"feat_{i}" for i in range(n_feat)])

    # ── 2. Left panel data: top N non-FP features ─────────────────────────────
    non_fp = imp[~imp.index.str.startswith("morgan_fp_")]
    top_n  = non_fp.sort_values(ascending=False).head(n_top)

    # ── 3. Right panel data: group importance normalized by group size ─────────
    group_data = []
    for label, feat_list in GROUPS:
        present   = [f for f in feat_list if f in imp.index]
        mean_val  = imp[present].sum() / len(feat_list) if present else 0.0
        group_data.append((label, float(mean_val)))

    grp_labels = [g[0] for g in group_data]
    grp_vals   = np.array([g[1] for g in group_data])

    # ── 4. Style ──────────────────────────────────────────────────────────────
    bar_color  = "#7B5EA7" if task == "cls" else "#2C7BB6"
    title_task = "Classification" if task == "cls" else "Regression"
    rep_label  = "PolyFP+pooled" if rep == "poly" else "RepFP+pooled"

    # ── 5. Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.8, 1.6], "wspace": 0.42}
    )

    # Left: top N non-FP
    ax = axes[0]
    ax.barh(range(len(top_n)), top_n.values,
            color=bar_color, alpha=0.88, edgecolor="none")
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels([_clean_name(n) for n in top_n.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Gain Importance", fontsize=9)
    ax.set_title(f"Top {n_top} Non-FP Features", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.3f}")
    )

    # Right: group importance
    ax = axes[1]
    ax.bar(range(len(grp_labels)), grp_vals,
           color=GROUP_COLORS[:len(grp_labels)],
           alpha=0.88, edgecolor="none", width=0.6)

    clean_labels = [l.replace("\n", " ") for l in grp_labels]
    ax.set_xticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean gain per feature\n(normalized by group size)", fontsize=8.5)
    ax.set_title("Feature Group\nImportance", fontsize=10, fontweight="bold")
    ax.set_ylim(0, grp_vals.max() * 1.25)

    for i, v in enumerate(grp_vals):
        if v > grp_vals.max() * 0.02:
            ax.text(i, v + grp_vals.max() * 0.03,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7.5, color="dimgrey")

    fig.suptitle(
        f"Feature Importance (Normalized Gain) — XGB ({rep_label}), {title_task}",
        fontsize=11, fontweight="bold", y=1.01
    )

    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # Summary
    print(f"\n  Feature group summary (mean gain per feature):")
    for label, val in group_data:
        print(f"    {label.replace(chr(10), ' '):<28s}: {val:.6f}")

    return imp

def plot_top20_all_features(
    model,
    rep:     str = "poly",
    task:    str = "cls",
    outpath: str = "fig_importance_top20.pdf",
    n_top:   int = 20,
):
    """Top N features including Morgan FP bits."""

    # ── Compute gain importance ───────────────────────────────────────────────
    score_dict = model.get_booster().get_score(importance_type="gain")
    n_feat     = model.n_features_in_
    mean_gain  = np.zeros(n_feat)
    for fname, val in score_dict.items():
        mean_gain[int(fname[1:])] = val

    total = mean_gain.sum()
    if total > 0:
        mean_gain = mean_gain / total

    if n_feat == len(ALL_NAMES):
        imp = pd.Series(mean_gain, index=ALL_NAMES)
    else:
        imp = pd.Series(mean_gain,
                        index=[f"feat_{i}" for i in range(n_feat)])

    top_n = imp.sort_values(ascending=False).head(n_top)

    # ── Color bars by group ───────────────────────────────────────────────────
    group_color_map = {}
    for (label, feat_list), color in zip(GROUPS, GROUP_COLORS):
        for f in feat_list:
            group_color_map[f] = color

    colors = [group_color_map.get(n, "#AAAAAA") for n in top_n.index]

    # ── Plot ──────────────────────────────────────────────────────────────────
    bar_color  = "#7B5EA7" if task == "cls" else "#2C7BB6"
    title_task = "Classification" if task == "cls" else "Regression"
    rep_label  = "PolyFP+pooled" if rep == "poly" else "RepFP+pooled"

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.barh(range(len(top_n)), top_n.values,
            color=colors, alpha=0.88, edgecolor="none")
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels([_clean_name(n) for n in top_n.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Gain Importance", fontsize=9)
    ax.set_title(
        f"Top {n_top} Features — XGB ({rep_label}), {title_task}",
        fontsize=10, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.3f}")
    )

    # ── Legend — show only groups present in top N ────────────────────────────
    import matplotlib.patches as mpatches
    present_colors = set(group_color_map.get(n) for n in top_n.index)
    legend_patches = [
        mpatches.Patch(facecolor=color, alpha=0.88,
                       label=label.replace("\n", " "))
        for (label, feat_list), color in zip(GROUPS, GROUP_COLORS)
        if color in present_colors
    ]
    ax.legend(handles=legend_patches, frameon=False,
              fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return imp