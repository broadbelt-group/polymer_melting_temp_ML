"""
SHAP Top-5 Feature Importance Analysis
=======================================
- Works with any scikit-learn tree-based model (RandomForest, XGBoost, GradientBoosting, etc.)
- Colors features by category on both bar and beeswarm plots
- Saves publication-ready figures
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# 1. FEATURE DEFINITIONS
# ──────────────────────────────────────────────

FP_NAMES    = [f"morgan_fp_{i}" for i in range(1024)]
DESC_NAMES  = [
    "MolWt", "LogP", "TPSA", "RotBonds", "Rings",
    "HBD", "HBA", "Heteroatoms", "NHOH",
    "AliphaticRings", "AmideBonds", "AromaticRings",
    "backbone_atoms", "sidechain_atoms", "avg_branch_length",
    "flexibility", "side_bulkiness_frac", "kier_flex",
]
ARCH_NAMES  = ["arch_alternating", "arch_block", "arch_random", "arch_homopolymer"]
STEREO_NAMES = [
    "stereo_isotactic", "stereo_syndiotactic", "stereo_stereo-irregular",
    "stereo_atactic", "stereo_achiral", "stereo_unknown",
]
SEQ_NAMES   = ["Pm", "Px", "asym"]
EDGE_NAMES  = ["edge_stereo"]
ALL_NAMES   = FP_NAMES + DESC_NAMES + ARCH_NAMES + STEREO_NAMES + SEQ_NAMES + EDGE_NAMES

# Auto-build feature → category mapping
feature_categories = (
    {f: "Morgan Fingerprint"   for f in FP_NAMES}
  | {f: "Node Feature" for f in DESC_NAMES}
  | {f: "Architecture"         for f in ARCH_NAMES}
  | {f: "Stereochemistry"      for f in STEREO_NAMES}
  | {f: "Global Feature"             for f in SEQ_NAMES}
  | {f: "Edge"                 for f in EDGE_NAMES}
)


# ──────────────────────────────────────────────
# 2. PLUG IN YOUR MODEL AND DATA
# ──────────────────────────────────────────────

if "model" not in globals() or "X" not in globals():
    raise NameError("Set `model` and `X` in your notebook before calling %run shap_analysis.py")


# mpl.rcParams['axes.prop_cycle'] = cycler(color=[
#     '#5c3c8b', '#92c36d', '#ee9432', '#496391', '#85a5cd', '#FDF3CC'
# ])
# ──────────────────────────────────────────────
# 3. CONFIGURATION
# ──────────────────────────────────────────────

TOP_N          = 5
INTERPRETABLE_ONLY = True
INTERPRETABLE_CATEGORIES = {"Node Feature", "Global Feature", "Stereochemistry", "Architecture", "Edge"} 
FIGURE_DIR     = "."
DPI            = 150
MAKE_BEESWARM  = False  # set True to generate the beeswarm plot

CATEGORY_COLORS = {
    "Morgan Fingerprint":   "#2196F3",   # blue
    "Node Feature": '#92c36d',   # green
    "Architecture":         "#FF9800",   # orange
    "Stereochemistry":      "#E91E63",   # pink
    "Global Feature":             '#5c3c8b',   # purple
    "Edge":                 "#00BCD4",   # cyan
}
DEFAULT_COLOR = "#78909C"   # grey fallback

# Optional: prettier display names for the bar chart
# Any feature not listed here will use its raw name as-is
DISPLAY_NAMES = {
    "MolWt":               "Molecular Weight",
    "LogP":                "LogP",
    "TPSA":                "TPSA",
    "RotBonds":            "Rotatable Bonds",
    "HBD":                 "H-Bond Donors",
    "HBA":                 "H-Bond Acceptors",
    "backbone_atoms":      "Backbone Length",
    "sidechain_atoms":     "Side Chain Length",
    "avg_branch_length":   "Branch Length",
    "flexibility":         "Flexibility",
    "side_bulkiness_frac": "Side Bulkiness Fraction",
    "kier_flex":           "Kier Flexibility",
    "Pm":                  "Pm",
    "Px":                  "Px",
    "asym":                "Asymmetry",
    "edge_stereo":         "Edge Stereo",
    # morgan fp bits — uncomment and name any that matter to you:
    # "morgan_fp_42":      "Aromatic Ring Bit",
}


# ──────────────────────────────────────────────
# 4. COMPUTE SHAP VALUES
# ──────────────────────────────────────────────

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X, columns=ALL_NAMES[:X.shape[1]])

print("Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X)

# Multi-class: average over classes
if shap_values.values.ndim == 3:
    sv             = np.abs(shap_values.values).mean(axis=2)
    shap_values_2d = shap_values.values.mean(axis=2)
else:
    sv             = np.abs(shap_values.values)
    shap_values_2d = shap_values.values

# ──────────────────────────────────────────────
# 5. IDENTIFY TOP-N FEATURES
# ──────────────────────────────────────────────

mean_abs_shap = sv.mean(axis=0)
feature_cols  = list(X.columns)

importance_df = pd.DataFrame({
    "feature":       feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "category":      [feature_categories.get(f, "Other") for f in feature_cols],
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("full_shap_importance_reg.csv", index=False)  # save full importance table 

if INTERPRETABLE_ONLY:
    ranked_df = importance_df[
        importance_df["category"].isin(INTERPRETABLE_CATEGORIES)
    ].reset_index(drop=True)
else:
    ranked_df = importance_df

top5 = ranked_df.head(TOP_N).reset_index(drop=True)
top5["color"] = top5["category"].map(CATEGORY_COLORS).fillna(DEFAULT_COLOR)

print(f"\nTop {TOP_N} features:")
print(top5[["feature", "category", "mean_abs_shap"]].to_string(index=False))

top5_idx    = [feature_cols.index(f) for f in top5["feature"]]
top5_names  = top5["feature"].tolist()
top5_colors = top5["color"].tolist()
present_cats = top5["category"].unique()

# ──────────────────────────────────────────────
# 6. PLOT 1 — BAR CHART
# ──────────────────────────────────────────────
 
fig, ax = plt.subplots(figsize=(6, 4))
 
names_rev  = top5_names[::-1]
values_rev = top5["mean_abs_shap"].values[::-1]
colors_rev = top5_colors[::-1]
 
bars = ax.barh(
    y         = range(TOP_N),
    width     = values_rev,
    color     = colors_rev,
    edgecolor = "white",
    height    = 0.6,
)
 
# Feature name inside each bar
x_max = values_rev.max()
for i, (bar, name, val) in enumerate(zip(bars, names_rev, values_rev)):
    label = DISPLAY_NAMES.get(name, name)  
    ax.text(val * 0.03, i, label, va="center", ha="left",
            fontsize=10, fontweight="bold", color="white",
            clip_on=False)
 
ax.set_xlabel(r"Mean SHAP Value, ($T_m$)", fontsize=11) #, ($^{\circ}$C)
#ax.set_title(f"Top {TOP_N} Most Important Features", fontsize=13, fontweight="bold")
ax.set_yticks([])                   # hide y-axis ticks — names are on bars
#ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlim(0, x_max * 1.1)       # give room for outside labels
 
legend_handles = [
    mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
    for c in present_cats
]
# ax.legend(handles=legend_handles, fontsize=9,
#           title_fontsize=9, loc="lower right")
 
plt.tight_layout()
bar_path = f"{FIGURE_DIR}/shap_top{TOP_N}_bar_reg.png"
plt.savefig(bar_path, dpi=DPI, bbox_inches="tight")
plt.show()
print(f"Saved: {bar_path}")

# ──────────────────────────────────────────────
# LEGEND — save as separate image
# ──────────────────────────────────────────────

fig_leg, ax_leg = plt.subplots(figsize=(3, len(CATEGORY_COLORS) * 0.4 + 0.3))
ax_leg.axis("off")

legend_handles = [
    mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
    for c in present_cats        # swap CATEGORY_COLORS.items() for this
]
ax_leg.legend(
    handles       = legend_handles,
    #title         = "Feature Category",
    title_fontsize = 10,
    fontsize      = 10,
    loc           = "center",
    frameon       = False,
    edgecolor     = "#cccccc",
    ncol=2
)

plt.tight_layout()
legend_path = f"{FIGURE_DIR}/shap_category_legend.png"
plt.savefig(legend_path, dpi=DPI, bbox_inches="tight")
plt.show()
print(f"Saved: {legend_path}")


# ──────────────────────────────────────────────
# 7. PLOT 2 — BEESWARM
# ──────────────────────────────────────────────

if MAKE_BEESWARM:
    shap_top5 = shap.Explanation(
        values        = shap_values_2d[:, top5_idx],
        base_values   = (shap_values.base_values
                         if shap_values.base_values.ndim == 1
                         else shap_values.base_values[:, 0]),
        data          = X.iloc[:, top5_idx].values,
        feature_names = top5_names,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(shap_top5, show=False, color_bar=True)

    ax = plt.gca()
    for label_obj in ax.get_yticklabels():
        fname = label_obj.get_text()
        cat   = feature_categories.get(fname, "Other")
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        label_obj.set_color(color)
        label_obj.set_fontweight("bold")

    ax.set_title(f"SHAP Beeswarm — Top {TOP_N} Features", fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
        for c in present_cats
    ]
    ax.legend(handles=legend_handles, title="Category", fontsize=9,
              title_fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    beeswarm_path = f"{FIGURE_DIR}/shap_top{TOP_N}_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    print(f"Saved: {beeswarm_path}")

# ──────────────────────────────────────────────
# 8. EXPORT SUMMARY TABLE
# ──────────────────────────────────────────────

csv_path = f"{FIGURE_DIR}/shap_top{TOP_N}_summary_reg.csv"
top5[["feature", "category", "mean_abs_shap"]].to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
