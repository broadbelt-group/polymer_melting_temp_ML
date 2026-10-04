
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. FEATURE DEFINITIONS

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


# 2. PLUG IN YOUR MODEL AND DATA

if "model" not in globals() or "X" not in globals():
    raise NameError("Set `model` and `X` in your notebook before calling %run shap_analysis.py")


# 3. CONFIGURATION

TOP_N          = 5
INTERPRETABLE_ONLY = True
INTERPRETABLE_CATEGORIES = {"Node Feature", "Global Feature", "Stereochemistry", "Architecture", "Edge"} 
FIGURE_DIR     = "."
DPI            = 150
MAKE_BEESWARM  = True  # set True to generate the beeswarm plot

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


# 4. COMPUTE SHAP VALUES

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

# 5. IDENTIFY TOP-N FEATURES

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

# 6. PLOT 1 — BAR CHART
 
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
 
ax.set_xlabel(r"Mean Absolute SHAP Value, ($T_m$)", fontsize=11)
ax.set_yticks([])                   # hide y-axis ticks — names are on bars
ax.set_xlim(0, x_max * 1.1)       # give room for outside labels

legend_handles = [
    mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
    for c in present_cats
]
 
plt.tight_layout()
bar_path = f"{FIGURE_DIR}/shap_top{TOP_N}_bar_reg.png"
plt.savefig(bar_path, dpi=DPI, bbox_inches="tight")
plt.show()
print(f"Saved: {bar_path}")

# LEGEND — save as separate image

fig_leg, ax_leg = plt.subplots(figsize=(3, len(CATEGORY_COLORS) * 0.4 + 0.3))
ax_leg.axis("off")

legend_handles = [
    mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
    for c in present_cats        # swap CATEGORY_COLORS.items() for this
]
ax_leg.legend(
    handles       = legend_handles,
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


# 7. PLOT 2 — BEESWARM

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

    #ax.set_title(f"SHAP Beeswarm — Top {TOP_N} Features", fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, DEFAULT_COLOR), label=c)
        for c in present_cats
    ]
    # ax.legend(handles=legend_handles, title="Category", fontsize=9,
    #           title_fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    beeswarm_path = f"{FIGURE_DIR}/shap_top{TOP_N}_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    print(f"Saved: {beeswarm_path}")

# 8. EXPORT SUMMARY TABLE

csv_path = f"{FIGURE_DIR}/shap_top{TOP_N}_summary_reg.csv"
top5[["feature", "category", "mean_abs_shap"]].to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── DIRECTIONAL SHAP (for SI, answers R2 comment 5) ──────────────────────────
# For each top feature: correlation between feature VALUE and its SHAP value.
# Positive corr -> higher feature value increases Tm; negative -> decreases Tm.

from scipy.stats import spearmanr

direction_rows = []
for f in top5_names:
    j = feature_cols.index(f)
    fval = X.iloc[:, j].values
    sval = shap_values_2d[:, j]          # SIGNED shap, not abs
    # guard against constant features
    if np.std(fval) < 1e-12:
        rho, direction = np.nan, "constant"
    else:
        rho, _ = spearmanr(fval, sval)
        direction = "↑ increases $T_m$" if rho > 0 else "↓ decreases $T_m$"
    direction_rows.append({
        "feature": DISPLAY_NAMES.get(f, f),
        "category": feature_categories.get(f, "Other"),
        "mean_abs_shap": float(sv[:, j].mean()),
        "value_shap_corr": rho,          # signed: the directionality R2 wants
        "direction": direction,
    })

direction_df = pd.DataFrame(direction_rows)
print("\nDirectional SHAP summary (feature value ↔ SHAP correlation):")
print(direction_df.to_string(index=False))
direction_df.to_csv(f"{FIGURE_DIR}/shap_top{TOP_N}_directional_reg.csv", index=False)

# ── DIRECTIONAL BAR — centered at 0, same top-5, same colors ─────────────────
from scipy.stats import spearmanr

# compute signed direction for each top feature (magnitude = importance)
signed_vals, colors_ord, labels_ord = [], [], []
for f in top5_names:
    j    = feature_cols.index(f)
    fval = X.iloc[:, j].values
    sval = shap_values_2d[:, j]
    imp  = sv[:, j].mean()                       # magnitude = mean|SHAP| (matches main fig)
    if np.std(fval) < 1e-12:
        sign = 0.0
    else:
        rho, _ = spearmanr(fval, sval)
        sign = np.sign(rho)
    signed_vals.append(sign * imp)               # signed length
    colors_ord.append(CATEGORY_COLORS.get(feature_categories.get(f, "Other"), DEFAULT_COLOR))
    labels_ord.append(DISPLAY_NAMES.get(f, f))

# reverse so most important is on top
signed_rev = signed_vals[::-1]
colors_rev = colors_ord[::-1]
labels_rev = labels_ord[::-1]

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.barh(range(TOP_N), signed_rev, color=colors_rev, edgecolor="white", height=0.6)

ax.axvline(0, color="#333333", lw=1)             # zero line

xmax = max(abs(v) for v in signed_rev)
for i, (val, label) in enumerate(zip(signed_rev, labels_rev)):
    # place label on the opposite side of the bar direction so it never overlaps
    if val >= 0:
        ax.text(-xmax*0.03, i, label, va="center", ha="right",
                fontsize=10, fontweight="bold", color="#222222", clip_on=False)
    else:
        ax.text( xmax*0.03, i, label, va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222222", clip_on=False)

ax.set_xlim(-xmax*1.25, xmax*1.25)
ax.set_yticks([])
ax.set_xlabel(r"Directional SHAP effect (← decreases   |   increases →)", fontsize=11)
# optional: label the two sides
# ax.text(0.98, 1.02, "increases $T_m$", transform=ax.transAxes, ha="right", fontsize=9, color="#555")
# ax.text(0.02, 1.02, "decreases $T_m$", transform=ax.transAxes, ha="left",  fontsize=9, color="#555")

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/shap_top{TOP_N}_directional_centered_reg.png", dpi=DPI, bbox_inches="tight")
plt.show()
