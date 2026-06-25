
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

DPI    = 300
OUTDIR = "Figures"

MODEL_LABELS_CLS = {
    "gine_pbsg":  "GINE (PBSG)",
    "gatv2_pbsg": "GATv2 (PBSG)",
    "xgb_ru":     None,   # no history for XGB
    "xgb_poly":   None,
}

MODEL_COLORS_CLS = {
    "gine_pbsg":  "#2C7BB6",
    "gatv2_pbsg": "#74ADD1",
}


def _plot_learning_curve(ax, history, label, color,
                          train_key, val_key, val_label):
    epochs = range(1, len(history[train_key]) + 1)
    ax.plot(epochs, history[train_key], color=color, lw=1.5,
            linestyle="--", alpha=0.7, label=f"{label} train")
    ax.plot(epochs, history[val_key], color=color, lw=2.0,
            label=f"{label} val ({val_label})")


# CLASSIFICATION

def plot_learning_curves_cls(path="test_training_histories.json"):
    with open(path) as f:
        histories = json.load(f)

    # Filter to GNN models only
    gnn_keys = [k for k in histories
                if MODEL_LABELS_CLS.get(k) is not None]

    if not gnn_keys:
        print("No GNN histories found in test_training_histories.json")
        return

    n = len(gnn_keys)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, mk in zip(axes, gnn_keys):
        h     = histories[mk]
        color = MODEL_COLORS_CLS.get(mk, "#333333")
        label = MODEL_LABELS_CLS[mk]

        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color=color, lw=1.5,
                linestyle="--", alpha=0.7, label="Train loss")
        ax.plot(epochs, h["val_loss"], color=color, lw=2.0,
                label="Val loss")

        # Mark best epoch (min val_loss)
        best_ep = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best_ep, color="grey", lw=1, linestyle=":",
                   alpha=0.7, label=f"Best epoch ({best_ep})")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss (BCE)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("GNN Training Curves — Classification (Test Retraining)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{OUTDIR}/figS_learning_curve_cls.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# REGRESSION

def plot_learning_curves_reg(path="test_training_histories_reg_full.json"):
    with open(path) as f:
        histories = json.load(f)

    gnn_keys = [k for k in histories]
    if not gnn_keys:
        print("No GNN histories found in test_training_histories_reg_full.json")
        return

    n = len(gnn_keys)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    if n == 1:
        axes = [axes]

    color = "#2C7BB6"
    for ax, mk in zip(axes, gnn_keys):
        h      = histories[mk]
        epochs = range(1, len(h["train_loss"]) + 1)

        ax2 = ax.twinx()

        # Loss on left axis
        ax.plot(epochs, h["train_loss"], color=color, lw=1.5,
                linestyle="--", alpha=0.6, label="Train loss")
        ax.plot(epochs, h["val_loss"], color=color, lw=2.0,
                label="Val loss")
        ax.set_ylabel("Huber Loss (scaled)", fontsize=9, color=color)
        ax.tick_params(axis="y", labelcolor=color)

        # R² on right axis
        ax2.plot(epochs, h["val_r2"], color="#D7191C", lw=1.5,
                 linestyle="-", alpha=0.8, label="Val R²")
        ax2.set_ylabel("Val R² (scaled)", fontsize=9, color="#D7191C")
        ax2.tick_params(axis="y", labelcolor="#D7191C")

        # Best epoch
        best_ep = int(np.argmax(h["val_r2"])) + 1
        ax.axvline(best_ep, color="grey", lw=1, linestyle=":",
                   alpha=0.7)
        ax.text(best_ep + 1, ax.get_ylim()[1] * 0.95,
                f"Best\nep {best_ep}", fontsize=7.5, color="grey")

        ax.set_xlabel("Epoch", fontsize=10)
        name = mk.replace("_", " ").title()
        ax.set_title(name, fontsize=10, fontweight="bold")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  frameon=False, fontsize=7.5, loc="center right")

    fig.suptitle("GNN Training Curves — Regression (Test Retraining)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{OUTDIR}/figS_learning_curve_reg.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")