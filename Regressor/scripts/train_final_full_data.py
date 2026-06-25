import os, json, numpy as np, joblib, torch
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

from scripts.CV import (
    set_seed, _graph_dims, _inject_y, scale_graphs, train_epoch_reg,
    build_gnn_model_reg, hp_to_args_reg, _build_classical_reg_tuned,
    load_tuned_hp_reg,
)

HP_FILE = "best_hp_reg.json"
OUT_DIR = "saved_models_final_fulldata"
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_XGB_RU    = True
TRAIN_GINE_PBSG = False


def load_full_data():
    X_fp_pooled_RU = np.load("Vectors/X_fp_pooled_RU.npy")
    y              = np.load("Vectors/y.npy").astype(float)

    assert y.ndim == 1, "y must be 1-D"
    assert len(np.unique(y)) > 5, (
        f"y has {len(np.unique(y))} unique values — looks like has_Tm not Tm.")
    assert y.min() < 50 and y.max() > 100, (
        f"y range [{y.min()},{y.max()}] doesn't look like Tm (C).")
    print(f"Full data: N={len(y)}  Tm range [{y.min():.0f},{y.max():.0f}] C")
    return X_fp_pooled_RU, y


def train_xgb_full(X, y, hp_all, key="xgb_fp_pooled_ru"):
    print(f"\nTraining XGB full-data deployment model ({key})...")
    hp = hp_to_args_reg(hp_all, key)               # raw dict for classical
    model = _build_classical_reg_tuned("xgb", hp)
    xscaler = StandardScaler().fit(X)
    model.fit(xscaler.transform(X), y)
    joblib.dump(model,   f"{OUT_DIR}/reg_xgb_pooled_ru_FULL.pkl")
    joblib.dump(xscaler, f"{OUT_DIR}/reg_xgb_pooled_ru_FULL_xscaler.pkl")
    print(f"  saved -> {OUT_DIR}/reg_xgb_pooled_ru_FULL.pkl")


def train_gine_full(graphs, y, hp_all, key="gine_pbsg"):
    print(f"\nTraining GINE full-data deployment model ({key})...")
    args, seed = hp_to_args_reg(hp_all, key)
    set_seed(seed)
    graphs = _inject_y([g.clone() for g in graphs], y)
    y_scaler = StandardScaler().fit(np.asarray(y).reshape(-1, 1))
    graphs_s = scale_graphs(graphs, y_scaler)
    in_ch, edge_dim, meta_dim = _graph_dims(graphs_s)
    loader = DataLoader(graphs_s, batch_size=args.batch_size, shuffle=True)
    model = build_gnn_model_reg("gine", in_ch, edge_dim, meta_dim, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    model.train()
    for epoch in range(1, args.epochs + 1):     # fixed epochs (no val/early-stop)
        loss = train_epoch_reg(model, loader, optimizer, device)
        if epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train_loss={loss:.4f}")
    model.eval()
    torch.save(model.state_dict(), f"{OUT_DIR}/reg_gine_pbsg_FULL.pt")
    joblib.dump(y_scaler,          f"{OUT_DIR}/reg_gine_pbsg_FULL_yscaler.pkl")
    print(f"  saved -> {OUT_DIR}/reg_gine_pbsg_FULL.pt")


def main():
    hp_all = load_tuned_hp_reg(HP_FILE)
    X_ru, y = load_full_data()
    if TRAIN_XGB_RU:    train_xgb_full(X_ru, y, hp_all, "xgb_fp_pooled_ru")
    if TRAIN_GINE_PBSG: train_gine_full(graphs, y, hp_all, "gine_pbsg")
    with open(f"{OUT_DIR}/README.txt", "w") as f:
        f.write(
            "FULL-DATA DEPLOYMENT MODELS - trained on ALL internal rows.\n"
            "Use ONLY for EXTERNAL polymer predictions (Table 2).\n"
            "NOT valid for Table 1 / test metrics (no held-out test set).\n"
            f"HPs from {HP_FILE}. GINE fixed-epoch (no early stop). "
            "Huber loss, StandardScaler - identical to CV.py.\n")
    print(f"\nDone -> {OUT_DIR}/  (EXTERNAL predictions only)")


if __name__ == "__main__":
    main()
