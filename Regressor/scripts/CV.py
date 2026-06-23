"""
cv_reg.py
=========
Reusable CV functions for all regression model/rep combinations.
Mirrors cv_cls.py structure exactly — same batch runner pattern,
same timing, same summary table.

Self-contained: includes regressor model classes, build_gnn_model_reg,
training functions, and CV runners. Dims extracted automatically per rep.

Target: Tm in Celsius (raw). Scaled internally per fold using
StandardScaler fit on train only, inverse-transformed for metrics.

Metrics: MAE, RMSE, R²

Models:
  GNN:       GCN, GINE, GATv2
  Classical: Ridge (RR), RF, XGB

Usage
-----
from cv_reg import run_all_gnn_cv_reg, run_all_classical_cv_reg, print_cv_summary_reg

graph_reps = {
    "PBSG":          graphs_PBSG_reg,
    "SMILES":        graphs_SMILES_reg,
    "SMILES+global": graphs_SMILESplusglobal_reg,
}
vector_reps = {
    "FP RU":          X_fp_RU,
    "FP+pooled RU":   X_fp_pooled_RU,
    "FP poly":        X_fp_poly,
    "FP+pooled poly": X_fp_pooled_poly,
}

# With tuned HPs:
hp = load_tuned_hp_reg("best_hp_reg.json")
gnn_results = run_all_gnn_cv_reg(graph_reps, folds_reg, df_reg, device, y=y, hp_dict=hp)
cls_results = run_all_classical_cv_reg(vector_reps, y, folds_reg, df_reg, hp_dict=hp)
print_cv_summary_reg(gnn_results + cls_results)

# Without tuned HPs (uses defaults):
gnn_results = run_all_gnn_cv_reg(graph_reps, folds_reg, df_reg, device, y=y)
cls_results = run_all_classical_cv_reg(vector_reps, y, folds_reg, df_reg)
print_cv_summary_reg(gnn_results + cls_results)
"""

import json
import random
import time
import numpy as np
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn import (Linear, Dropout, BatchNorm1d, Sequential, ReLU)
from torch.optim import Adam
from torch_geometric.nn import (GCNConv, GINEConv, GATv2Conv, global_mean_pool)
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _graph_dims(graphs):
    g = graphs[0]
    in_ch    = g.x.shape[1]
    edge_dim = g.edge_attr.shape[1] if g.edge_attr is not None else 0
    meta_dim = g.architecture.shape[-1] if hasattr(g, "architecture") else 0
    return in_ch, edge_dim, meta_dim


def _inject_y(graphs, y_array):
    """
    Inject external y values into graph objects in-place.
    Only injects if g.y is None — leaves existing y values untouched.
    """
    n_injected = 0
    for i, g in enumerate(graphs):
        if g.y is None:
            g.y = torch.tensor([y_array[i]], dtype=torch.float32)
            n_injected += 1
    if n_injected > 0:
        print(f"  Injected y into {n_injected}/{len(graphs)} graphs.")
    return graphs


def compute_metrics(labels, preds):
    """Regression metrics in original (Celsius) space."""
    mae  = float(mean_absolute_error(labels, preds))
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    r2   = float(r2_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def collect_subgroup_results_reg(df, all_labels, all_preds, all_indices,
                                  arch_col="copolymer_type",
                                  stereo_col="stereo_class"):
    """Subgroup breakdown for regression — MAE, RMSE, R² per group."""
    sub = df.iloc[all_indices].copy().reset_index(drop=True)
    sub["_label"] = all_labels
    sub["_pred"]  = all_preds
    result = {}
    for col in [arch_col, stereo_col]:
        if col not in sub.columns:
            continue
        result[col] = {}
        for grp, grp_df in sub.groupby(col):
            yt = grp_df["_label"].values
            yp = grp_df["_pred"].values
            result[col][grp] = {
                "n":    len(yt),
                "mae":  float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "r2":   float(r2_score(yt, yp))
                        if len(np.unique(yt)) > 1 else 0.0,
            }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TUNED HPs
# ══════════════════════════════════════════════════════════════════════════════

def load_tuned_hp_reg(path="best_hp_reg.json"):
    """Load tuned HPs from Optuna JSON output."""
    with open(path) as f:
        return json.load(f)


def hp_to_args_reg(hp_dict, key):
    """
    Convert a HP dict entry to SimpleNamespace (GNN) or dict (classical).

    Parameters
    ----------
    hp_dict : full dict from best_hp_reg.json
    key     : e.g. "gine_pbsg", "xgb_fp_pooled_ru", "rr_fp_ru"

    Returns
    -------
    For GNN:       (SimpleNamespace args, int seed)
    For classical: dict of model kwargs
    """
    hp = hp_dict[key]
    if hp["type"] == "gnn":
        args = SimpleNamespace(
            hidden       = hp["hidden"],
            layers       = hp["layers"],
            dropout      = hp["dropout"],
            lr           = hp["lr"],
            weight_decay = hp["weight_decay"],
            epochs       = hp.get("epochs", 300),
            patience     = hp.get("patience", 50),
            batch_size   = hp.get("batch_size", 32),
        )
        return args, hp["seed"]
    else:
        return hp  # classical — return raw dict


def _build_classical_reg_tuned(model_name, hp):
    """Build classical regressor with tuned HPs from JSON."""
    if model_name == "rr":
        return Ridge(alpha=hp.get("alpha", 1.0))
    elif model_name == "rf":
        return RandomForestRegressor(
            n_estimators     = hp.get("n_estimators", 100),
            max_depth        = hp.get("max_depth", None),
            min_samples_split= hp.get("min_samples_split", 2),
            max_features     = hp.get("max_features", "sqrt"),
            n_jobs=-1, random_state=42,
        )
    elif model_name == "xgb":
        return XGBRegressor(
            n_estimators     = hp.get("n_estimators", 100),
            max_depth        = hp.get("max_depth", 6),
            learning_rate    = hp.get("learning_rate", 0.1),
            subsample        = hp.get("subsample", 0.8),
            colsample_bytree = hp.get("colsample_bytree", 0.8),
            min_child_weight = hp.get("min_child_weight", 1),
            reg_alpha        = hp.get("reg_alpha", 0),
            reg_lambda       = hp.get("reg_lambda", 1),
            eval_metric="rmse", tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# GNN REGRESSOR MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, meta_dim, hidden=64, n_layers=3,
                 dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drop  = Dropout(dropout)
        self.convs.append(GCNConv(in_channels, hidden))
        self.bns.append(BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(BatchNorm1d(hidden))
        self.regressor = Linear(hidden + meta_dim, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "architecture") and \
                self.regressor.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.regressor(x).squeeze(-1)


class GINERegressor(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, meta_dim, hidden=64,
                 n_layers=3, dropout=0.3):
        super().__init__()
        self.edge_dim = max(edge_dim, 1)
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drop  = Dropout(dropout)
        def mlp(a, b): return Sequential(Linear(a, b), ReLU(), Linear(b, b))
        self.convs.append(GINEConv(mlp(in_channels, hidden),
                                   edge_dim=self.edge_dim))
        self.bns.append(BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GINEConv(mlp(hidden, hidden),
                                       edge_dim=self.edge_dim))
            self.bns.append(BatchNorm1d(hidden))
        self.regressor = Linear(hidden + meta_dim, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ea = data.edge_attr
        if ea is None:
            ea = torch.zeros(ei.shape[1], self.edge_dim, device=x.device)
        ea = ea.float()
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei, ea)))
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "architecture") and \
                self.regressor.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.regressor(x).squeeze(-1)


class GATv2Regressor(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, meta_dim, hidden=64,
                 n_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drop  = Dropout(dropout)
        self.convs.append(GATv2Conv(in_channels, hidden // heads,
                                    heads=heads, edge_dim=edge_dim,
                                    concat=True))
        self.bns.append(BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden, hidden // heads,
                                        heads=heads, edge_dim=edge_dim,
                                        concat=True))
            self.bns.append(BatchNorm1d(hidden))
        self.regressor = Linear(hidden + meta_dim, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ea = data.edge_attr.float() if data.edge_attr is not None else None
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei, edge_attr=ea)))
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "architecture") and \
                self.regressor.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.regressor(x).squeeze(-1)


def build_gnn_model_reg(arch, in_ch, edge_dim, meta_dim, args=None):
    if args is None:
        args = GNN_DEFAULTS_REG[arch]
    if arch == "gcn":
        return GCNRegressor(in_ch, meta_dim,
                            hidden=args.hidden, n_layers=args.layers,
                            dropout=args.dropout)
    elif arch == "gine":
        return GINERegressor(in_ch, edge_dim, meta_dim,
                             hidden=args.hidden, n_layers=args.layers,
                             dropout=args.dropout)
    elif arch == "gatv2":
        return GATv2Regressor(in_ch, edge_dim, meta_dim,
                              hidden=args.hidden, n_layers=args.layers,
                              dropout=args.dropout)
    raise ValueError(f"Unknown arch: {arch}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def scale_graphs(gs, y_scaler):
    out = []
    for g in gs:
        g2   = g.clone()
        g2.y = torch.tensor(
            y_scaler.transform([[g.y.item()]])[0], dtype=torch.float32
        )
        out.append(g2)
    return out


def train_epoch_reg(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.huber_loss(out, batch.y.float(), delta=1.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_reg(model, loader, device):
    model.eval()
    preds, labels, total_loss = [], [], 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        out        = model(batch)
        total_loss += F.huber_loss(
            out, batch.y.float(), delta=1.0
        ).item() * batch.num_graphs
        preds.extend(out.cpu().tolist())
        labels.extend(batch.y.float().cpu().tolist())
    return (total_loss / len(loader.dataset),
            np.array(labels), np.array(preds))


def train_gnn_reg(model, train_loader, val_loader, args, device):
    optimizer    = Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
    best_val_r2  = -np.inf
    best_state   = None
    patience_ct  = 0
    history      = {"train_loss": [], "val_loss": [],
                    "val_mae": [], "val_r2": []}

    for epoch in range(1, args.epochs + 1):
        train_loss       = train_epoch_reg(model, train_loader,
                                           optimizer, device)
        val_loss, vl, vp = evaluate_reg(model, val_loader, device)
        val_mae          = mean_absolute_error(vl, vp)
        val_r2           = r2_score(vl, vp) \
                           if len(np.unique(vl)) > 1 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state  = deepcopy(model.state_dict())
            patience_ct = 0
        else:
            patience_ct += 1

        if patience_ct >= args.patience:
            print(f"    Early stop at epoch {epoch}  "
                  f"best_val_r2={best_val_r2:.4f}")
            break

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"val_mae={val_mae:.3f} "
                  f"val_r2={val_r2:.3f}")

    model.load_state_dict(best_state)
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT HPs
# ══════════════════════════════════════════════════════════════════════════════

GNN_DEFAULTS_REG = {
    "gcn":   SimpleNamespace(hidden=64,  layers=3, dropout=0.3, lr=3e-4,
                             weight_decay=1e-4, epochs=300, patience=50,
                             batch_size=32),
    "gine":  SimpleNamespace(hidden=128, layers=4, dropout=0.3, lr=3e-4,
                             weight_decay=1e-4, epochs=300, patience=50,
                             batch_size=32),
    "gatv2": SimpleNamespace(hidden=64,  layers=3, dropout=0.3, lr=3e-4,
                             weight_decay=1e-4, epochs=300, patience=50,
                             batch_size=32),
}

GNN_DEFAULT_SEEDS_REG = {"gcn": 0, "gine": 0, "gatv2": 1}


# ══════════════════════════════════════════════════════════════════════════════
# GNN CV
# ══════════════════════════════════════════════════════════════════════════════

def run_gnn_cv_reg(arch, graphs, folds, df, device,
                   name=None, args=None, seed=None, verbose=True):
    """
    5-fold CV for one GNN arch × graph rep combination (regression).

    Scaling: StandardScaler fit on train fold only.
             Metrics computed in original Celsius space.

    Note: graphs must have g.y set before calling. Use _inject_y() or
    run via run_all_gnn_cv_reg(y=y) which handles injection automatically.
    """
    name = name or arch.upper()
    args = args or GNN_DEFAULTS_REG[arch]
    seed = seed if seed is not None else GNN_DEFAULT_SEEDS_REG[arch]

    in_ch, edge_dim, meta_dim = _graph_dims(graphs)
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GNN REG CV: {name}  seed={seed}  "
              f"in_ch={in_ch}  edge_dim={edge_dim}  meta_dim={meta_dim}")
        print(f"{'='*60}")

    fold_metrics    = []
    all_val_labels  = []
    all_val_preds   = []
    all_val_indices = []
    cv_start        = time.time()

    for fold_i, fold in enumerate(folds):
        fold_start   = time.time()
        train_graphs = [graphs[i] for i in fold["train"]]
        val_graphs   = [graphs[i] for i in fold["val"]]

        # Fit scaler on train fold only
        y_train_raw = np.array([g.y.item() for g in train_graphs])
        y_scaler    = StandardScaler()
        y_scaler.fit(y_train_raw.reshape(-1, 1))

        train_s = scale_graphs(train_graphs, y_scaler)
        val_s   = scale_graphs(val_graphs,   y_scaler)

        train_loader = DataLoader(train_s, batch_size=args.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_s,   batch_size=args.batch_size,
                                  shuffle=False)

        model = build_gnn_model_reg(arch, in_ch, edge_dim, meta_dim,
                                    args=args).to(device)
        model, _ = train_gnn_reg(model, train_loader, val_loader,
                                  args, device)

        _, labels_s, preds_s = evaluate_reg(model, val_loader, device)

        # Inverse transform to Celsius
        labels = y_scaler.inverse_transform(
            labels_s.reshape(-1, 1)).flatten()
        preds  = y_scaler.inverse_transform(
            preds_s.reshape(-1, 1)).flatten()

        m = compute_metrics(labels, preds)
        fold_metrics.append(m)
        all_val_labels.extend(labels.tolist())
        all_val_preds.extend(preds.tolist())
        all_val_indices.extend(fold["val"])

        fold_time = time.time() - fold_start
        if verbose:
            print(f"  Fold {fold_i+1}: MAE={m['mae']:.2f}  "
                  f"RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}  "
                  f"[{fold_time:.0f}s]")

    total_time = time.time() - cv_start
    mean = {k: float(np.mean([f[k] for f in fold_metrics]))
            for k in ["mae", "rmse", "r2"]}
    std  = {k: float(np.std([f[k] for f in fold_metrics]))
            for k in ["mae", "rmse", "r2"]}

    if verbose:
        print(f"  {name} mean CV: MAE={mean['mae']:.2f} ± {std['mae']:.2f}  "
              f"RMSE={mean['rmse']:.2f} ± {std['rmse']:.2f}  "
              f"R²={mean['r2']:.4f} ± {std['r2']:.4f}  "
              f"[total {total_time:.0f}s]")

    return {
        "name":      name,
        "arch":      arch,
        "folds":     fold_metrics,
        "mean":      mean,
        "std":       std,
        "time_s":    total_time,
        "subgroups": collect_subgroup_results_reg(
            df,
            np.array(all_val_labels),
            np.array(all_val_preds),
            all_val_indices,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL CV
# ══════════════════════════════════════════════════════════════════════════════

def _build_classical_reg(model_name):
    if model_name == "rr":
        return Ridge(alpha=1.0)
    elif model_name == "rf":
        return RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    elif model_name == "xgb":
        return XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="rmse", tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_classical_cv_reg(model_name, X, y, folds, df,
                          name=None, verbose=True,
                          tuned_hp=None):
    """
    5-fold CV for one classical model × vector rep combination (regression).

    Scaling: StandardScaler on X fit on train fold only.
             y is raw Celsius — no scaling needed for classical models.
    """
    name = name or model_name.upper()

    if verbose:
        print(f"\n── Classical REG CV: {name} "
              f"──────────────────────────────────")

    fold_metrics    = []
    all_val_labels  = []
    all_val_preds   = []
    all_val_indices = []
    cv_start        = time.time()

    for fold_i, fold in enumerate(folds):
        fold_start = time.time()

        X_scaler = StandardScaler()
        X_train  = X_scaler.fit_transform(X[fold["train"]])
        X_val    = X_scaler.transform(X[fold["val"]])
        y_train  = y[fold["train"]]
        y_val    = y[fold["val"]]

        # Use tuned HPs if provided, else defaults
        clf = (_build_classical_reg_tuned(model_name, tuned_hp)
               if tuned_hp else _build_classical_reg(model_name))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        m = compute_metrics(y_val, preds)
        fold_metrics.append(m)
        all_val_labels.extend(y_val.tolist())
        all_val_preds.extend(preds.tolist())
        all_val_indices.extend(fold["val"])

        fold_time = time.time() - fold_start
        if verbose:
            print(f"  Fold {fold_i+1}: MAE={m['mae']:.2f}  "
                  f"RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}  "
                  f"[{fold_time:.0f}s]")

    total_time = time.time() - cv_start
    mean = {k: float(np.mean([f[k] for f in fold_metrics]))
            for k in ["mae", "rmse", "r2"]}
    std  = {k: float(np.std([f[k] for f in fold_metrics]))
            for k in ["mae", "rmse", "r2"]}

    if verbose:
        print(f"  {name} mean CV: MAE={mean['mae']:.2f} ± {std['mae']:.2f}  "
              f"RMSE={mean['rmse']:.2f} ± {std['rmse']:.2f}  "
              f"R²={mean['r2']:.4f} ± {std['r2']:.4f}  "
              f"[total {total_time:.0f}s]")

    return {
        "name":      name,
        "model":     model_name,
        "folds":     fold_metrics,
        "mean":      mean,
        "std":       std,
        "time_s":    total_time,
        "subgroups": collect_subgroup_results_reg(
            df,
            np.array(all_val_labels),
            np.array(all_val_preds),
            all_val_indices,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_gnn_cv_reg(graph_reps, folds, df, device,
                        y=None, archs=None, verbose=True,
                        hp_dict=None):
    """
    Run all GNN arch × graph rep combinations for regression.

    Parameters
    ----------
    graph_reps : dict of {rep_name: graph_list}
    y          : numpy array of Tm values — injected if g.y is None
    archs      : list — defaults to ["gcn", "gine", "gatv2"]
    hp_dict    : loaded JSON from best_hp_reg.json (optional)
                 If provided uses tuned HPs, falls back to defaults.
    """
    archs   = archs or ["gcn", "gine", "gatv2"]
    results = []

    for rep_name, graphs in graph_reps.items():
        if y is not None and graphs[0].y is None:
            print(f"  [{rep_name}] g.y is None — injecting from y array...")
            graphs = _inject_y(graphs, y)

        rep_key = rep_name.lower().replace("+", "_").replace(" ", "_")
        # rep_key = (rep_name.lower()
        #    .replace("+global", "_gl")
        #    .replace("+", "_")
        #    .replace(" ", "_"))

        for arch in archs:
            hp_key     = f"{arch}_{rep_key}"
            args, seed = None, None

            if hp_dict and hp_key in hp_dict:
                args, seed = hp_to_args_reg(hp_dict, hp_key)
                print(f"  Using tuned HPs for {hp_key}")
            else:
                if hp_dict:
                    print(f"  Warning: {hp_key} not in hp_dict — using defaults")

            r = run_gnn_cv_reg(arch, graphs, folds, df, device,
                               name=f"{arch.upper()} ({rep_name})",
                               args=args, seed=seed,
                               verbose=verbose)
            results.append(r)
    return results


def run_all_classical_cv_reg(vector_reps, y, folds, df,
                              models=None, verbose=True,
                              hp_dict=None):
    """
    Run all classical model × vector rep combinations for regression.

    Parameters
    ----------
    vector_reps : dict of {rep_name: X_array}
    y           : numpy array of Tm values in Celsius
    models      : list — defaults to ["rr", "rf", "xgb"]
    hp_dict     : loaded JSON from best_hp_reg.json (optional)
    """
    models  = models or ["rr", "rf", "xgb"]
    results = []

    for rep_name, X in vector_reps.items():
        rep_key = rep_name.lower().replace("+", "_").replace(" ", "_")

        for model_name in models:
            hp_key   = f"{model_name}_{rep_key}"
            tuned_hp = None

            if hp_dict and hp_key in hp_dict:
                tuned_hp = hp_to_args_reg(hp_dict, hp_key)
                print(f"  Using tuned HPs for {hp_key}")
            else:
                if hp_dict:
                    print(f"  Warning: {hp_key} not in hp_dict — using defaults")

            r = run_classical_cv_reg(
                model_name, X, y, folds, df,
                name=f"{model_name.upper()} ({rep_name})",
                tuned_hp=tuned_hp,
                verbose=verbose,
            )
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_cv_summary_reg(results, sort_by="r2"):
    """
    Ranked summary table. Sort by R² descending (or MAE/RMSE ascending).
    """
    reverse = sort_by == "r2"
    rows = []
    for r in results:
        t = r.get("time_s", 0)
        rows.append({
            "Name":  r["name"],
            "MAE":   f"{r['mean']['mae']:.2f} ± {r['std']['mae']:.2f}",
            "RMSE":  f"{r['mean']['rmse']:.2f} ± {r['std']['rmse']:.2f}",
            "R²":    f"{r['mean']['r2']:.4f} ± {r['std']['r2']:.4f}",
            "Time":  f"{t/60:.1f}m" if t >= 60 else f"{t:.0f}s",
            "_sort": r["mean"][sort_by],
        })
    rows.sort(key=lambda x: x["_sort"], reverse=reverse)

    col_w = {"Name": 30, "MAE": 16, "RMSE": 16, "R²": 16, "Time": 8}
    sep   = "=" * sum(col_w.values())
    print(f"\n{sep}")
    print(f"  REG CV SUMMARY  (sorted by {sort_by.upper()})")
    print(sep)
    print("".join(f"{k:<{w}}" for k, w in col_w.items()))
    print("-" * sum(col_w.values()))
    for row in rows:
        print("".join(f"{row[k]:<{w}}" for k, w in col_w.items()))
    print(sep)