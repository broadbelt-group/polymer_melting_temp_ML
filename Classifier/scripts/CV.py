"""
cv_cls.py
=========
Reusable CV functions for all classification model/rep combinations.

Self-contained: includes model classes, build_gnn_model, training
functions, and CV runners. IN_CHANNELS / EDGE_DIM / META_DIM are
extracted automatically from each graph list — no globals needed.

Usage
-----
from cv_cls import run_all_gnn_cv, run_all_classical_cv, print_cv_summary

graph_reps = {
    "PBSG":          graphs_PBSG,
    "SMILES":        graphs_SMILES,
    "SMILES+global": graphs_SMILESplusglobal,
}
vector_reps = {
    "FP RU":          X_fp_RU,
    "FP+pooled RU":   X_fp_pooled_RU,
    "FP poly":        X_fp_poly,
    "FP+pooled poly": X_fp_pooled_poly,
}

# With tuned HPs:
hp = load_tuned_hp_cls("best_hp_full.json")
gnn_results = run_all_gnn_cv(graph_reps, folds, df, device, hp_dict=hp)
cls_results = run_all_classical_cv(vector_reps, y, folds, df, hp_dict=hp)
print_cv_summary(gnn_results + cls_results)

# Without tuned HPs (uses defaults):
gnn_results = run_all_gnn_cv(graph_reps, folds, df, device)
cls_results = run_all_classical_cv(vector_reps, y, folds, df)
print_cv_summary(gnn_results + cls_results)
"""

import json
import random
import time
import numpy as np
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn import (Linear, Dropout, BatchNorm1d,
                      Sequential, ReLU)
from torch.optim import Adam
from torch_geometric.nn import (GCNConv, GINEConv, GATv2Conv,
                                 global_mean_pool)
from torch_geometric.loader import DataLoader

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, roc_curve,
                              precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TUNED HPs
# ══════════════════════════════════════════════════════════════════════════════

def load_tuned_hp_cls(path="best_hp_full.json"):
    """Load tuned HPs from Optuna JSON output."""
    with open(path) as f:
        return json.load(f)


def hp_to_args_cls(hp_dict, key):
    """
    Convert a HP dict entry to SimpleNamespace (GNN) or dict (classical).

    Parameters
    ----------
    hp_dict : full dict from best_hp_full.json
    key     : e.g. "gine_pbsg", "xgb_fp_pooled_ru", "lr_fp_ru"

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
            epochs       = hp.get("epochs", 200),
            patience     = hp.get("patience", 30),
            batch_size   = hp.get("batch_size", 32),
        )
        return args, hp["seed"]
    else:
        return hp  # classical — return raw dict


def _build_classical_tuned(model_name, hp, scale_pos_weight=1.0):
    """Build classical classifier with tuned HPs from JSON."""
    if model_name == "lr":
        return LogisticRegression(
            C            = hp.get("C", 1.0),
            max_iter     = hp.get("max_iter", 1000),
            class_weight = "balanced",
            solver       = "lbfgs",
            random_state = 42,
        )
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators     = hp.get("n_estimators", 100),
            max_depth        = hp.get("max_depth", None),
            min_samples_split= hp.get("min_samples_split", 2),
            max_features     = hp.get("max_features", "sqrt"),
            class_weight     = "balanced",
            n_jobs=-1, random_state=42,
        )
    elif model_name == "xgb":
        return XGBClassifier(
            n_estimators     = hp.get("n_estimators", 100),
            max_depth        = hp.get("max_depth", 6),
            learning_rate    = hp.get("learning_rate", 0.1),
            subsample        = hp.get("subsample", 0.8),
            colsample_bytree = hp.get("colsample_bytree", 0.8),
            min_child_weight = hp.get("min_child_weight", 1),
            reg_alpha        = hp.get("reg_alpha", 0),
            reg_lambda       = hp.get("reg_lambda", 1),
            scale_pos_weight = scale_pos_weight,
            eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# GNN MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class GCNModel(torch.nn.Module):
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
        self.classifier = Linear(hidden + meta_dim, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "architecture") and self.classifier.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.classifier(x).squeeze(-1)


class GINEModel(torch.nn.Module):
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
        self.classifier = Linear(hidden + meta_dim, 1)

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
        if hasattr(data, "architecture") and self.classifier.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.classifier(x).squeeze(-1)


class GATv2Model(torch.nn.Module):
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
        self.classifier = Linear(hidden + meta_dim, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ea = data.edge_attr.float() if data.edge_attr is not None else None
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei, edge_attr=ea)))
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "architecture") and self.classifier.in_features > x.shape[1]:
            meta = data.architecture.float()
            if meta.dim() == 3:
                meta = meta.squeeze(1)
            x = torch.cat([x, meta], dim=-1)
        return self.classifier(x).squeeze(-1)


def build_gnn_model(arch, in_ch, edge_dim, meta_dim, args=None):
    if args is None:
        args = GNN_DEFAULTS[arch]
    if arch == "gcn":
        return GCNModel(in_ch, meta_dim,
                        hidden=args.hidden, n_layers=args.layers,
                        dropout=args.dropout)
    elif arch == "gine":
        return GINEModel(in_ch, edge_dim, meta_dim,
                         hidden=args.hidden, n_layers=args.layers,
                         dropout=args.dropout)
    elif arch == "gatv2":
        return GATv2Model(in_ch, edge_dim, meta_dim,
                          hidden=args.hidden, n_layers=args.layers,
                          dropout=args.dropout)
    raise ValueError(f"Unknown arch: {arch}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(batch), batch.y.float()
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()
    preds, labels, probs, total_loss = [], [], [], 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        out        = model(batch)
        total_loss += (F.binary_cross_entropy_with_logits(
            out, batch.y.float()).item() * batch.num_graphs)
        prob = torch.sigmoid(out).cpu().numpy()
        probs.extend(prob.tolist())
        preds.extend((prob > 0.5).astype(int).tolist())
        labels.extend(batch.y.long().cpu().tolist())
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = np.array(preds)
    auroc  = (roc_auc_score(labels, probs)
              if len(np.unique(labels)) > 1 else 0.0)
    auprc  = (average_precision_score(labels, probs)
              if len(np.unique(labels)) > 1 else 0.0)
    return total_loss / len(loader.dataset), labels, preds, probs, auroc, auprc


def train_gnn(model, train_loader, val_loader, args, device):
    optimizer   = Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    best_auroc  = -1
    best_state  = None
    patience_ct = 0
    history     = {"train_loss": [], "val_loss": [],
                   "val_auroc": [], "val_auprc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _, _, val_auroc, val_auprc = evaluate_gnn(
            model, val_loader, device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)

        if val_auroc > best_auroc:
            best_auroc  = val_auroc
            best_state  = deepcopy(model.state_dict())
            patience_ct = 0
        else:
            patience_ct += 1

        if patience_ct >= args.patience:
            print(f"    Early stop at epoch {epoch}  "
                  f"best_auroc={best_auroc:.4f}")
            break

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} "
                  f"val_auprc={val_auprc:.4f}")

    model.load_state_dict(best_state)
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "acc":   accuracy_score(y_true, y_pred),
        "f1":    f1_score(y_true, y_pred, average="macro",
                          zero_division=0),
        "auroc": (roc_auc_score(y_true, y_prob)
                  if len(np.unique(y_true)) > 1 else 0.0),
        "auprc": (average_precision_score(y_true, y_prob)
                  if len(np.unique(y_true)) > 1 else 0.0),
    }


def full_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":         accuracy_score(y_true, y_pred),
        "Macro F1":         f1_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "AUROC":            roc_auc_score(y_true, y_prob),
        "AUPRC":            average_precision_score(y_true, y_prob),
        "No Tm Precision":  precision_score(y_true, y_pred,
                                            pos_label=0, zero_division=0),
        "No Tm Recall":     recall_score(y_true, y_pred,
                                         pos_label=0, zero_division=0),
        "Has Tm Precision": precision_score(y_true, y_pred,
                                            pos_label=1, zero_division=0),
        "Has Tm Recall":    recall_score(y_true, y_pred,
                                         pos_label=1, zero_division=0),
    }


def optimal_threshold(labels, probs):
    _, _, thresholds = roc_curve(labels, probs)
    f1s = [f1_score(labels, (probs >= t).astype(int), zero_division=0)
           for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


def collect_subgroup_results(df, all_labels, all_preds, all_indices,
                              arch_col="copolymer_type",
                              stereo_col="stereo_class"):
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
                "n":   len(yt),
                "acc": accuracy_score(yt, yp),
                "f1":  f1_score(yt, yp, zero_division=0),
            }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT HPs
# ══════════════════════════════════════════════════════════════════════════════

GNN_DEFAULTS = {
    "gcn":   SimpleNamespace(hidden=64,  layers=3, dropout=0.2, lr=1e-3,
                             weight_decay=1e-4, epochs=200, patience=30,
                             batch_size=32),
    "gine":  SimpleNamespace(hidden=128, layers=4, dropout=0.2, lr=1e-3,
                             weight_decay=1e-4, epochs=200, patience=30,
                             batch_size=32),
    "gatv2": SimpleNamespace(hidden=64,  layers=3, dropout=0.2, lr=1e-3,
                             weight_decay=1e-4, epochs=200, patience=30,
                             batch_size=32),
}

GNN_DEFAULT_SEEDS = {"gcn": 0, "gine": 0, "gatv2": 1}


# ══════════════════════════════════════════════════════════════════════════════
# GNN CV
# ══════════════════════════════════════════════════════════════════════════════

def run_gnn_cv(arch, graphs, folds, df, device,
               name=None, args=None, seed=None, verbose=True):
    name = name or arch.upper()
    args = args or GNN_DEFAULTS[arch]
    seed = seed if seed is not None else GNN_DEFAULT_SEEDS[arch]

    in_ch, edge_dim, meta_dim = _graph_dims(graphs)
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GNN CV: {name}  seed={seed}  "
              f"in_ch={in_ch}  edge_dim={edge_dim}  meta_dim={meta_dim}")
        print(f"{'='*60}")

    fold_metrics    = []
    all_val_labels  = []
    all_val_preds   = []
    all_val_indices = []
    cv_start = time.time()

    for fold_i, fold in enumerate(folds):
        fold_start   = time.time()
        train_loader = DataLoader(
            [graphs[i] for i in fold["train"]],
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            [graphs[i] for i in fold["val"]],
            batch_size=args.batch_size, shuffle=False,
        )

        model = build_gnn_model(arch, in_ch, edge_dim, meta_dim,
                                args=args).to(device)
        model, _ = train_gnn(model, train_loader, val_loader, args, device)

        _, labels, _, probs, _, _ = evaluate_gnn(model, val_loader, device)
        thresh = optimal_threshold(labels, probs)
        preds  = (probs >= thresh).astype(int)
        m      = compute_metrics(labels, preds, probs)

        fold_metrics.append(m)
        all_val_labels.extend(labels.tolist())
        all_val_preds.extend(preds.tolist())
        all_val_indices.extend(fold["val"])

        fold_time = time.time() - fold_start
        if verbose:
            print(f"  Fold {fold_i+1}: acc={m['acc']:.4f}  "
                  f"f1={m['f1']:.4f}  auroc={m['auroc']:.4f}  "
                  f"auprc={m['auprc']:.4f}  [{fold_time:.0f}s]")

    total_time = time.time() - cv_start
    mean = {k: float(np.mean([f[k] for f in fold_metrics]))
            for k in ["acc", "f1", "auroc", "auprc"]}
    std  = {k: float(np.std([f[k] for f in fold_metrics]))
            for k in ["acc", "f1", "auroc", "auprc"]}

    if verbose:
        print(f"  {name} mean CV: acc={mean['acc']:.4f}  "
              f"f1={mean['f1']:.4f}  auroc={mean['auroc']:.4f}  "
              f"auprc={mean['auprc']:.4f}  [total {total_time:.0f}s]")

    return {
        "name":       name,
        "arch":       arch,
        "folds":      fold_metrics,
        "mean":       mean,
        "std":        std,
        "time_s":     total_time,
        "subgroups":  collect_subgroup_results(
            df,
            np.array(all_val_labels),
            np.array(all_val_preds),
            all_val_indices,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL CV
# ══════════════════════════════════════════════════════════════════════════════

def _build_classical(model_name, scale_pos_weight=1.0):
    if model_name == "lr":
        return LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
    elif model_name == "xgb":
        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_classical_cv(model_name, X, y, folds, df,
                     name=None, scale_pos_weight=None, verbose=True,
                     tuned_hp=None):
    """5-fold CV for one classical model × vector rep combination."""
    name = name or model_name.upper()

    if scale_pos_weight is None:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    if verbose:
        print(f"\n── Classical CV: {name} "
              f"──────────────────────────────────")

    fold_metrics    = []
    all_val_labels  = []
    all_val_preds   = []
    all_val_indices = []
    cv_start = time.time()

    for fold_i, fold in enumerate(folds):
        fold_start = time.time()

        # Use tuned HPs if provided, else defaults
        clf = (_build_classical_tuned(model_name, tuned_hp, scale_pos_weight)
               if tuned_hp else _build_classical(model_name, scale_pos_weight))

        clf.fit(X[fold["train"]], y[fold["train"]])

        probs  = clf.predict_proba(X[fold["val"]])[:, 1]
        thresh = optimal_threshold(y[fold["val"]], probs)
        preds  = (probs >= thresh).astype(int)
        m      = compute_metrics(y[fold["val"]], preds, probs)

        fold_metrics.append(m)
        all_val_labels.extend(y[fold["val"]].tolist())
        all_val_preds.extend(preds.tolist())
        all_val_indices.extend(fold["val"])

        fold_time = time.time() - fold_start
        if verbose:
            print(f"  Fold {fold_i+1}: acc={m['acc']:.4f}  "
                  f"f1={m['f1']:.4f}  auroc={m['auroc']:.4f}  "
                  f"auprc={m['auprc']:.4f}  [{fold_time:.0f}s]")

    total_time = time.time() - cv_start
    mean = {k: float(np.mean([f[k] for f in fold_metrics]))
            for k in ["acc", "f1", "auroc", "auprc"]}
    std  = {k: float(np.std([f[k] for f in fold_metrics]))
            for k in ["acc", "f1", "auroc", "auprc"]}

    if verbose:
        print(f"  {name} mean CV: acc={mean['acc']:.4f}  "
              f"f1={mean['f1']:.4f}  auroc={mean['auroc']:.4f}  "
              f"auprc={mean['auprc']:.4f}  [total {total_time:.0f}s]")

    return {
        "name":      name,
        "model":     model_name,
        "folds":     fold_metrics,
        "mean":      mean,
        "std":       std,
        "time_s":    total_time,
        "subgroups": collect_subgroup_results(
            df,
            np.array(all_val_labels),
            np.array(all_val_preds),
            all_val_indices,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_gnn_cv(graph_reps, folds, df, device,
                   archs=None, verbose=True,
                   hp_dict=None):
    """
    Run all GNN arch × graph rep combinations.

    Parameters
    ----------
    graph_reps : dict of {rep_name: graph_list}
    archs      : list — defaults to ["gcn", "gine", "gatv2"]
    hp_dict    : loaded JSON from best_hp_full.json (optional)
                 If provided uses tuned HPs, falls back to defaults.
    """
    archs   = archs or ["gcn", "gine", "gatv2"]
    results = []

    for rep_name, graphs in graph_reps.items():
        rep_key = rep_name.lower().replace("+", "_").replace(" ", "_")

        for arch in archs:
            hp_key     = f"{arch}_{rep_key}"
            args, seed = None, None

            if hp_dict and hp_key in hp_dict:
                args, seed = hp_to_args_cls(hp_dict, hp_key)
                print(f"  Using tuned HPs for {hp_key}")
            else:
                if hp_dict:
                    print(f"  Warning: {hp_key} not in hp_dict — using defaults")

            r = run_gnn_cv(arch, graphs, folds, df, device,
                           name=f"{arch.upper()} ({rep_name})",
                           args=args, seed=seed,
                           verbose=verbose)
            results.append(r)
    return results


def run_all_classical_cv(vector_reps, y, folds, df,
                         models=None, verbose=True,
                         hp_dict=None):
    """
    Run all classical model × vector rep combinations.

    Parameters
    ----------
    vector_reps : dict of {rep_name: X_array}
    models      : list — defaults to ["lr", "rf", "xgb"]
    hp_dict     : loaded JSON from best_hp_full.json (optional)
    """
    models  = models or ["lr", "rf", "xgb"]
    results = []

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    for rep_name, X in vector_reps.items():
        rep_key = rep_name.lower().replace("+", "_").replace(" ", "_")

        for model_name in models:
            hp_key   = f"{model_name}_{rep_key}"
            tuned_hp = None

            if hp_dict and hp_key in hp_dict:
                tuned_hp = hp_to_args_cls(hp_dict, hp_key)
                print(f"  Using tuned HPs for {hp_key}")
            else:
                if hp_dict:
                    print(f"  Warning: {hp_key} not in hp_dict — using defaults")

            r = run_classical_cv(
                model_name, X, y, folds, df,
                name=f"{model_name.upper()} ({rep_name})",
                scale_pos_weight=scale_pos_weight,
                tuned_hp=tuned_hp,
                verbose=verbose,
            )
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_cv_summary(results, sort_by="auprc"):
    """Ranked summary table across all model/rep combinations."""
    rows = []
    for r in results:
        t = r.get("time_s", 0)
        rows.append({
            "Name":  r["name"],
            "Acc":   f"{r['mean']['acc']:.4f} ± {r['std']['acc']:.4f}",
            "F1":    f"{r['mean']['f1']:.4f} ± {r['std']['f1']:.4f}",
            "AUROC": f"{r['mean']['auroc']:.4f} ± {r['std']['auroc']:.4f}",
            "AUPRC": f"{r['mean']['auprc']:.4f} ± {r['std']['auprc']:.4f}",
            "Time":  f"{t/60:.1f}m" if t >= 60 else f"{t:.0f}s",
            "_sort": r["mean"][sort_by],
        })
    rows.sort(key=lambda x: x["_sort"], reverse=True)

    col_w = {"Name": 30, "Acc": 18, "F1": 18, "AUROC": 18, "AUPRC": 18, "Time": 8}
    sep   = "=" * sum(col_w.values())
    print(f"\n{sep}")
    print(f"  CV SUMMARY  (sorted by {sort_by.upper()})")
    print(sep)
    print("".join(f"{k:<{w}}" for k, w in col_w.items()))
    print("-" * sum(col_w.values()))
    for row in rows:
        print("".join(f"{row[k]:<{w}}" for k, w in col_w.items()))
    print(sep)