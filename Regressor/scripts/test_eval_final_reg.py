"""
test_evaluation_reg.py
======================
Final held-out test set evaluation for the three tuned regression models:
  - GINE (PBSG) tuned
  - XGB (FP+pooled RU) tuned
  - XGB (FP+pooled poly) tuned

Workflow
--------
1. Retrain each model on the FULL trainval set using tuned HPs
2. Evaluate once on the held-out test set
3. Report MAE, RMSE, R² + subgroup breakdown by stereo and architecture
4. Save results to test_results_reg.csv, test_results_reg_subgroups.json,
   test_predictions_reg.json

Usage
-----
Assumes the following are loaded as globals in your notebook:
  graphs_PBSG      : PBSG regression graph list (with y injected)
  X_fp_pooled_RU   : FP+pooled RU feature array
  X_fp_pooled_poly : FP+pooled poly feature array
  y                : numpy array of Tm values in Celsius (407,)
  df_reg           : regression dataframe (407 rows)
  trainval_idx_reg : array of trainval indices
  test_idx_reg     : array of test indices
  device           : torch device

  from test_evaluation_reg import run_all_test_reg
  test_results_reg = run_all_test_reg(
      graphs_PBSG, X_fp_pooled_RU, X_fp_pooled_poly,
      y, df_reg, trainval_idx_reg, test_idx_reg, device
  )
"""

import json
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from copy import deepcopy

import torch
import joblib
from torch_geometric.loader import DataLoader
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from CV import (
    build_gnn_model_reg, train_gnn_reg, evaluate_reg,
    scale_graphs, compute_metrics, collect_subgroup_results_reg,
    set_seed, _graph_dims,
)


# ══════════════════════════════════════════════════════════════════════════════
# TUNED HPs — update with your Optuna results
# ══════════════════════════════════════════════════════════════════════════════

GINE_REG_HP = SimpleNamespace(
    hidden=64, layers=4, dropout=0.07876, lr=0.001409,
    weight_decay=8.2e-5, epochs=300, patience=50, batch_size=32
)
GINE_REG_SEED = 3

XGB_RU_REG_PARAMS = dict(
    n_estimators=296, max_depth=3, learning_rate=0.25222,
    subsample=0.8876, colsample_bytree=0.9697,
    min_child_weight=9, reg_alpha=6.07e-4, reg_lambda=0.2371,
    eval_metric="rmse", tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0,
)

XGB_POLY_REG_PARAMS = dict(
    n_estimators=346, max_depth=3, learning_rate=0.10279,
    subsample=0.7315, colsample_bytree=0.6171,
    min_child_weight=7, reg_alpha=1.44e-7, reg_lambda=0.2412,
    eval_metric="rmse", tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0,
)

GCN_PBSG = SimpleNamespace(
    hidden=64, layers=2, dropout=0.496324, lr=0.00299362,
    weight_decay=3.86e-06, epochs=300, patience=50, batch_size=32
)
GCN_PBSG_SEED = 10

GCN_SMILES = SimpleNamespace(
    hidden=64, layers=5, dropout=0.233955, lr=0.00373814,
    weight_decay=0.0001015, epochs=300, patience=50, batch_size=32
)
GCN_SMILES_SEED = 3

GCN_SMILES_GL = SimpleNamespace(
    hidden=128, layers=4, dropout=0.157178, lr=0.00104026,
    weight_decay=0.00052808, epochs=300, patience=50, batch_size=32
)
GCN_SMILES_GL_SEED = 2

GINE_PBSG = SimpleNamespace(
    hidden=256, layers=3, dropout=0.186695, lr=0.00158151,
    weight_decay=8.667e-05, epochs=300, patience=50, batch_size=32
)
GINE_PBSG_SEED = 42

GINE_SMILES = SimpleNamespace(
    hidden=64, layers=3, dropout=0.126965, lr= 0.0013683,
    weight_decay=3.5e-06, epochs=300, patience=50, batch_size=32
)
GINE_SMILES_SEED = 10

GINE_SMILES_GL = SimpleNamespace(
    hidden=128, layers=4, dropout=0.143875, lr=0.00115474,
    weight_decay=0.00021791, epochs=300, patience=50, batch_size=32
)
GINE_SMILES_GL_SEED = 1

GATV2_PBSG = SimpleNamespace(
    hidden=256, layers=2, dropout=0.009053, lr= 0.0087259,
    weight_decay=1.294e-05, epochs=300, patience=50, batch_size=32
)
GATV2_PBSG_SEED = 0

GATV2_SMILES = SimpleNamespace(
    hidden=256, layers=2, dropout=0.318294, lr= 0.00518449,
    weight_decay=1.957e-05, epochs=300, patience=50, batch_size=32
)
GATV2_SMILES_SEED = 0





GINE_REG_HP = SimpleNamespace(
    hidden=64, layers=4, dropout=0.07876, lr=0.001409,
    weight_decay=8.2e-5, epochs=300, patience=50, batch_size=32
)
GINE_REG_SEED = 3


XGB_RU_REG_PARAMS = dict(
    n_estimators=296, max_depth=3, learning_rate=0.25222,
    subsample=0.8876, colsample_bytree=0.9697,
    min_child_weight=9, reg_alpha=6.07e-4, reg_lambda=0.2371,
    eval_metric="rmse", tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0,
)

XGB_POLY_REG_PARAMS = dict(
    n_estimators=346, max_depth=3, learning_rate=0.10279,
    subsample=0.7315, colsample_bytree=0.6171,
    min_child_weight=7, reg_alpha=1.44e-7, reg_lambda=0.2412,
    eval_metric="rmse", tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0,
)


# ══════════════════════════════════════════════════════════════════════════════
# FULL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def full_metrics_reg(labels, preds):
    """Extended regression metrics."""
    mae  = float(mean_absolute_error(labels, preds))
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    r2   = float(r2_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0
    # Mean absolute percentage error (where Tm != 0)
    nonzero = labels != 0
    mape = float(np.mean(np.abs((labels[nonzero] - preds[nonzero]) /
                                 labels[nonzero])) * 100) if nonzero.any() else 0.0
    # Max error
    max_err = float(np.max(np.abs(labels - preds)))
    return {
        "MAE":     mae,
        "RMSE":    rmse,
        "R²":      r2,
        "MAPE (%)": mape,
        "Max Error": max_err,
    }


def _print_metrics(name, metrics):
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS: {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<22s}: {v:.4f}")


def _print_subgroups(subgroups):
    for col, groups in subgroups.items():
        print(f"\n  Subgroup: {col}")
        print(f"  {'Group':<20s} {'n':>5} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        print(f"  {'-'*53}")
        for grp, m in sorted(groups.items()):
            print(f"  {str(grp):<20s} {m['n']:>5} "
                  f"{m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# GNN TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gnn_test_reg(arch, graphs, trainval_idx, test_idx,
                           df, device, args, seed, name):
    """
    Retrain GNN regressor on full trainval, evaluate on test.
    Scales y on trainval, inverse transforms for metrics.
    """
    set_seed(seed)
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    train_graphs = [graphs[i] for i in trainval_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    # Fit scaler on trainval only
    y_train_raw = np.array([g.y.item() for g in train_graphs])
    y_scaler    = StandardScaler()
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    train_s = scale_graphs(train_graphs, y_scaler)
    test_s  = scale_graphs(test_graphs,  y_scaler)

    train_loader = DataLoader(train_s, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_s,  batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print(f"  Retraining {name} on full trainval (n={len(trainval_idx)})")
    print(f"{'='*60}")

    model = build_gnn_model_reg(arch, in_ch, edge_dim, meta_dim,
                                args=args).to(device)
    model, history = train_gnn_reg(model, train_loader, test_loader,
                                   args, device)

    _, labels_s, preds_s = evaluate_reg(model, test_loader, device)

    # Inverse transform to Celsius
    labels = y_scaler.inverse_transform(labels_s.reshape(-1, 1)).flatten()
    preds  = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()

    m = full_metrics_reg(labels, preds)
    _print_metrics(name, m)

    subgroups = collect_subgroup_results_reg(df, labels, preds,
                                             list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name":      name,
        "metrics":   m,
        "preds":     preds,
        "labels":    labels,
        "subgroups": subgroups,
        "history":   history,
        "model":     model,     # keep for saving
        "y_scaler":  y_scaler,  # keep for inverse transform in prediction
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_xgb_test_reg(X, y, trainval_idx, test_idx,
                           df, params, name):
    """
    Retrain XGB regressor on full trainval, evaluate on test.
    Scales X on trainval only.
    """
    print(f"\n{'='*60}")
    print(f"  Retraining {name} on full trainval (n={len(trainval_idx)})")
    print(f"{'='*60}")

    X_scaler = StandardScaler()
    X_train  = X_scaler.fit_transform(X[trainval_idx])
    X_test   = X_scaler.transform(X[test_idx])
    y_train  = y[trainval_idx]
    labels   = y[test_idx]

    clf = XGBRegressor(**params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    m = full_metrics_reg(labels, preds)
    _print_metrics(name, m)

    subgroups = collect_subgroup_results_reg(df, labels, preds,
                                             list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name":      name,
        "metrics":   m,
        "preds":     preds,
        "labels":    labels,
        "subgroups": subgroups,
        "clf":       clf,
        "X_scaler":  X_scaler,  # keep for prediction pipeline
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════

def run_all_test_reg(graphs_PBSG, X_fp_pooled_RU, X_fp_pooled_poly,
                     y, df_reg, trainval_idx, test_idx, device):
    """
    Run test evaluation for all three tuned regression models.
    """
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)
    results      = {}

    # ── GINE (PBSG) ──────────────────────────────────────────────────────────
    results["gine_pbsg"] = evaluate_gnn_test_reg(
        "gine", graphs_PBSG, trainval_idx, test_idx,
        df_reg, device, GINE_REG_HP, GINE_REG_SEED,
        "GINE (PBSG) tuned"
    )

    # ── XGB (FP+pooled RU) ───────────────────────────────────────────────────
    results["xgb_ru"] = evaluate_xgb_test_reg(
        X_fp_pooled_RU, y, trainval_idx, test_idx,
        df_reg, XGB_RU_REG_PARAMS, "XGB (FP+pooled RU) tuned"
    )

    # ── XGB (FP+pooled poly) ─────────────────────────────────────────────────
    results["xgb_poly"] = evaluate_xgb_test_reg(
        X_fp_pooled_poly, y, trainval_idx, test_idx,
        df_reg, XGB_POLY_REG_PARAMS, "XGB (FP+pooled poly) tuned"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  FINAL REGRESSION TEST SUMMARY")
    print(f"{'='*75}")
    metric_keys = ["MAE", "RMSE", "R²", "MAPE (%)", "Max Error"]
    col_w = 16
    header = f"{'Model':<30}" + "".join(f"{k:<{col_w}}" for k in metric_keys)
    print(header)
    print("-" * (30 + col_w * len(metric_keys)))
    for key, r in results.items():
        row = f"{r['name']:<30}"
        for k in metric_keys:
            row += f"{r['metrics'][k]:<{col_w}.4f}"
        print(row)
    print("=" * (30 + col_w * len(metric_keys)))

    # ── Save ──────────────────────────────────────────────────────────────────
    rows = []
    for key, r in results.items():
        row = {"name": r["name"]}
        row.update(r["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv("test_results_reg.csv", index=False)
    print(f"\nSaved → test_results_reg.csv")

    # Subgroups
    subgroups_out = {k: v["subgroups"] for k, v in results.items()}
    with open("test_results_reg_subgroups.json", "w") as f:
        json.dump(subgroups_out, f, indent=2)
    print(f"Saved → test_results_reg_subgroups.json")

    # Predictions for scatter/residual plots
    preds_out = {
        k: {
            "preds":  v["preds"].tolist(),
            "labels": v["labels"].tolist(),
        }
        for k, v in results.items()
    }
    with open("test_predictions_reg.json", "w") as f:
        json.dump(preds_out, f, indent=2)
    print(f"Saved → test_predictions_reg.json")

    # Training histories for GNN learning curves
    histories_out = {}
    for key, r in results.items():
        if "history" in r:
            histories_out[key] = {
                k: [float(x) for x in vals]
                for k, vals in r["history"].items()
            }
    with open("test_training_histories_reg.json", "w") as f:
        json.dump(histories_out, f, indent=2)
    print(f"Saved → test_training_histories_reg.json")

    # ── Save trained models for prediction pipeline ────────────────
    os.makedirs("saved_models_top", exist_ok=True)

    # GINE state dict + y_scaler
    gine_model   = results["gine_pbsg"].get("model")
    gine_scaler  = results["gine_pbsg"].get("y_scaler")
    if gine_model is not None:
        torch.save(gine_model.state_dict(),
                   "saved_models_top/reg_gine_pbsg.pt")
        print("Saved → saved_models_top/reg_gine_pbsg.pt")
    if gine_scaler is not None:
        joblib.dump(gine_scaler, "saved_models_top/reg_gine_pbsg_yscaler.pkl")
        print("Saved → saved_models_top/reg_gine_pbsg_yscaler.pkl")

    # XGB regressors + X_scalers
    for key, model_fname, scaler_fname in [
        ("xgb_ru",
         "saved_models_top/reg_xgb_pooled_ru.pkl",
         "saved_models_top/reg_xgb_pooled_ru_xscaler.pkl"),
        ("xgb_poly",
         "saved_models_top/reg_xgb_pooled_poly.pkl",
         "saved_models_top/reg_xgb_pooled_poly_xscaler.pkl"),
    ]:
        clf     = results[key].get("clf")
        scaler  = results[key].get("X_scaler")
        if clf is not None:
            joblib.dump(clf, model_fname)
            print(f"Saved → {model_fname}")
        if scaler is not None:
            joblib.dump(scaler, scaler_fname)
            print(f"Saved → {scaler_fname}")

    return results