"""
test_evaluation_reg.py
======================
Final held-out test set evaluation for ALL regression model/rep combinations:
  GNN  (9): GCN, GINE, GATv2 × {PBSG, SMILES, SMILES+global}
  Classical (12): RR, RF, XGB × {FP+pooled RU, FP RU, FP+pooled poly, FP poly}

HPs loaded from best_hp_reg.json.

Usage
-----
  from test_evaluation_reg import run_all_test_reg
  test_results_reg = run_all_test_reg(
      graphs_PBSG, graphs_SMILES, graphs_SMILES_gl,
      X_fp_pooled_RU, X_fp_RU, X_fp_pooled_poly, X_fp_poly,
      y, df_reg, trainval_idx_reg, test_idx_reg, device,
      hp_path="best_hp_reg.json",
  )
"""

import json
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
import time

import torch
import joblib
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from scripts.CV import (
    build_gnn_model_reg, train_gnn_reg, evaluate_reg,
    scale_graphs, collect_subgroup_results_reg,
    set_seed, _graph_dims,
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TUNED HPs FROM JSON
# ══════════════════════════════════════════════════════════════════════════════

def _load_gnn_hp(hp_dict, key):
    """Convert JSON GNN entry to (SimpleNamespace args, int seed)."""
    hp = hp_dict[key]
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


def _build_classical(hp_dict, key):
    """Build classical model from JSON HP entry."""
    hp         = hp_dict[key]
    model_name = hp["model"]
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
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def full_metrics_reg(labels, preds):
    mae     = float(mean_absolute_error(labels, preds))
    rmse    = float(np.sqrt(np.mean((labels - preds) ** 2)))
    r2      = float(r2_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0
    nonzero = labels != 0
    mape    = float(np.mean(np.abs((labels[nonzero] - preds[nonzero]) /
                                    labels[nonzero])) * 100) if nonzero.any() else 0.0
    max_err = float(np.max(np.abs(labels - preds)))
    return {"MAE": mae, "RMSE": rmse, "R²": r2,
            "MAPE (%)": mape, "Max Error": max_err}


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
    """Retrain GNN regressor on full trainval, evaluate on test."""
    set_seed(seed)
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    train_graphs = [graphs[i] for i in trainval_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    y_train_raw = np.array([g.y.item() for g in train_graphs])
    y_scaler    = StandardScaler()
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    train_s = scale_graphs(train_graphs, y_scaler)
    test_s  = scale_graphs(test_graphs,  y_scaler)

    train_loader = DataLoader(train_s, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_s,  batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print(f"  Retraining {name} on full trainval (n={len(trainval_idx)})")
    print(f"  hidden={args.hidden}, layers={args.layers}, "
          f"dropout={args.dropout:.3f}, lr={args.lr:.5f}, seed={seed}")
    print(f"{'='*60}")

    model = build_gnn_model_reg(arch, in_ch, edge_dim, meta_dim,
                                args=args).to(device)
    t0             = time.time()
    model, history = train_gnn_reg(model, train_loader, test_loader, args, device)
    train_time_s   = time.time() - t0
    print(f"  Training time: {train_time_s/60:.2f} min")
  

    _, labels_s, preds_s = evaluate_reg(model, test_loader, device)
    labels = y_scaler.inverse_transform(labels_s.reshape(-1, 1)).flatten()
    preds  = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()

    m = full_metrics_reg(labels, preds)
    m["train_time_min"] = round(train_time_s / 60, 2)
    _print_metrics(name, m)
    subgroups = collect_subgroup_results_reg(df, labels, preds, list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name": name, "metrics": m, "preds": preds, "labels": labels,
        "subgroups": subgroups, "history": history,
        "model": model, "y_scaler": y_scaler,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_classical_test_reg(X, y, trainval_idx, test_idx,
                                 df, clf, name):
    """Retrain classical regressor on full trainval, evaluate on test."""
    print(f"\n── {name} ──────────────────────────────────────────")

    X_scaler = StandardScaler()
    X_train  = X_scaler.fit_transform(X[trainval_idx])
    X_test   = X_scaler.transform(X[test_idx])
    y_train  = y[trainval_idx]
    labels   = y[test_idx]

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time_s = time.time() - t0
    preds = clf.predict(X_test)

    m = full_metrics_reg(labels, preds)
    m["train_time_min"] = round(train_time_s / 60, 2)
    _print_metrics(name, m)
    subgroups = collect_subgroup_results_reg(df, labels, preds, list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name": name, "metrics": m, "preds": preds, "labels": labels,
        "subgroups": subgroups, "clf": clf, "X_scaler": X_scaler,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════

def run_all_test_reg(graphs_PBSG, graphs_SMILES, graphs_SMILES_gl,
                     X_fp_pooled_RU, X_fp_RU, X_fp_pooled_poly, X_fp_poly,
                     y, df_reg, trainval_idx, test_idx, device,
                     hp_path="best_hp_reg.json",
                     save_dir="saved_models_all"):
    """
    Run test evaluation for all 21 regression model/rep combinations.

    Parameters
    ----------
    hp_path  : path to best_hp_reg.json
    save_dir : directory to save best models (GINE PBSG + XGB pooled)
    """
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    with open(hp_path) as f:
        hp_dict = json.load(f)
    print(f"Loaded HPs from {hp_path}  ({len(hp_dict)} entries)")

    results = {}

    # ── GNN jobs ──────────────────────────────────────────────────────────────
    gnn_jobs = [
        # (arch, graphs, hp_key, result_key, display_name)
        ("gcn",   graphs_PBSG,      "gcn_pbsg",       "gcn_pbsg",       "GCN (PBSG)"),
        ("gcn",   graphs_SMILES,    "gcn_smiles",      "gcn_smiles",     "GCN (SMILES)"),
        ("gcn",   graphs_SMILES_gl, "gcn_smiles_gl",   "gcn_smiles_gl",  "GCN (SMILES+global)"),
        ("gine",  graphs_PBSG,      "gine_pbsg",       "gine_pbsg",      "GINE (PBSG)"),
        ("gine",  graphs_SMILES,    "gine_smiles",     "gine_smiles",    "GINE (SMILES)"),
        ("gine",  graphs_SMILES_gl, "gine_smiles_gl",  "gine_smiles_gl", "GINE (SMILES+global)"),
        ("gatv2", graphs_PBSG,      "gatv2_pbsg",      "gatv2_pbsg",     "GATv2 (PBSG)"),
        ("gatv2", graphs_SMILES,    "gatv2_smiles",    "gatv2_smiles",   "GATv2 (SMILES)"),
        ("gatv2", graphs_SMILES_gl, "gatv2_smiles_gl", "gatv2_smiles_gl","GATv2 (SMILES+global)"),
    ]

    for arch, graphs, hp_key, result_key, name in gnn_jobs:
        args, seed = _load_gnn_hp(hp_dict, hp_key)
        results[result_key] = evaluate_gnn_test_reg(
            arch, graphs, trainval_idx, test_idx,
            df_reg, device, args, seed, name
        )

    # ── Classical jobs ────────────────────────────────────────────────────────
    classical_jobs = [
        # (X, hp_key, result_key, display_name)
        (X_fp_pooled_RU,   "rr_fp_pooled_ru",   "rr_fp_pooled_ru",   "RR (FP+pooled RU)"),
        (X_fp_RU,          "rr_fp_ru",           "rr_fp_ru",          "RR (FP RU)"),
        (X_fp_pooled_poly, "rr_fp_pooled_poly",  "rr_fp_pooled_poly", "RR (FP+pooled poly)"),
        (X_fp_poly,        "rr_fp_poly",         "rr_fp_poly",        "RR (FP poly)"),
        (X_fp_pooled_RU,   "rf_fp_pooled_ru",   "rf_fp_pooled_ru",   "RF (FP+pooled RU)"),
        (X_fp_RU,          "rf_fp_ru",           "rf_fp_ru",          "RF (FP RU)"),
        (X_fp_pooled_poly, "rf_fp_pooled_poly",  "rf_fp_pooled_poly", "RF (FP+pooled poly)"),
        (X_fp_poly,        "rf_fp_poly",         "rf_fp_poly",        "RF (FP poly)"),
        (X_fp_pooled_RU,   "xgb_fp_pooled_ru",  "xgb_fp_pooled_ru",  "XGB (FP+pooled RU)"),
        (X_fp_RU,          "xgb_fp_ru",          "xgb_fp_ru",         "XGB (FP RU)"),
        (X_fp_pooled_poly, "xgb_fp_pooled_poly", "xgb_fp_pooled_poly","XGB (FP+pooled poly)"),
        (X_fp_poly,        "xgb_fp_poly",        "xgb_fp_poly",       "XGB (FP poly)"),
    ]

    for X, hp_key, result_key, name in classical_jobs:
        clf = _build_classical(hp_dict, hp_key)
        results[result_key] = evaluate_classical_test_reg(
            X, y, trainval_idx, test_idx, df_reg, clf, name
        )

    # ── Summary table sorted by R² ────────────────────────────────────────────
    metric_keys = ["MAE", "RMSE", "R²", "MAPE (%)", "train_time_min"]
    #metric_keys = ["MAE", "RMSE", "R²", "MAPE (%)"]
    col_w = 13
    sep   = "=" * (32 + col_w * len(metric_keys))

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]["metrics"]["R²"],
                            reverse=True)

    print(f"\n{sep}")
    print(f"  FINAL REGRESSION TEST SUMMARY  (sorted by R²)")
    print(sep)
    print(f"{'Model':<32}" + "".join(f"{k:<{col_w}}" for k in metric_keys))
    print("-" * (32 + col_w * len(metric_keys)))
    for _, r in sorted_results:
        row = f"{r['name']:<32}"
        for k in metric_keys:
            row += f"{r['metrics'][k]:<{col_w}.4f}"
        print(row)
    print(sep)

    # ── Save outputs ──────────────────────────────────────────────────────────
    rows = []
    for key, r in results.items():
        row = {"name": r["name"]}
        row.update(r["metrics"])
        rows.append(row)
    pd.DataFrame(rows).sort_values("R²", ascending=False)\
      .to_csv("test_results_reg_full.csv", index=False)
    print(f"\nSaved → test_results_reg_full.csv")

    with open("test_results_reg_subgroups_full.json", "w") as f:
        json.dump({k: v["subgroups"] for k, v in results.items()}, f, indent=2)
    print(f"Saved → test_results_reg_subgroups_full.json")

    with open("test_predictions_reg_full.json", "w") as f:
        json.dump({
            k: {"preds": v["preds"].tolist(), "labels": v["labels"].tolist()}
            for k, v in results.items()
        }, f, indent=2)
    print(f"Saved → test_predictions_reg_full.json")

    histories_out = {
        k: {m: [float(x) for x in vals] for m, vals in v["history"].items()}
        for k, v in results.items() if "history" in v
    }
    with open("test_training_histories_reg_full.json", "w") as f:
        json.dump(histories_out, f, indent=2)
    print(f"Saved → test_training_histories_reg_full.json")

    # ── Save best models for prediction pipeline ───────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    # GINE PBSG — state dict + y_scaler
    gine_model  = results["gine_pbsg"].get("model")
    gine_scaler = results["gine_pbsg"].get("y_scaler")
    if gine_model is not None:
        torch.save(gine_model.state_dict(),
                   f"{save_dir}/reg_gine_pbsg.pt")
        print(f"Saved → {save_dir}/reg_gine_pbsg.pt")
    if gine_scaler is not None:
        joblib.dump(gine_scaler, f"{save_dir}/reg_gine_pbsg_yscaler.pkl")
        print(f"Saved → {save_dir}/reg_gine_pbsg_yscaler.pkl")

    # XGB pooled models + scalers
    for key, model_fname, scaler_fname in [
        ("xgb_fp_pooled_ru",
         f"{save_dir}/reg_xgb_pooled_ru.pkl",
         f"{save_dir}/reg_xgb_pooled_ru_xscaler.pkl"),
        ("xgb_fp_pooled_poly",
         f"{save_dir}/reg_xgb_pooled_poly.pkl",
         f"{save_dir}/reg_xgb_pooled_poly_xscaler.pkl"),
    ]:
        clf    = results[key].get("clf")
        scaler = results[key].get("X_scaler")
        if clf is not None:
            joblib.dump(clf, model_fname)
            print(f"Saved → {model_fname}")
        if scaler is not None:
            joblib.dump(scaler, scaler_fname)
            print(f"Saved → {scaler_fname}")

    return results