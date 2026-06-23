"""
test_evaluation.py
==================
Final held-out test set evaluation for the four tuned models:
  - GINE (PBSG) tuned
  - GATv2 (PBSG) tuned
  - XGB (FP+pooled RU) tuned
  - XGB (FP+pooled poly) tuned

Workflow
--------
1. Retrain each model on the FULL trainval set (all 5 folds combined)
   using tuned HPs
2. Evaluate once on the held-out test set
3. Report full metrics + subgroup breakdown
4. Save results to test_results.csv and test_results_subgroups.json

Usage
-----
Run cells in order in a new notebook. Assumes the following are
already loaded as globals:
  graphs_PBSG, X_fp_pooled_RU, X_fp_pooled_poly,
  y, df, trainval_idx, test_idx, device
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
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve

from CV import (
    build_gnn_model, train_gnn, evaluate_gnn,
    full_metrics, optimal_threshold, collect_subgroup_results,
    set_seed, _graph_dims,
)


# ══════════════════════════════════════════════════════════════════════════════
# TUNED HPs — paste your final values here
# ══════════════════════════════════════════════════════════════════════════════

GINE_HP = SimpleNamespace(
    hidden=64, layers=4, dropout=0.2643, lr=3.213e-3,
    weight_decay=4.7e-5, epochs=200, patience=30, batch_size=32
)
GINE_SEED = 10

GATV2_HP = SimpleNamespace(
    hidden=128, layers=3, dropout=0.2062, lr=9.433e-3,
    weight_decay=3.484e-4, epochs=200, patience=30, batch_size=32
)
GATV2_SEED = 0

XGB_RU_PARAMS = dict(
    n_estimators=54, max_depth=7, learning_rate=0.07499,
    subsample=0.8875, colsample_bytree=0.7256,
    min_child_weight=1, reg_alpha=7.17e-5, reg_lambda=8.16e-4,
)

XGB_POLY_PARAMS = dict(
    n_estimators=460, max_depth=7, learning_rate=0.04260,
    subsample=0.8096, colsample_bytree=0.9149,
    min_child_weight=2, reg_alpha=0.10609, reg_lambda=0.01044,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _xgb_full_params(params, y_trainval):
    n_neg = (y_trainval == 0).sum()
    n_pos = (y_trainval == 1).sum()
    return {
        **params,
        "scale_pos_weight": float(n_neg / n_pos),
        "eval_metric":      "logloss",
        "tree_method":      "hist",
        "n_jobs":           -1,
        "random_state":     42,
        "verbosity":        0,
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
        print(f"  {'Group':<20s} {'n':>5} {'Acc':>8} {'F1':>8}")
        print(f"  {'-'*45}")
        for grp, m in sorted(groups.items()):
            print(f"  {str(grp):<20s} {m['n']:>5} {m['acc']:>8.4f} {m['f1']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# GNN TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gnn_test(arch, graphs, trainval_idx, test_idx, df,
                      device, args, seed, name):
    """
    Retrain GNN on full trainval, evaluate on test.

    Parameters
    ----------
    arch        : "gine" | "gatv2"
    graphs      : full graph list
    trainval_idx: indices for training
    test_idx    : indices for test
    df          : full dataframe
    device      : torch device
    args        : SimpleNamespace of tuned HPs
    seed        : int
    name        : display name

    Returns
    -------
    dict with metrics, probs, preds, labels, subgroups
    """
    set_seed(seed)
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    train_loader = DataLoader(
        [graphs[i] for i in trainval_idx],
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        [graphs[i] for i in test_idx],
        batch_size=args.batch_size, shuffle=False,
    )

    print(f"\n{'='*60}")
    print(f"  Retraining {name} on full trainval (n={len(trainval_idx)})")
    print(f"{'='*60}")

    model = build_gnn_model(arch, in_ch, edge_dim, meta_dim,
                            args=args).to(device)
    model, history = train_gnn(model, train_loader, test_loader, args, device)

    _, labels, _, probs, _, _ = evaluate_gnn(model, test_loader, device)
    thresh = optimal_threshold(labels, probs)
    preds  = (probs >= thresh).astype(int)
    m      = full_metrics(labels, preds, probs)

    _print_metrics(name, m)
    subgroups = collect_subgroup_results(
        df, labels, preds, list(test_idx)
    )
    _print_subgroups(subgroups)


    return {
        "name":      name,
        "metrics":   m,
        "probs":     probs,
        "preds":     preds,
        "labels":    labels,
        "thresh":    thresh,
        "subgroups": subgroups,
        "history":   history,
        "model":     model,   # keep for saving
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_xgb_test(X, y, trainval_idx, test_idx, df, params, name):
    """
    Retrain XGB on full trainval, evaluate on test.

    Parameters
    ----------
    X           : full feature array
    y           : full label array
    trainval_idx: indices for training
    test_idx    : indices for test
    df          : full dataframe
    params      : dict of tuned XGB HPs (without scale_pos_weight etc.)
    name        : display name

    Returns
    -------
    dict with metrics, probs, preds, labels, subgroups
    """
    print(f"\n{'='*60}")
    print(f"  Retraining {name} on full trainval (n={len(trainval_idx)})")
    print(f"{'='*60}")

    full_params = _xgb_full_params(params, y[trainval_idx])
    clf = XGBClassifier(**full_params)
    clf.fit(X[trainval_idx], y[trainval_idx])

    probs  = clf.predict_proba(X[test_idx])[:, 1]
    labels = y[test_idx]
    thresh = optimal_threshold(labels, probs)
    preds  = (probs >= thresh).astype(int)
    m      = full_metrics(labels, preds, probs)

    _print_metrics(name, m)
    subgroups = collect_subgroup_results(
        df, labels, preds, list(test_idx)
    )
    _print_subgroups(subgroups)

    return {
        "name":      name,
        "metrics":   m,
        "probs":     probs,
        "preds":     preds,
        "labels":    labels,
        "thresh":    thresh,
        "subgroups": subgroups,
        "clf":       clf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════

def run_all_test(graphs_PBSG, X_fp_pooled_RU, X_fp_pooled_poly,
                 y, df, trainval_idx, test_idx, device):
    """
    Run test evaluation for all four tuned models.

    Returns
    -------
    dict of results keyed by model name
    """
    results = {}

    # ── GNNs ──────────────────────────────────────────────────────
    results["gine_pbsg"] = evaluate_gnn_test(
        "gine", graphs_PBSG, trainval_idx, test_idx, df,
        device, GINE_HP, GINE_SEED, "GINE (PBSG) tuned"
    )
    results["gatv2_pbsg"] = evaluate_gnn_test(
        "gatv2", graphs_PBSG, trainval_idx, test_idx, df,
        device, GATV2_HP, GATV2_SEED, "GATv2 (PBSG) tuned"
    )

    # ── Classical ─────────────────────────────────────────────────
    results["xgb_ru"] = evaluate_xgb_test(
        X_fp_pooled_RU, y, trainval_idx, test_idx, df,
        XGB_RU_PARAMS, "XGB (FP+pooled RU) tuned"
    )
    results["xgb_poly"] = evaluate_xgb_test(
        X_fp_pooled_poly, y, trainval_idx, test_idx, df,
        XGB_POLY_PARAMS, "XGB (FP+pooled poly) tuned"
    )

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FINAL TEST SUMMARY")
    print(f"{'='*80}")
    metric_keys = ["Accuracy", "Macro F1", "AUROC", "AUPRC",
                   "No Tm Recall", "Has Tm Recall"]
    col_w = 22
    header = f"{'Model':<28}" + "".join(f"{k:<{col_w}}" for k in metric_keys)
    print(header)
    print("-" * (28 + col_w * len(metric_keys)))
    for key, r in results.items():
        row = f"{r['name']:<28}"
        for k in metric_keys:
            row += f"{r['metrics'][k]:<{col_w}.4f}"
        print(row)
    print("=" * (28 + col_w * len(metric_keys)))

    # ── Save ──────────────────────────────────────────────────────
    # Metrics CSV
    rows = []
    for key, r in results.items():
        row = {"name": r["name"]}
        row.update(r["metrics"])
        rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv("test_results.csv", index=False)
    print(f"\nSaved → test_results.csv")

    # Subgroups JSON
    subgroups_out = {k: v["subgroups"] for k, v in results.items()}
    with open("test_results_subgroups.json", "w") as f:
        json.dump(subgroups_out, f, indent=2)
    print(f"Saved → test_results_subgroups.json")

    # Predictions (for ROC/PR curve plotting later)
    preds_out = {
        k: {
            "probs":  v["probs"].tolist(),
            "preds":  v["preds"].tolist(),
            "labels": v["labels"].tolist(),
            "thresh": float(v["thresh"]),
        }
        for k, v in results.items()
    }
    with open("test_predictions.json", "w") as f:
        json.dump(preds_out, f, indent=2)
    print(f"Saved → test_predictions.json")

    # Save training histories for learning curve plots
    histories_out = {}
    for key, r in results.items():
        if "history" in r:  # GNNs only
            histories_out[key] = {
                k: [float(x) for x in vals]
                for k, vals in r["history"].items()
            }

    with open("test_training_histories.json", "w") as f:
        json.dump(histories_out, f, indent=2)
    print("Saved → test_training_histories.json")

    # ── Save trained models for prediction pipeline ────────────────
    os.makedirs("saved_models_top", exist_ok=True)

    # GINE state dict
    gine_model = results["gine_pbsg"].get("model")
    if gine_model is not None:
        torch.save(gine_model.state_dict(),
                   "saved_models_top/cls_gine_pbsg.pt")
        print("Saved → saved_models_top/cls_gine_pbsg.pt")

    # GATv2 state dict
    gatv2_model = results["gatv2_pbsg"].get("model")
    if gatv2_model is not None:
        torch.save(gatv2_model.state_dict(),
                   "saved_models_top/cls_gatv2_pbsg.pt")
        print("Saved → saved_models_top/cls_gatv2_pbsg.pt")

    # XGB classifiers
    for key, fname in [
        ("xgb_ru",   "saved_models_top/cls_xgb_pooled_ru.pkl"),
        ("xgb_poly", "saved_models_top/cls_xgb_pooled_poly.pkl"),
    ]:
        clf = results[key].get("clf")
        if clf is not None:
            joblib.dump(clf, fname)
            print(f"Saved → {fname}")

    # Optimal thresholds
    thresholds = {k: float(v["thresh"]) for k, v in results.items()}
    with open("saved_models_top/cls_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Saved → saved_models_top/cls_thresholds.json")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK USAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# In your new notebook, after loading all data:
#
# from test_evaluation import run_all_test
#
# test_results = run_all_test(
#     graphs_PBSG      = graphs_PBSG,
#     X_fp_pooled_RU   = X_fp_pooled_RU,
#     X_fp_pooled_poly = X_fp_pooled_poly,
#     y                = y,
#     df               = df,
#     trainval_idx     = trainval_idx,
#     test_idx         = test_idx,
#     device           = device,
# )
#
# Three files saved:
#   test_results.csv            — metrics table for paper
#   test_results_subgroups.json — subgroup breakdown for paper
#   test_predictions.json       — probs/preds/labels for ROC/PR plots