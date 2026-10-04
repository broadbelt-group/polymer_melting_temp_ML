
import json
import os
import time
import numpy as np
import pandas as pd
from types import SimpleNamespace

import torch
import joblib
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scripts.CV import (
    build_gnn_model, train_gnn, evaluate_gnn,
    full_metrics, optimal_threshold, collect_subgroup_results,
    set_seed, _graph_dims,
)


# LOAD TUNED HPs FROM JSON

def _load_gnn_hp(hp_dict, key):
    hp = hp_dict[key]
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


def _build_classical(hp_dict, key, scale_pos_weight=1.0):
    hp         = hp_dict[key]
    model_name = hp["model"]
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
            eval_metric      = "logloss",
            tree_method      = "hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


# PRINT HELPERS

def _print_metrics(name, metrics):
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS: {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22s}: {v:.4f}")
        else:
            print(f"  {k:<22s}: {v}")


def _print_subgroups(subgroups):
    for col, groups in subgroups.items():
        print(f"\n  Subgroup: {col}")
        print(f"  {'Group':<20s} {'n':>5} {'Acc':>8} {'F1':>8}")
        print(f"  {'-'*45}")
        for grp, m in sorted(groups.items()):
            print(f"  {str(grp):<20s} {m['n']:>5} "
                  f"{m['acc']:>8.4f} {m['f1']:>8.4f}")


# GNN TEST EVALUATION

def evaluate_gnn_test(arch, graphs, trainval_idx, test_idx, df,
                      device, args, seed, name):
    set_seed(seed)
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    edge_dim=0
    meta_dim=0

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
    print(f"  hidden={args.hidden}, layers={args.layers}, "
          f"dropout={args.dropout:.3f}, lr={args.lr:.5f}, seed={seed}")
    print(f"{'='*60}")

    model          = build_gnn_model(arch, in_ch, edge_dim=0, meta_dim=0,
                                     args=args).to(device)
    t0             = time.time()
    model, history = train_gnn(model, train_loader, test_loader, args, device)
    train_time_s   = time.time() - t0
    train_time_min = round(train_time_s / 60, 2)
    print(f"  Training time: {train_time_min:.2f} min")

    _, labels, _, probs, _, _ = evaluate_gnn(model, test_loader, device)
    thresh = optimal_threshold(labels, probs)
    preds  = (probs >= thresh).astype(int)
    m      = full_metrics(labels, preds, probs)
    m["train_time_min"] = train_time_min

    _print_metrics(name, m)
    subgroups = collect_subgroup_results(df, labels, preds, list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name": name, "metrics": m, "probs": probs, "preds": preds,
        "labels": labels, "thresh": thresh, "subgroups": subgroups,
        "history": history, "model": model,
    }


# CLASSICAL TEST EVALUATION

def evaluate_classical_test(X, y, trainval_idx, test_idx, df, clf, name):
    print(f"\n── {name} ──────────────────────────────────────────")

    t0             = time.time()
    clf.fit(X[trainval_idx], y[trainval_idx])
    train_time_s   = time.time() - t0
    train_time_min = round(train_time_s / 60, 2)

    probs  = clf.predict_proba(X[test_idx])[:, 1]
    labels = y[test_idx]
    thresh = optimal_threshold(labels, probs)
    preds  = (probs >= thresh).astype(int)
    m      = full_metrics(labels, preds, probs)
    m["train_time_min"] = train_time_min

    _print_metrics(name, m)
    subgroups = collect_subgroup_results(df, labels, preds, list(test_idx))
    _print_subgroups(subgroups)

    return {
        "name": name, "metrics": m, "probs": probs, "preds": preds,
        "labels": labels, "thresh": thresh, "subgroups": subgroups,
        "clf": clf,
    }


# RUN ALL

def run_all_test(X_fp_RU, X_fp_poly,
                 y, df, trainval_idx, test_idx, device,
                 hp_path="best_hp_full.json",
                 save_dir="saved_models_all"):
    trainval_idx = np.array(trainval_idx)
    test_idx     = np.array(test_idx)

    with open(hp_path) as f:
        hp_dict = json.load(f)
    print(f"Loaded HPs from {hp_path}  ({len(hp_dict)} entries)")

    # Class imbalance weight for XGB
    n_neg = (y[trainval_idx] == 0).sum()
    n_pos = (y[trainval_idx] == 1).sum()
    scale_pos_weight = float(n_neg / n_pos)

    results = {}


    # ── Classical jobs ────────────────────────────────────────────────────────
    classical_jobs = [
        # (X, hp_key, result_key, display_name)
        (X_fp_RU,          "xgb_fp_ru",          "xgb_fp_ru",         "XGB (FP RU)"),
        (X_fp_poly,        "xgb_fp_poly",        "xgb_fp_poly",       "XGB (FP poly)"),
    ]

    for X, hp_key, result_key, name in classical_jobs:
        clf = _build_classical(hp_dict, hp_key, scale_pos_weight)
        results[result_key] = evaluate_classical_test(
            X, y, trainval_idx, test_idx, df, clf, name
        )

    # ── Summary table sorted by AUPRC ─────────────────────────────────────────
    metric_keys = ["Accuracy", "Macro F1", "AUROC", "AUPRC",
                   "Has Tm Recall", "No Tm Recall", "train_time_min"]
    col_w = 16
    sep   = "=" * (32 + col_w * len(metric_keys))

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]["metrics"]["AUPRC"],
                            reverse=True)

    print(f"\n{sep}")
    print(f"  FINAL CLASSIFICATION TEST SUMMARY  (sorted by AUPRC)")
    print(sep)
    print(f"{'Model':<32}" + "".join(f"{k:<{col_w}}" for k in metric_keys))
    print("-" * (32 + col_w * len(metric_keys)))
    for _, r in sorted_results:
        row = f"{r['name']:<32}"
        for k in metric_keys:
            v = r["metrics"].get(k, 0.0)
            row += f"{v:<{col_w}.4f}"
        print(row)
    print(sep)

    # ── Save outputs ──────────────────────────────────────────────────────────
    rows = []
    for key, r in results.items():
        row = {"name": r["name"]}
        row.update(r["metrics"])
        rows.append(row)
    pd.DataFrame(rows).sort_values("AUPRC", ascending=False)\
      .to_csv("test_results-ablation.csv", index=False)
    print(f"\nSaved → test_results-ablation.csv")

    with open("test_results_subgroups-ablation.json", "w") as f:
        json.dump({k: v["subgroups"] for k, v in results.items()}, f, indent=2)
    print(f"Saved → test_results_subgroups-ablation.json")

    with open("test_predictions-ablation.json", "w") as f:
        json.dump({
            k: {
                "probs":  v["probs"].tolist(),
                "preds":  v["preds"].tolist(),
                "labels": v["labels"].tolist(),
                "thresh": float(v["thresh"]),
            }
            for k, v in results.items()
        }, f, indent=2)
    print(f"Saved → test_predictions-ablation.json")

    histories_out = {
        k: {m: [float(x) for x in vals] for m, vals in v["history"].items()}
        for k, v in results.items() if "history" in v
    }
    with open("test_training_histories-ablation.json", "w") as f:
        json.dump(histories_out, f, indent=2)
    print(f"Saved → test_training_histories-ablation.json")


    return results