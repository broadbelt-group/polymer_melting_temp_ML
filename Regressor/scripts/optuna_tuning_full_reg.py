"""
optuna_tuning_full_reg.py
=========================
Full HP tuning for ALL 27 regression model/rep combinations
with layers as a tunable hyperparameter.

Saves to optuna_tuning_full_reg.db (separate from original).
Best HPs saved to best_hp_reg.json.

Models tuned:
  GNNs (3 archs × 3 reps = 9):
    GCN, GINE, GATv2 × PBSG, SMILES, SMILES+global

  Classical (3 models × 6 reps = 18):
    RR, RF, XGB × FP+pooled RU, FP+pooled poly, FP RU, FP poly,
                  FP SMILES, FP SMILES+global

Objective: mean_r2 - 0.5 * std_r2

Resumable: all studies persist to optuna_tuning_full_reg.db via SQLite.

Usage
-----
  from optuna_tuning_full_reg import tune_all_reg
  tune_all_reg(
      graphs_PBSG      = graphs_PBSG,
      graphs_SMILES    = graphs_SMILES,
      graphs_SMILES_gl = graphs_SMILES_gl,
      X_fp_pooled_RU   = X_fp_pooled_RU,
      X_fp_pooled_poly = X_fp_pooled_poly,
      X_fp_RU          = X_fp_RU,
      X_fp_poly        = X_fp_poly,
      X_fp_SMILES      = X_fp_SMILES,
      X_fp_SMILES_gl   = X_fp_SMILES_gl,
      y=y, folds=folds_reg, df=df_reg, device=device,
      n_trials=30,
  )
"""

import json
import warnings
import numpy as np
import optuna
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from CV import (
    build_gnn_model_reg, train_gnn_reg, evaluate_reg,
    scale_graphs, set_seed, _graph_dims,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB      = "sqlite:///optuna_tuning_full_reg.db"
HP_PATH = "best_hp_reg.json"

SEED_CANDIDATES = [0, 1, 2, 3, 4, 10, 42]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_hp():
    try:
        with open(HP_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_hp(hp_dict):
    with open(HP_PATH, "w") as f:
        json.dump(hp_dict, f, indent=2)
    print(f"  Saved → {HP_PATH}")

def _run_study(study_name, objective, n_trials):
    study = optuna.create_study(
        direction      = "maximize",
        study_name     = study_name,
        sampler        = optuna.samplers.TPESampler(seed=42),
        storage        = DB,
        load_if_exists = True,
    )
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = n_trials - completed

    print(f"\n{'='*60}")
    print(f"  Tuning: {study_name}")
    print(f"  Completed: {completed}/{n_trials}  |  Running: {remaining} more")
    print(f"{'='*60}")

    if remaining > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=remaining,
                           show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best trial #{best.number}  objective={best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:25s}: {v}")
    return study

def _fold_r2(labels, preds):
    return float(r2_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_gnn_hp(trial):
    """GNN HP search space — layers free."""
    return SimpleNamespace(
        hidden       = trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        layers       = trial.suggest_int("layers", 2, 5),
        dropout      = trial.suggest_float("dropout", 0.0, 0.5),
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        epochs       = 300,
        patience     = 50,
        batch_size   = 32,
    )

def _suggest_xgb_hp(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3,
                                                log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0,
                                                log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0,
                                                log=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVES
# ══════════════════════════════════════════════════════════════════════════════

def _gnn_objective(arch, graphs, folds, device):
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    def objective(trial):
        hp   = _suggest_gnn_hp(trial)
        seed = trial.suggest_categorical("seed", SEED_CANDIDATES)
        set_seed(seed)

        fold_r2s = []
        for fold in folds:
            train_graphs = [graphs[i] for i in fold["train"]]
            val_graphs   = [graphs[i] for i in fold["val"]]

            y_train = np.array([g.y.item() for g in train_graphs])
            scaler  = StandardScaler()
            scaler.fit(y_train.reshape(-1, 1))

            train_s = scale_graphs(train_graphs, scaler)
            val_s   = scale_graphs(val_graphs,   scaler)

            train_loader = DataLoader(train_s, batch_size=hp.batch_size,
                                      shuffle=True)
            val_loader   = DataLoader(val_s,   batch_size=hp.batch_size,
                                      shuffle=False)

            model = build_gnn_model_reg(
                arch, in_ch, edge_dim, meta_dim, args=hp
            ).to(device)
            model, _ = train_gnn_reg(model, train_loader, val_loader,
                                      hp, device)
            _, labels_s, preds_s = evaluate_reg(model, val_loader, device)

            labels = scaler.inverse_transform(
                labels_s.reshape(-1, 1)).flatten()
            preds  = scaler.inverse_transform(
                preds_s.reshape(-1, 1)).flatten()
            fold_r2s.append(_fold_r2(labels, preds))

        return float(np.mean(fold_r2s)) - 0.5 * float(np.std(fold_r2s))

    return objective


def _rr_objective(X, y, folds):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
        fold_r2s = []
        for fold in folds:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X[fold["train"]])
            X_val   = scaler.transform(X[fold["val"]])
            clf = Ridge(alpha=alpha)
            clf.fit(X_train, y[fold["train"]])
            preds = clf.predict(X_val)
            fold_r2s.append(_fold_r2(y[fold["val"]], preds))
        return float(np.mean(fold_r2s)) - 0.5 * float(np.std(fold_r2s))
    return objective


def _rf_objective(X, y, folds):
    def objective(trial):
        n_est       = trial.suggest_int("n_estimators", 50, 500)
        max_depth   = trial.suggest_int("max_depth", 3, 20)
        min_samples = trial.suggest_int("min_samples_split", 2, 20)
        max_feat    = trial.suggest_categorical(
            "max_features", ["sqrt", "log2"])
        fold_r2s = []
        for fold in folds:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X[fold["train"]])
            X_val   = scaler.transform(X[fold["val"]])
            clf = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth,
                min_samples_split=min_samples, max_features=max_feat,
                n_jobs=-1, random_state=42,
            )
            clf.fit(X_train, y[fold["train"]])
            preds = clf.predict(X_val)
            fold_r2s.append(_fold_r2(y[fold["val"]], preds))
        return float(np.mean(fold_r2s)) - 0.5 * float(np.std(fold_r2s))
    return objective


def _xgb_objective(X, y, folds):
    def objective(trial):
        params = _suggest_xgb_hp(trial)
        params.update({
            "eval_metric":  "rmse",
            "tree_method":  "hist",
            "n_jobs":       -1,
            "random_state": 42,
            "verbosity":    0,
        })
        fold_r2s = []
        for fold in folds:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X[fold["train"]])
            X_val   = scaler.transform(X[fold["val"]])
            clf = XGBRegressor(**params)
            clf.fit(X_train, y[fold["train"]])
            preds = clf.predict(X_val)
            fold_r2s.append(_fold_r2(y[fold["val"]], preds))
        return float(np.mean(fold_r2s)) - 0.5 * float(np.std(fold_r2s))
    return objective


# ══════════════════════════════════════════════════════════════════════════════
# HP EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def _hp_from_gnn_study(study, arch, rep):
    p = study.best_trial.params
    return {
        "type":         "gnn",
        "arch":         arch,
        "rep":          rep,
        "seed":         p["seed"],
        "hidden":       p["hidden_dim"],
        "layers":       p["layers"],
        "dropout":      round(p["dropout"], 6),
        "lr":           round(p["lr"], 8),
        "weight_decay": round(p["weight_decay"], 8),
        "epochs":       300,
        "patience":     50,
        "batch_size":   32,
    }

def _hp_from_rr_study(study):
    p = study.best_trial.params
    return {
        "type":  "classical",
        "model": "rr",
        "alpha": round(p["alpha"], 6),
    }

def _hp_from_rf_study(study):
    p = study.best_trial.params
    return {
        "type":              "classical",
        "model":             "rf",
        "n_estimators":      p["n_estimators"],
        "max_depth":         p["max_depth"],
        "min_samples_split": p["min_samples_split"],
        "max_features":      p["max_features"],
    }

def _hp_from_xgb_study(study):
    p = study.best_trial.params
    return {
        "type":             "classical",
        "model":            "xgb",
        "n_estimators":     p["n_estimators"],
        "max_depth":        p["max_depth"],
        "learning_rate":    round(p["learning_rate"], 8),
        "subsample":        round(p["subsample"], 8),
        "colsample_bytree": round(p["colsample_bytree"], 8),
        "min_child_weight": p["min_child_weight"],
        "reg_alpha":        round(p["reg_alpha"], 8),
        "reg_lambda":       round(p["reg_lambda"], 8),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def tune_all_reg(
    graphs_PBSG,
    graphs_SMILES,
    graphs_SMILES_gl,
    X_fp_pooled_RU,
    X_fp_pooled_poly,
    X_fp_RU,
    X_fp_poly,
    y,
    folds,
    df,
    device,
    n_trials = 30,
):
    hp = _load_hp()

    # ── GNNs (9 combos) ───────────────────────────────────────────────────────
    gnn_jobs = [
        # (arch, graphs, study_name, hp_key, rep_label)
        ("gcn",   graphs_PBSG,      "reg_gcn_pbsg",        "gcn_pbsg",        "PBSG"),
        ("gcn",   graphs_SMILES,    "reg_gcn_smiles",      "gcn_smiles",      "SMILES"),
        ("gcn",   graphs_SMILES_gl, "reg_gcn_smiles_gl",   "gcn_smiles_gl",   "SMILES+global"),
        ("gine",  graphs_PBSG,      "reg_gine_pbsg",       "gine_pbsg",       "PBSG"),
        ("gine",  graphs_SMILES,    "reg_gine_smiles",     "gine_smiles",     "SMILES"),
        ("gine",  graphs_SMILES_gl, "reg_gine_smiles_gl",  "gine_smiles_gl",  "SMILES+global"),
        ("gatv2", graphs_PBSG,      "reg_gatv2_pbsg",      "gatv2_pbsg",      "PBSG"),
        ("gatv2", graphs_SMILES,    "reg_gatv2_smiles",    "gatv2_smiles",    "SMILES"),
        ("gatv2", graphs_SMILES_gl, "reg_gatv2_smiles_gl", "gatv2_smiles_gl", "SMILES+global"),
    ]

    for arch, graphs, study_name, hp_key, rep in gnn_jobs:
        study = _run_study(
            study_name,
            _gnn_objective(arch, graphs, folds, device),
            n_trials,
        )
        hp[hp_key] = _hp_from_gnn_study(study, arch, rep)
        _save_hp(hp)

    # ── RR (6 reps) ───────────────────────────────────────────────────────────
    rr_jobs = [
        ("reg_rr_fp_pooled_ru",   X_fp_pooled_RU,   "rr_fp_pooled_ru",   "FP+pooled RU"),
        ("reg_rr_fp_pooled_poly", X_fp_pooled_poly, "rr_fp_pooled_poly", "FP+pooled poly"),
        ("reg_rr_fp_ru",          X_fp_RU,          "rr_fp_ru",          "FP RU"),
        ("reg_rr_fp_poly",        X_fp_poly,        "rr_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in rr_jobs:
        study = _run_study(study_name, _rr_objective(X, y, folds),
                           n_trials=15)
        hp[hp_key] = _hp_from_rr_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    # ── RF (4 reps) ───────────────────────────────────────────────────────────
    rf_jobs = [
        ("reg_rf_fp_pooled_ru",   X_fp_pooled_RU,   "rf_fp_pooled_ru",   "FP+pooled RU"),
        ("reg_rf_fp_pooled_poly", X_fp_pooled_poly, "rf_fp_pooled_poly", "FP+pooled poly"),
        ("reg_rf_fp_ru",          X_fp_RU,          "rf_fp_ru",          "FP RU"),
        ("reg_rf_fp_poly",        X_fp_poly,        "rf_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in rf_jobs:
        study = _run_study(study_name, _rf_objective(X, y, folds),
                           n_trials=n_trials)
        hp[hp_key] = _hp_from_rf_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    # ── XGB (4 reps) ──────────────────────────────────────────────────────────
    xgb_jobs = [
        ("reg_xgb_fp_pooled_ru",   X_fp_pooled_RU,   "xgb_fp_pooled_ru",   "FP+pooled RU"),
        ("reg_xgb_fp_pooled_poly", X_fp_pooled_poly, "xgb_fp_pooled_poly", "FP+pooled poly"),
        ("reg_xgb_fp_ru",          X_fp_RU,          "xgb_fp_ru",          "FP RU"),
        ("reg_xgb_fp_poly",        X_fp_poly,        "xgb_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in xgb_jobs:
        study = _run_study(study_name, _xgb_objective(X, y, folds),
                           n_trials=n_trials)
        hp[hp_key] = _hp_from_xgb_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    print("\n✓ All 21 regression models tuned.")
    print(f"  DB:  {DB}")
    print(f"  HPs: {HP_PATH}")
    return hp


def load_best_hp(path=HP_PATH):
    with open(path) as f:
        return json.load(f)


def print_study_summary(n_top=3):
    studies = optuna.get_all_study_summaries(storage=DB)
    for s in sorted(studies, key=lambda x: x.study_name):
        print(f"\n── {s.study_name}  best={s.best_trial.value:.4f}")