
import json
import warnings
import numpy as np
import optuna
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from CV import (
    build_gnn_model, train_gnn, evaluate_gnn,
    compute_metrics, optimal_threshold,
    set_seed, _graph_dims,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB      = "sqlite:///optuna_tuning_full.db"
HP_PATH = "best_hp_full.json"

SEED_CANDIDATES = [0, 1, 2, 3, 4, 10, 42]


# HELPERS

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


# SEARCH SPACES

def _suggest_gnn_hp(trial):
    return SimpleNamespace(
        hidden       = trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        layers       = trial.suggest_int("layers", 2, 5),
        dropout      = trial.suggest_float("dropout", 0.0, 0.5),
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        epochs       = 200,
        patience     = 30,
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


# OBJECTIVES

def _gnn_objective(arch, graphs, folds, device):
    in_ch, edge_dim, meta_dim = _graph_dims(graphs)

    def objective(trial):
        hp   = _suggest_gnn_hp(trial)
        seed = trial.suggest_categorical("seed", SEED_CANDIDATES)
        set_seed(seed)

        fold_auprcs = []
        for fold in folds:
            train_loader = DataLoader(
                [graphs[i] for i in fold["train"]],
                batch_size=hp.batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                [graphs[i] for i in fold["val"]],
                batch_size=hp.batch_size, shuffle=False,
            )
            model = build_gnn_model(
                arch, in_ch, edge_dim, meta_dim, args=hp
            ).to(device)
            model, _ = train_gnn(model, train_loader, val_loader,
                                  hp, device)
            _, labels, _, probs, _, _ = evaluate_gnn(
                model, val_loader, device)

            if len(np.unique(labels)) < 2:
                fold_auprcs.append(0.5)
                continue

            thresh = optimal_threshold(labels, probs)
            preds  = (probs >= thresh).astype(int)
            m      = compute_metrics(labels, preds, probs)
            fold_auprcs.append(m["auprc"])

        return float(np.mean(fold_auprcs)) - 0.5 * float(np.std(fold_auprcs))

    return objective


def _lr_objective(X, y, folds):
    def objective(trial):
        C        = trial.suggest_float("C", 1e-4, 1e2, log=True)
        max_iter = trial.suggest_categorical("max_iter", [200, 500, 1000])
        fold_auprcs = []
        for fold in folds:
            clf = LogisticRegression(
                C=C, max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs", random_state=42,
            )
            clf.fit(X[fold["train"]], y[fold["train"]])
            probs = clf.predict_proba(X[fold["val"]])[:, 1]
            fold_auprcs.append(
                average_precision_score(y[fold["val"]], probs))
        return float(np.mean(fold_auprcs)) - 0.5 * float(np.std(fold_auprcs))
    return objective


def _rf_objective(X, y, folds):
    def objective(trial):
        n_est       = trial.suggest_int("n_estimators", 50, 500)
        max_depth   = trial.suggest_int("max_depth", 3, 15)
        min_samples = trial.suggest_int("min_samples_split", 2, 20)
        max_feat    = trial.suggest_categorical(
            "max_features", ["sqrt", "log2"])
        fold_auprcs = []
        for fold in folds:
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth,
                min_samples_split=min_samples, max_features=max_feat,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )
            clf.fit(X[fold["train"]], y[fold["train"]])
            probs = clf.predict_proba(X[fold["val"]])[:, 1]
            fold_auprcs.append(
                average_precision_score(y[fold["val"]], probs))
        return float(np.mean(fold_auprcs)) - 0.5 * float(np.std(fold_auprcs))
    return objective


def _xgb_objective(X, y, folds):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw   = float(n_neg / n_pos)

    def objective(trial):
        params = _suggest_xgb_hp(trial)
        params.update({
            "scale_pos_weight": spw,
            "eval_metric":      "logloss",
            "tree_method":      "hist",
            "n_jobs":           -1,
            "random_state":     42,
            "verbosity":        0,
        })
        fold_auprcs = []
        for fold in folds:
            clf = XGBClassifier(**params)
            clf.fit(X[fold["train"]], y[fold["train"]])
            probs = clf.predict_proba(X[fold["val"]])[:, 1]
            fold_auprcs.append(
                average_precision_score(y[fold["val"]], probs))
        return float(np.mean(fold_auprcs)) - 0.5 * float(np.std(fold_auprcs))
    return objective


# HP EXTRACTORS

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
        "epochs":       200,
        "patience":     30,
        "batch_size":   32,
    }

def _hp_from_lr_study(study):
    p = study.best_trial.params
    return {
        "type":     "classical",
        "model":    "lr",
        "C":        round(p["C"], 6),
        "max_iter": p["max_iter"],
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


# MAIN

def tune_all_cls(
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
        ("gcn",   graphs_PBSG,      "cls_gcn_pbsg",        "gcn_pbsg",        "PBSG"),
        ("gcn",   graphs_SMILES,    "cls_gcn_smiles",      "gcn_smiles",      "SMILES"),
        ("gcn",   graphs_SMILES_gl, "cls_gcn_smiles_gl",   "gcn_smiles_gl",   "SMILES+global"),
        ("gine",  graphs_PBSG,      "cls_gine_pbsg",       "gine_pbsg",       "PBSG"),
        ("gine",  graphs_SMILES,    "cls_gine_smiles",     "gine_smiles",     "SMILES"),
        ("gine",  graphs_SMILES_gl, "cls_gine_smiles_gl",  "gine_smiles_gl",  "SMILES+global"),
        ("gatv2", graphs_PBSG,      "cls_gatv2_pbsg",      "gatv2_pbsg",      "PBSG"),
        ("gatv2", graphs_SMILES,    "cls_gatv2_smiles",    "gatv2_smiles",    "SMILES"),
        ("gatv2", graphs_SMILES_gl, "cls_gatv2_smiles_gl", "gatv2_smiles_gl", "SMILES+global"),
    ]

    for arch, graphs, study_name, hp_key, rep in gnn_jobs:
        study = _run_study(
            study_name,
            _gnn_objective(arch, graphs, folds, device),
            n_trials,
        )
        hp[hp_key] = _hp_from_gnn_study(study, arch, rep)
        _save_hp(hp)

    # ── LR (4 reps) ───────────────────────────────────────────────────────────
    lr_jobs = [
        ("cls_lr_fp_pooled_ru",   X_fp_pooled_RU,   "lr_fp_pooled_ru",   "FP+pooled RU"),
        ("cls_lr_fp_pooled_poly", X_fp_pooled_poly, "lr_fp_pooled_poly", "FP+pooled poly"),
        ("cls_lr_fp_ru",          X_fp_RU,          "lr_fp_ru",          "FP RU"),
        ("cls_lr_fp_poly",        X_fp_poly,        "lr_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in lr_jobs:
        study = _run_study(study_name, _lr_objective(X, y, folds),
                           n_trials=15)
        hp[hp_key] = _hp_from_lr_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    # ── RF (4 reps) ───────────────────────────────────────────────────────────
    rf_jobs = [
        ("cls_rf_fp_pooled_ru",   X_fp_pooled_RU,   "rf_fp_pooled_ru",   "FP+pooled RU"),
        ("cls_rf_fp_pooled_poly", X_fp_pooled_poly, "rf_fp_pooled_poly", "FP+pooled poly"),
        ("cls_rf_fp_ru",          X_fp_RU,          "rf_fp_ru",          "FP RU"),
        ("cls_rf_fp_poly",        X_fp_poly,        "rf_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in rf_jobs:
        study = _run_study(study_name, _rf_objective(X, y, folds),
                           n_trials=n_trials)
        hp[hp_key] = _hp_from_rf_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    # ── XGB (4 reps) ──────────────────────────────────────────────────────────
    xgb_jobs = [
        ("cls_xgb_fp_pooled_ru",   X_fp_pooled_RU,   "xgb_fp_pooled_ru",   "FP+pooled RU"),
        ("cls_xgb_fp_pooled_poly", X_fp_pooled_poly, "xgb_fp_pooled_poly", "FP+pooled poly"),
        ("cls_xgb_fp_ru",          X_fp_RU,          "xgb_fp_ru",          "FP RU"),
        ("cls_xgb_fp_poly",        X_fp_poly,        "xgb_fp_poly",        "FP poly"),
    ]

    for study_name, X, hp_key, rep in xgb_jobs:
        study = _run_study(study_name, _xgb_objective(X, y, folds),
                           n_trials=n_trials)
        hp[hp_key] = _hp_from_xgb_study(study)
        hp[hp_key]["rep"] = rep
        _save_hp(hp)

    print("\n✓ All 27 classification models tuned.")
    print(f"  DB:  {DB}")
    print(f"  HPs: {HP_PATH}")
    return hp


def load_best_hp(path=HP_PATH):
    with open(path) as f:
        return json.load(f)


def print_study_summary(n_top=3):
    import optuna
    storage = optuna.storages.RDBStorage(DB)
    studies = optuna.get_all_study_summaries(storage=DB)
    for s in sorted(studies, key=lambda x: x.study_name):
        print(f"\n── {s.study_name}  best={s.best_trial.value:.4f}")