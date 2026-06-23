"""
run_random_stratified_reg.py
============================
Random stratified train/val/test split for polymer Tm regression.

Differences from classification version:
  - Dataset is already filtered to has_Tm=True (no label axis)
  - Stratification uses Tm quartile bins instead of has_Tm
    so each fold sees the full range of Tm values
  - Output: random_stratified_splits_reg.json

Stratification label: polymer_class | stereo_class | Tm_bin
  - polymer_class: SMARTS-derived (same 8 classes as classification)
  - stereo_class:  from dataframe column
  - Tm_bin:        quartile bin (Q1/Q2/Q3/Q4) of Tm in Celsius

Progressive fallback:
  Level 0: class|stereo|Tm_bin   (finest)
  Level 1: class|Tm_bin          (if stratum n < 2)
  Level 2: Tm_bin only           (last resort — always n >> 2)

Split structure
---------------
  15% held-out test  →  StratifiedShuffleSplit
  85% train+val      →  StratifiedKFold 5-fold

Output: JSON with same schema as classification split
  {"test": [...], "folds": [{"train": [...], "val": [...]}, ...]}
"""

import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from rdkit import Chem

# ── Column names ───────────────────────────────────────────────────────────────
COLS = dict(
    smiles = "repeat_units",
    stereo = "stereo_class",
    tm     = "Tm",
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_FOLDS       = 5
TEST_FRACTION = 0.15
OUTPUT_FILE   = "random_stratified_splits_reg.json"

# ── SMARTS polymer class definitions (same as classification) ──────────────────
_POLYMER_CLASS_SMARTS = [
    ("polyamide",     ["[NX3H][CX3](=O)",
                       "[NX3]([CX3](=O))[CX3](=O)"]),
    ("polyester",     ["[OX2][CX3](=O)[!N]",
                       "[CX3](=O)[OX2][CX3](=O)"]),
    ("polycarbonate", ["[OX2][CX3](=O)[OX2]"]),
    ("polyurethane",  ["[NX3H][CX3](=O)[OX2]",
                       "[OX2][CX3](=O)[NX3H]"]),
    ("polyether",     ["[CX4][OX2][CX4]",
                       "[cX3][OX2][CX4]",
                       "[CX4][OX2][cX3]"]),
    ("polyolefin",    ["[CX4H2][CX4H2]",
                       "[CX4H2][CX4H]([*])",
                       "[CX3H]=[CX3H2]",
                       "[CX3H]=[CX3H]"]),
    ("polysiloxane",  ["[Si][OX2][Si]"]),
]
_FALLBACK_CLASS = "other"


# ── SMARTS helpers (identical to classification version) ───────────────────────

def _compile_patterns():
    compiled = []
    for cls, smarts_list in _POLYMER_CLASS_SMARTS:
        pats = [Chem.MolFromSmarts(s) for s in smarts_list]
        compiled.append((cls, pats))
    return compiled


def _classify_one(smiles: str, patterns) -> str:
    units = [s.strip() for s in smiles.split(".") if s.strip()]
    unit_classes = []
    for unit in units:
        mol = Chem.MolFromSmiles(unit)
        if mol is None:
            unit_classes.append(_FALLBACK_CLASS)
            continue
        assigned = _FALLBACK_CLASS
        for cls, pats in patterns:
            if any(mol.HasSubstructMatch(p) for p in pats if p is not None):
                assigned = cls
                break
        unit_classes.append(assigned)
    if not unit_classes:
        return _FALLBACK_CLASS
    counts = Counter(unit_classes)
    top_n = counts.most_common(1)[0][1]
    top = [c for c, n in counts.items() if n == top_n]
    return top[0] if len(top) == 1 else "mixed"


def assign_polymer_classes(df: pd.DataFrame) -> pd.Series:
    patterns = _compile_patterns()
    return df[COLS["smiles"]].fillna("").apply(
        lambda s: _classify_one(s, patterns)
    )


# ── Tm quartile binning ────────────────────────────────────────────────────────

def _tm_bins(df: pd.DataFrame) -> np.ndarray:
    tm = df[COLS["tm"]].values
    try:
        bins = pd.qcut(tm, q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                       duplicates="drop")
    except ValueError:
        bins = pd.qcut(tm, q=3, labels=["Q1", "Q2", "Q3"],
                       duplicates="drop")
    return np.array(bins, dtype=str)


# ── Stratification label with progressive fallback ────────────────────────────

def _build_strat_labels(df: pd.DataFrame) -> np.ndarray:
    poly_cls = assign_polymer_classes(df).values
    stereo   = df[COLS["stereo"]].fillna("unknown").values
    tm_bin   = _tm_bins(df)

    l0 = np.array([f"{c}|{s}|{b}" for c, s, b in
                   zip(poly_cls, stereo, tm_bin)], dtype=object)
    l1 = np.array([f"{c}|{b}" for c, b in
                   zip(poly_cls, tm_bin)], dtype=object)
    l2 = np.array(tm_bin, dtype=object)

    counts0 = Counter(l0)
    counts1 = Counter(l1)

    # Initial assignment
    strat = np.where(
        np.array([counts0[x] >= 2 for x in l0]),
        l0,
        np.where(
            np.array([counts1[x] >= 2 for x in l1]),
            l1,
            l2,
        )
    ).astype(object)

    # Iterative collapse using fresh counts each iteration
    for _ in range(20):
        counts = Counter(strat)
        bad_idx = [i for i, s in enumerate(strat) if counts[s] < 2]
        if not bad_idx:
            break
        # Recount l2 after current assignments to avoid creating new singletons
        strat_copy = strat.copy()
        for i in bad_idx:
            strat_copy[i] = l2[i]
        # Check if collapse created new singletons in l2
        new_counts = Counter(strat_copy)
        for i in bad_idx:
            if new_counts[strat_copy[i]] < 2:
                # l2 itself is a singleton — merge with nearest bin
                # by just using a fixed safe label
                strat_copy[i] = "Q2"  # guaranteed large bin
        strat = strat_copy

    final_counts = Counter(strat)
    bad = {k: v for k, v in final_counts.items() if v < 2}
    assert not bad, f"Singletons remain: {bad}"

    return strat

# ── Coverage diagnostics ───────────────────────────────────────────────────────

def _print_coverage(df: pd.DataFrame, idx: np.ndarray, name: str):
    sub = df.iloc[idx]
    tm  = sub[COLS["tm"]]
    stereo_dist = sub[COLS["stereo"]].value_counts(normalize=True).to_dict()
    print(f"  {name:20s}  n={len(sub):4d}  "
          f"Tm={tm.mean():.1f}±{tm.std():.1f}°C  "
          f"[{tm.min():.0f}–{tm.max():.0f}]  "
          f"stereo={ {k: f'{100*v:.0f}%' for k, v in stereo_dist.items()} }")


# ── Main split function ────────────────────────────────────────────────────────

def run_random_stratified_reg(
    df: pd.DataFrame,
    output_file: str = OUTPUT_FILE,
    seed: int        = 0,
    verbose: bool    = True,
) -> dict:
    """
    Build a random stratified train/val/test split for Tm regression.

    Parameters
    ----------
    df          : regression dataframe (has_Tm=True subset)
    output_file : path to save JSON
    seed        : random seed
    verbose     : print diagnostics

    Returns
    -------
    dict with keys "test" and "folds"
    """
    n       = len(df)
    idx_all = np.arange(n)
    strat   = _build_strat_labels(df)

    if verbose:
        counts = Counter(strat)
        print(f"Stratification summary (regression):")
        print(f"  Dataset size:     {n}")
        print(f"  Unique strata:    {len(counts)}")
        print(f"  Min stratum size: {min(counts.values())}")
        print(f"  Tm range:         "
              f"{df[COLS['tm']].min():.1f}–"
              f"{df[COLS['tm']].max():.1f}°C")
        print(f"  Tm quartile bins:")
        for b in ["Q1", "Q2", "Q3", "Q4"]:
            sub = df[df[COLS["tm"]].isin(
                df[COLS["tm"]][np.array([b in s for s in strat])]
            )]
            bin_tm = df.iloc[[i for i, s in enumerate(strat) if b in s]][COLS["tm"]]
            if len(bin_tm) > 0:
                print(f"    {b}: n={len(bin_tm):3d}  "
                      f"{bin_tm.min():.1f}–{bin_tm.max():.1f}°C")

    # ── Stage 1: carve out test set ───────────────────────────────────────────
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=seed
    )
    trainval_idx, test_idx = next(splitter_test.split(idx_all, strat))

    # ── Stage 2: 5-fold CV on trainval ───────────────────────────────────────
    strat_tv = strat[trainval_idx]
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    folds = []
    for train_rel, val_rel in kf.split(trainval_idx, strat_tv):
        folds.append({
            "train": trainval_idx[train_rel].tolist(),
            "val":   trainval_idx[val_rel].tolist(),
        })

    # ── Diagnostics ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\nSplit summary (seed={seed}):")
        _print_coverage(df, trainval_idx, "Train+Val")
        _print_coverage(df, test_idx,     "Test")

        print("\nCV fold balance:")
        for i, fold in enumerate(folds):
            _print_coverage(df, np.array(fold["train"]), f"Fold {i+1} train")
            _print_coverage(df, np.array(fold["val"]),   f"Fold {i+1} val  ")
            print()

        # Polymer class coverage
        poly_cls      = assign_polymer_classes(df)
        test_classes  = set(poly_cls.iloc[test_idx].unique())
        train_classes = set(poly_cls.iloc[trainval_idx].unique())
        missing       = train_classes - test_classes
        if missing:
            warnings.warn(
                f"Polymer classes in train but not test: {missing}. "
                f"Consider running seed_search_random_stratified_reg()."
            )
        else:
            print(f"  All polymer classes represented in test. ✓")

        # Stereo class coverage
        test_stereo  = set(df[COLS["stereo"]].iloc[test_idx].dropna().unique())
        train_stereo = set(df[COLS["stereo"]].iloc[trainval_idx].dropna().unique())
        missing_s    = train_stereo - test_stereo
        if missing_s:
            warnings.warn(
                f"Stereo classes in train but not test: {missing_s}. "
                f"Consider running seed_search_random_stratified_reg()."
            )
        else:
            print(f"  All stereo classes represented in test. ✓")

    output = {"test": test_idx.tolist(), "folds": folds}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {output_file}")
    return output


# ── Seed search ────────────────────────────────────────────────────────────────

def seed_search_random_stratified_reg(
    df: pd.DataFrame,
    n_seeds: int     = 50,
    output_file: str = OUTPUT_FILE,
    verbose: bool    = True,
) -> int:
    """
    Search seeds 0..n_seeds-1 and return the best one.

    Scoring: penalise seeds where any polymer class or stereo class
    present in train is missing from test. Among clean seeds, prefer
    the one with Tm mean and std in test closest to global.
    """
    strat     = _build_strat_labels(df)
    poly_cls  = assign_polymer_classes(df).values
    stereo    = df[COLS["stereo"]].fillna("unknown").values
    tm        = df[COLS["tm"]].values
    global_mean = tm.mean()
    global_std  = tm.std()
    n           = len(df)
    idx_all     = np.arange(n)

    results = []
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_FRACTION, random_state=seed
        )
        tv_idx, t_idx = next(sss.split(idx_all, strat))

        missing_cls    = len(set(poly_cls[tv_idx]) - set(poly_cls[t_idx]))
        missing_stereo = len(set(stereo[tv_idx])   - set(stereo[t_idx]))
        # Deviation of test Tm distribution from global
        tm_mean_dev    = abs(tm[t_idx].mean() - global_mean)
        tm_std_dev     = abs(tm[t_idx].std()  - global_std)
        tm_dev         = tm_mean_dev + tm_std_dev

        results.append({
            "seed":           seed,
            "n_test":         len(t_idx),
            "missing_cls":    missing_cls,
            "missing_stereo": missing_stereo,
            "tm_mean_dev":    round(tm_mean_dev, 3),
            "tm_std_dev":     round(tm_std_dev, 3),
            "tm_dev":         round(tm_dev, 3),
        })

    results_df = pd.DataFrame(results)
    clean = results_df[
        (results_df["missing_cls"] == 0) &
        (results_df["missing_stereo"] == 0)
    ]
    if len(clean) == 0:
        warnings.warn("No seed with full class coverage. Using best available.")
        clean = results_df

    best_row  = clean.sort_values("tm_dev").iloc[0]
    best_seed = int(best_row["seed"])

    if verbose:
        print(f"Seed search results (top 10 clean seeds by Tm deviation):")
        print(clean.sort_values("tm_dev").head(10).to_string(index=False))
        print(f"\nBest seed: {best_seed}  "
              f"(n_test={int(best_row['n_test'])}, "
              f"tm_dev={best_row['tm_dev']:.3f})")

    run_random_stratified_reg(df, output_file=output_file,
                               seed=best_seed, verbose=verbose)
    return best_seed


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Import and call with your regression dataframe:")
    print()
    print("  from run_random_stratified_reg import (")
    print("      seed_search_random_stratified_reg,")
    print("      run_random_stratified_reg,")
    print("  )")
    print()
    print("  # Run seed search first (recommended):")
    print("  best_seed = seed_search_random_stratified_reg(df_reg, n_seeds=50)")
    print()
    print("  # Or run directly with a specific seed:")
    print("  output = run_random_stratified_reg(df_reg, seed=0)")