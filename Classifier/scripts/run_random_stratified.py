
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from rdkit import Chem

# ── Column names ───────────────────────────────────────────────────────────────
COLS = dict(
    smiles       = "repeat_units",
    stereo       = "stereo_class",
    copol_type   = "copolymer_type",
    label        = "has_Tm",
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_FOLDS       = 5
TEST_FRACTION = 0.15
MIN_STRATUM   = N_FOLDS + 1   # minimum n to use a stratum without fallback
OUTPUT_FILE   = "random_stratified_splits_cls.json"

# ── SMARTS polymer class definitions ──────────────────────────────────────────
# Priority order: first match wins per repeat unit.
# For copolymers: majority vote; tie → "mixed".
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


# ── SMARTS helpers ─────────────────────────────────────────────────────────────

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


# ── Stratification label with progressive fallback ────────────────────────────

def _build_strat_labels(df: pd.DataFrame) -> np.ndarray:
    poly_cls = assign_polymer_classes(df).values
    stereo   = df[COLS["stereo"]].fillna("unknown").values
    label    = df[COLS["label"]].astype(str).values

    l0 = np.array([f"{c}|{s}|{l}" for c, s, l in zip(poly_cls, stereo, label)],
                  dtype=object)
    l2 = np.array(label, dtype=object)   # "True" / "False" — always n >> 2

    counts0 = Counter(l0)

    strat = np.array(
        [l0[i] if counts0[l0[i]] >= 2 else l2[i] for i in range(len(l0))],
        dtype=object
    )

    final_counts = Counter(strat)
    bad = {k: v for k, v in final_counts.items() if v < 2}
    assert not bad, f"Singletons remain: {bad}"

    return strat


# ── Coverage diagnostics ───────────────────────────────────────────────────────

def _print_coverage(df: pd.DataFrame, idx: np.ndarray, name: str):
    sub = df.iloc[idx]
    n   = len(sub)
    has_tm_pct = 100 * sub[COLS["label"]].mean()
    stereo_dist = sub[COLS["stereo"]].value_counts(normalize=True).to_dict()
    print(f"  {name:20s}  n={n:4d}  has_Tm={has_tm_pct:5.1f}%  "
          f"stereo={ {k: f'{100*v:.0f}%' for k, v in stereo_dist.items()} }")


# ── Main split function ────────────────────────────────────────────────────────

def run_random_stratified(
    df: pd.DataFrame,
    output_file: str  = OUTPUT_FILE,
    seed: int         = 42,
    verbose: bool     = True,
) -> dict:
    n = len(df)
    idx_all = np.arange(n)

    # Build stratification labels
    strat = _build_strat_labels(df)

    if verbose:
        counts = Counter(strat)
        n_singletons = sum(1 for v in counts.values() if v == 1)
        print(f"Stratification summary:")
        print(f"  Unique strata:    {len(counts)}")
        print(f"  Singletons (n=1): {n_singletons}")
        print(f"  Min stratum size: {min(counts.values())}")

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

        # Polymer class coverage in test vs train
        poly_cls = assign_polymer_classes(df)
        test_classes  = set(poly_cls.iloc[test_idx].unique())
        train_classes = set(poly_cls.iloc[trainval_idx].unique())
        missing = train_classes - test_classes
        if missing:
            warnings.warn(
                f"Polymer classes in train but not test: {missing}. "
                f"Consider running seed_search_random_stratified()."
            )
        else:
            print(f"  All polymer classes represented in test. ✓")

        # Stereo class coverage
        test_stereo  = set(df[COLS["stereo"]].iloc[test_idx].dropna().unique())
        train_stereo = set(df[COLS["stereo"]].iloc[trainval_idx].dropna().unique())
        missing_s = train_stereo - test_stereo
        if missing_s:
            warnings.warn(
                f"Stereo classes in train but not test: {missing_s}. "
                f"Consider running seed_search_random_stratified()."
            )
        else:
            print(f"  All stereo classes represented in test. ✓")

    output = {"test": test_idx.tolist(), "folds": folds}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {output_file}")
    return output


# ── Seed search ────────────────────────────────────────────────────────────────

def seed_search_random_stratified(
    df: pd.DataFrame,
    n_seeds: int    = 50,
    output_file: str = OUTPUT_FILE,
    verbose: bool   = True,
) -> int:
    strat     = _build_strat_labels(df)
    poly_cls  = assign_polymer_classes(df).values
    stereo    = df[COLS["stereo"]].fillna("unknown").values
    label     = df[COLS["label"]].astype(bool).values
    global_tm = label.mean()
    n         = len(df)
    idx_all   = np.arange(n)

    results = []
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_FRACTION, random_state=seed
        )
        tv_idx, t_idx = next(sss.split(idx_all, strat))

        missing_cls    = len(set(poly_cls[tv_idx]) - set(poly_cls[t_idx]))
        missing_stereo = len(set(stereo[tv_idx])   - set(stereo[t_idx]))
        tm_dev         = abs(label[t_idx].mean() - global_tm)
        n_test         = len(t_idx)

        results.append({
            "seed":           seed,
            "n_test":         n_test,
            "missing_cls":    missing_cls,
            "missing_stereo": missing_stereo,
            "tm_dev":         tm_dev,
        })

    results_df = pd.DataFrame(results)
    # Prefer zero missing; within those, minimise label deviation
    clean = results_df[
        (results_df["missing_cls"] == 0) &
        (results_df["missing_stereo"] == 0)
    ]
    if len(clean) == 0:
        warnings.warn("No seed found with full class coverage. Using best available.")
        clean = results_df

    best_row  = clean.sort_values("tm_dev").iloc[0]
    best_seed = int(best_row["seed"])

    if verbose:
        print(f"Seed search results (top 10 clean seeds by label deviation):")
        print(clean.sort_values("tm_dev").head(10).to_string(index=False))
        print(f"\nBest seed: {best_seed}  "
              f"(n_test={int(best_row['n_test'])}, "
              f"tm_dev={best_row['tm_dev']:.4f})")

    # Build and save the split with best seed
    run_random_stratified(df, output_file=output_file, seed=best_seed,
                          verbose=verbose)
    return best_seed


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Import and call with your dataframe:")
    print()
    print("  from run_random_stratified import run_random_stratified")
    print("  output = run_random_stratified(df_cls)")
    print()
    print("Or run seed search first:")
    print("  from run_random_stratified import seed_search_random_stratified")
    print("  best_seed = seed_search_random_stratified(df_cls, n_seeds=50)")