"""
train_final_full_data_cls.py
============================
Train the FULL-DATA deployment CLASSIFIER (best model: XGB FP+pooled poly) on ALL
classifier rows, for predicting EXTERNAL polymers (Table 2 case studies).

HARD RULE: external predictions ONLY. Reported metrics (Table 1) come from the
held-out lineage (saved_models_top/). Writes to saved_models_final_fulldata/.

Threshold: carries over the REPORTED threshold (validated on held-out test),
rather than recomputing — no held-out set exists in full-data mode.

USAGE (from the Classifier directory, in the `pbsg` env):
    python train_final_full_data_cls.py     OR     %run in a Classifier/ notebook
"""
import os, json, numpy as np, joblib

# ── classifier infrastructure — CONFIRM these names match your Classifier/scripts/CV.py ──
# The classifier analog of the regressor builders. If your function is named
# differently, change this import line only.
from scripts.CV import _build_classical_tuned, hp_to_args_cls, load_tuned_hp_cls   # <-- confirm names

HP_FILE        = "best_hp_full.json"
OUT_DIR        = "saved_models_final_fulldata"
BEST_CLS_KEY   = "xgb_fp_pooled_poly"          # best classifier (AUPRC 0.976)
THRESH_KEY     = "xgb_fp_pooled_poly"                      # key in cls_thresholds.json
REPORTED_THRESH_JSON = "saved_models_all/cls_thresholds-full.json"   # reported thresholds
os.makedirs(OUT_DIR, exist_ok=True)


def load_full_data_cls():
    # ADJUST paths to your full classifier feature array + label
    X = np.load("Vectors/X_fp_pooled_poly.npy")     # <-- poly features (classifier)
    y = np.load("Vectors/y.npy").astype(int)    # <-- has_Tm label (0/1)

    assert y.ndim == 1, "y must be 1-D"
    uniq = set(np.unique(y).tolist())
    assert uniq <= {0, 1} and len(uniq) == 2, (
        f"y must be binary has_Tm (0/1); got unique={uniq}")
    assert len(y) == X.shape[0], "row mismatch X vs y"
    print(f"Full classifier data: N={len(y)}  pos={int(y.sum())} neg={int((1-y).sum())}")
    return X, y


def main():
    from sklearn.preprocessing import StandardScaler

    hp_all = load_tuned_hp_cls(HP_FILE)
    X, y = load_full_data_cls()

    hp = hp_to_args_cls(hp_all, BEST_CLS_KEY)          # raw dict for classical
    model = _build_classical_tuned("xgb", hp)      # XGBClassifier with tuned HPs

    # XGB classifier in your pipeline uses UNSCALED poly features (per predict.py).
    # If your classifier pipeline scales them, add a StandardScaler here to match.
    model.fit(X, y)

    joblib.dump(model, f"{OUT_DIR}/cls_xgb_pooled_poly_FULL.pkl")
    print(f"  saved -> {OUT_DIR}/cls_xgb_pooled_poly_FULL.pkl")

    # carry over the REPORTED threshold (validated on held-out test)
    with open(REPORTED_THRESH_JSON) as f:
        reported = json.load(f)
    thr = reported.get(THRESH_KEY, 0.5)
    with open(f"{OUT_DIR}/cls_thresholds_FULL.json", "w") as f:
        json.dump({THRESH_KEY: thr}, f, indent=2)
    print(f"  carried over threshold {THRESH_KEY}={thr} (from reported held-out run)")

    with open(f"{OUT_DIR}/README_cls.txt", "w") as f:
        f.write(
            "FULL-DATA DEPLOYMENT CLASSIFIER (xgb_fp_pooled_poly) - all rows.\n"
            "EXTERNAL predictions only. NOT for Table 1 metrics.\n"
            f"Threshold carried over from {REPORTED_THRESH_JSON} (held-out-validated).\n")
    print(f"\nDone -> {OUT_DIR}/  (EXTERNAL predictions only)")


if __name__ == "__main__":
    main()
