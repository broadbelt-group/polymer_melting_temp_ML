# Polymer Melting Temperature Prediction

Machine-learning pipeline for predicting polymer crystallizability (melting temperature classifier) and
melting temperature *T*<sub>m</sub> (regressor) from polymer structure with a coarse-grained **polymer bead sequence graph ([PBSG](https://github.com/mmilrod/PBSG.git))** representation.

This repository accompanies the manuscript *("Stereochemistry-Aware Classification and Regression of Polymer Melting Temperatures Employing Coarse-Grained Representation")* and contains
the main analysis pipeline.

---

## Repository layout

```
.
├── Raw_Data/                  Source datasets (with references)
├── Classifier/                Crystallizability (has_Tm) pipeline
├── Final_Models/              Final models and prediction notebook
├── Regressor/                 Tm regression pipeline
├── pbsg.yml                   Conda environment for all notebooks except 06_shap
├── shap.yml                   Conda environment for SHAP analysis
└── README.md
```

Each pipeline directory contains a numbered notebook sequence, the supporting
`.py` modules they call, cached feature representations, trained models, and the
figure scripts for the manuscript.

---

## Making Predictions

```
.
├── Classifier_final_fulldata/ Final classifer model, DoV, and threshold
├── Regressor_final_fulldata/  Final regressor model, scaler, and DoV
├── dov.py                     Contains DoV class
└── 07_predict.ipynb           Pm-sweep figures, out of domain case-study predictions
```

---

## Environment

All notebooks except `Regressor/06_shap.ipynb` and `Classifier/06_shap.ipynb`:
```bash
conda env create -f pbsg.yml
conda activate pbsg
```

`Regressor/06_shap.ipynb` requires the pinned environment (XGBoost version must match the saved pickles):
```bash
conda env create -f shap.yml
conda activate shap
```
This paper makes use of the **[PBSG](https://github.com/mmilrod/PBSG.git)** representation (which was developed for this project) and is installable at pypi:
```
pip install pbsg
```

---

## Reproduction order

Both pipelines follow the same stage sequence. Run notebooks in numeric order from a fresh kernel.

**Classifier** (`Classifier/`)
| # | Notebook | Produces |
|---|----------|----------|
| 00 | `00_splitting` | `random_stratified_splits_cls.json` |
| 01 | `01_hyperparameter_tuning` | `best_hp_full.json` |
| 02 | `02_cross_validation` | `cv_results*.csv` |
| 03 | `03_final_test` | `test_results*.csv`, `saved_models_all/`, `saved_models_final_fulldata/`|
| 04 | `04_DoV` | `dov_cls.pkl`,  `dov_cls_full.pkl`|
| 05 | `05_analysis_figs` | manuscript and SI figures |
| 06 | `06_shap` | SHAP values, bar chart, summary CSV — **run in `shap`**|


**Regressor** (`Regressor/`)
| # | Notebook | Produces |
|---|----------|----------|
| 00 | `00_splitting` | `random_stratified_splits_reg.json` |
| 01 | `01_hyperparameter_tuning` | `best_hp_reg.json` |
| 02 | `02_cross_validation` | model/representation selection, `cv_results_reg*.csv` |
| 03 | `03_final_test` | `test_results_reg*.csv`, `saved_models_all/`, `saved_models_final_fulldata/`|
| 04 | `04_DoV` | `dov_reg.pkl`, `dov_reg_full.pkl`, `regressor_test_dov_scores.csv` |
| 05 | `05_analysis_figs` | manuscript figures (see provenance table) |
| 06 | `06_shap` | SHAP values, bar chart, summary CSV — **run in `shap`** |

**Splits are canonical artifacts.** The `random_stratified_splits_*.json` files
define the exact train/val/test partitions used for all reported results — load
these directly to reproduce the paper. `run_random_stratified.py` is the generator
(provided for full transparency); regenerating requires the recorded seed and will
otherwise produce a different partition.


**Hyperparameters are fixed.** Selected HPs live in `best_hp_full.json` (classifier)
and `best_hp_reg.json` (regressor). The Optuna search scripts (`optuna_*.py`) are
included for full transparency.

---

## Citation


---
