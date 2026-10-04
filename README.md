# Polymer Melting Temperature Prediction

Machine-learning pipeline for predicting polymer melting temperature through classification and regression from polymer structure with a coarse-grained **polymer bead sequence graph ([PBSG](https://github.com/mmilrod/PBSG.git))** representation.

This repository accompanies the manuscript *("Stereochemistry-Aware Classification and Regression of Polymer Melting Temperatures")* and contains
the main analysis pipeline.

---

## Repository layout

```
.
├── Raw_Data/                  Source datasets (with references)
├── Classifier/                has_Tm pipeline
├── Final_Models/              Final models and prediction notebook
├── Regressor/                 Tm regression pipeline
├── Supplemental_Analysis/     Code for sections S3-S5
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

Note that `sys.path.append(str(Path(PATH TO PBSG)))` needs to be replaced with the path to your copy of PBSG in all relevant files. This can be done easily in VSCode by using the Find+Replace function. This will not be necessary once PBSG is made available on pypi. 
At the time of publication, it is not yet available on pypi.

**Splits are canonical artifacts.** The `random_stratified_splits_*.json` files
define the exact train/val/test partitions used for all reported results — load
these directly to reproduce the paper. `run_random_stratified.py` is the generator
(provided for full transparency); regenerating requires the recorded seed and will
otherwise produce a different partition.


**Hyperparameters are fixed.** Selected HPs live in `best_hp_full.json` (classifier)
and `best_hp_reg.json` (regressor). The Optuna search scripts (`optuna_*.py`) are
included for full transparency.

**Supplemental Analysis.**
All code necessary to reproduce supplemental analysis.
Section S3: Results of an extrapolation stress-test via Butina splitting
Section S4: Evaluating the impact of stereochemical features on model performance
Section S5:Quantifying prediction uncertainty arising from stochastic polymer stereochemical sequence generation

---

## Citation
If you use or reference these models or datasets please reference:

> **Stereochemistry-Aware Classification and Regression of Polymer Melting Temperatures.**
> Maya L. Milrod, Kevin M. Shebek, A. Nolan Wilson, Keith E.-J. Tyo, Eugene Y.-X. Chen, Tobin J. Marks, and Linda J. Broadbelt. *Cell Reports Physical Science*, **2026**.

A DOI will be added upon publication.

---
