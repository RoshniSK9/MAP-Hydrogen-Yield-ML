# Machine Learning Framework for Hydrogen Yield Prediction from Microwave-Assisted Pyrolysis

This repository contains the code and dataset associated with the following paper:

> [Citation to be added upon acceptance]

---

## Overview

This study develops a unified machine learning framework for predicting hydrogen yield from microwave-assisted pyrolysis (MAP) of diverse waste feedstocks, including biomass, plastic waste, and mixed municipal solid waste. A dataset of 241 experimental datapoints compiled from 16 peer-reviewed studies was assembled, incorporating 25 input features spanning feedstock properties, microwave operating conditions, microwave absorber dielectric properties, and catalyst properties.

Six machine learning models were evaluated and compared:
- **XGBoost** (best performing, test R² = 0.69)
- Random Forest (RF)
- Histogram-based Gradient Boosting Regression (HGBR)
- Ridge Regression
- Principal Component Analysis + Linear Regression (PCA+LR)
- Support Vector Regression (SVR)

SHAP analysis was applied to the best performing model (XGBoost) to identify the key input features governing hydrogen yield and provide data-driven guidance for experimental design and catalyst selection.

---

## Repository Structure

```
map-hydrogen-yield-ml/
│
├── H2_pred_ML_models/           # Python package containing all source code
│   ├── __init__.py              # Makes the folder a Python package
│   ├── run.py                   # Main script — trains all models, saves results
│   ├── data.py                  # Data loading and cleaning
│   ├── preprocess.py            # Preprocessing pipelines for all model families
│   ├── models.py                # Model definitions and hyperparameter search spaces
│   ├── metrics.py               # RMSE, R² and bootstrap confidence intervals
│   ├── plots.py                 # Parity plots (IJHE publication style)
│   └── shap_analysis.py        # SHAP analysis — beeswarm, bar, waterfall, dependence plots
│
├── Dataset_Modelling.xlsx       # Compiled experimental dataset (241 datapoints, 25 input features and 1 output variable)
├── notebook.ipynb               # Jupyter notebook to run the full pipeline
├── requirements.txt             # Essential Python dependencies
├── environment.yml              # Full conda environment for exact reproducibility
└── README.md                    # This file
```

---

## Dataset

The dataset (`Dataset_Modelling.xlsx`) contains 241 experimental datapoints compiled from 16 peer-reviewed studies on microwave-assisted pyrolysis. Input features are organised into four categories:

| Category | Features |
|---|---|
| Feedstock properties | Particle size, carbon, hydrogen, nitrogen, oxygen, sulfur, moisture, volatile matter, fixed carbon, ash content |
| Microwave operating parameters | Pyrolysis temperature, heating rate, microwave power, isothermal time |
| Microwave absorber properties | Absorber identity, absorber particle size, dielectric constant (ε′), dielectric loss tangent (tan δ), feedstock to absorber ratio |
| Catalyst properties | Catalyst identity, surface area, pore diameter, metal loading, feedstock to catalyst ratio, catalyst particle size |

**Target variable:** Hydrogen yield (mmol H₂/g feedstock), range 0–71 mmol H₂/g feedstock.

**Missing values:** Two types of missing values are encoded in the dataset:
- `MISSING_NA` — feature not applicable to the experiment (e.g. catalyst properties when no catalyst was used)
- `MISSING_NR` — feature exists but was not reported in the original study

---

## Getting Started

If you are new to Python and conda, follow these steps before proceeding to the Installation section.

**1. Install conda:**
Download and install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). Miniconda is a lightweight installer for conda, a package and environment manager for Python.

**2. Install Git:**
Download and install Git from [https://git-scm.com/downloads](https://git-scm.com/downloads). Git is used to download the repository to your computer.

**3. Open a terminal:**
- On **macOS**: press Command + Space, type Terminal, and press Enter
- On **Windows**: search for Anaconda Prompt in the Start menu and open it
- On **Linux**: open your system terminal

**4. Verify your installation:**
Run the following commands to confirm conda and Git are installed correctly:
```bash
conda --version
git --version
```
You should see version numbers printed for both. If not, revisit steps 1 and 2.

Once conda and Git are installed and working, proceed to the Installation section below.

---

## Installation
**Requirements:** Python 3.12.11

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/RoshniSK9/MAP-Hydrogen-Yield-ML.git
cd MAP-Hydrogen-Yield-ML
```

**Step 2 — Set up the environment:**

Option A — Recommended for macOS (Apple Silicon) users. Recreates the exact conda environment used in this study including Jupyter:
```bash
conda env create -f environment.yml
conda activate h2_pred
```

Option B — Recommended for Windows and Linux users, or macOS users who prefer a minimal install. Installs only the essential packages:
```bash
conda create -n h2_pred python=3.12.11
conda activate h2_pred
pip install -r requirements.txt
pip install jupyterlab
```

**Step 3 — Launch Jupyter and open the notebook:**
```bash
jupyter lab
```
Open `notebook.ipynb` and run all cells to reproduce the results.

---

## Usage

Open `notebook.ipynb` in Jupyter and run all cells. The notebook imports the `H2_pred_ML_models` package and runs the full pipeline:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from H2_pred_ML_models.run import run_all

results = run_all(
    file_path="Dataset_Modelling.xlsx",
    target_col="H2 yield (mmol/g of feedstock)",
    categorical_cols=["Catalyst (-)", "Microwave absorber (-)"],
    test_size=0.20,
    random_state=30,
    k=10,
    n_iter_search=50,
    n_jobs_model=2,
    n_jobs_search=1,
    zoom_max=80,
    out_root="results_H2_pred_ML_models",
)
```

This will:
1. Load and preprocess the dataset
2. Split into 80% training and 20% test sets (random seed = 30)
3. Train all six models with 10-fold cross-validation and randomised hyperparameter search (50 iterations)
4. Evaluate performance on the test set with bootstrap confidence intervals (2000 resamples)
5. Generate parity plots for all models
6. Run SHAP analysis on XGBoost
7. Save all results to a timestamped output directory (`results_H2_pred_ML_models/run_<timestamp>/`)

---

## Output Files

Running the pipeline produces a timestamped output directory containing:

| File | Description |
|---|---|
| `summary_results_run_*.csv` | Test R² and 95% bootstrap CI for all models |
| `train_set.csv` | Training set (164 datapoints) |
| `test_set.csv` | Test set (41 datapoints) |
| `*_combined_full.png` | Parity plots for all models |
| `shap_beeswarm_*.png` | SHAP beeswarm plot (XGBoost) |
| `shap_bar_*.png` | SHAP bar plot (XGBoost) |
| `shap_dependence_*.png` | SHAP dependence plots for top 10 features |
| `shap_meanabs_*.csv` | Mean absolute SHAP values for all features |

---

## Key Results

| Model | Test R² | 95% CI |
|---|---|---|
| XGBoost | 0.69 | [0.47, 0.82] |
| Random Forest | 0.67 | [0.41, 0.81] |
| HGBR | 0.64 | [0.39, 0.78] |
| Ridge Regression | 0.56 | [0.34, 0.70] |
| PCA+LR | 0.56 | [0.32, 0.72] |
| SVR | 0.49 | [0.20, 0.68] |

**Top 5 SHAP features (XGBoost, test set):**

| Rank | Feature | Mean Absolute SHAP |
|---|---|---|
| 1 | Hydrogen content (wt%) | 5.6 |
| 2 | Feedstock to catalyst ratio (wt/wt) | 5.1 |
| 3 | Metal loading in catalyst (wt%) | 2.8 |
| 4 | Pyrolysis temperature (°C) | 1.8 |
| 5 | Microwave power (W) | 1.4 |

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.12.11 |
| scikit-learn | 1.7.1 |
| XGBoost | 3.0.1 |
| SHAP | 0.47.2 |
| NumPy | 2.0.0 |
| pandas | 2.3.1 |
| matplotlib | 3.10.0 |
| scipy | 1.16.0 |
| openpyxl | 3.1.5 |

---

## Reproducibility
The train-test split uses a fixed random seed of 30. All hyperparameter searches use reproducible random states. This code was developed and tested on macOS 15 (Apple M5, MacBook Air) using Python 3.12.11. The package versions listed in `requirements.txt` and `environment.yml` are compatible with Apple Silicon. Users on Windows or Linux should use `requirements.txt` with Option B to install dependencies, as `environment.yml` was exported from macOS and may not be compatible with other operating systems. To exactly reproduce the results reported in the paper, ensure the package versions listed in `requirements.txt` are installed.

---

## Citation

If you use this code or dataset in your research, please cite:

```
[Citation to be added upon paper acceptance]
```

---

## License

This repository is licensed under the MIT License. The dataset is licensed under CC BY 4.0.

---

## Contact

Roshni Sajiv Kumar (roshni.sajivkumar@ucalgary.ca, University of Calgary)
