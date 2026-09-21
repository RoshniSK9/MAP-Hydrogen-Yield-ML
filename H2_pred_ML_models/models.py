# models.py
# Defines all models used in the unified project:
#   Tree-based (mean-imputed or NaN-native):
#     - XGBoost (XGBRegressor) — handles NaN natively
#     - Random Forest          — mean imputation + z-score
#     - HGBR                   — handles NaN natively
#   Linear:
#     - Ridge (L2-regularised) — mean imputation + z-score
#     - PCA + LR               — dimensionality reduction + LR
#   Kernel-based:
#     - SVR                    — dense z-score
#
# Removed models:
#   - LightGBM: not widely used in MAP ML literature; redundant with XGBoost
#   - LR:       near-underdetermined with 160+ features and 164 training samples
#   - SVD+LR:   redundant with PCA+LR; was underfitting (train = test nRMSE)
#   - LightGBM log1p: skewness = 0.37 does not justify log transformation
#
# Note: rare category grouping removed from all models. Ridge and SVR have
# their own regularisation (L2 penalty, C parameter) that handles rare OHE
# columns by shrinking their coefficients, making pre-grouping unnecessary
# and inconsistent with tree-based models which see all categories.

import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from xgboost import XGBRegressor

from .preprocess import (
    build_preprocessor_keep_nan_for_xgb,
    build_preprocessor_for_rf_mean_zscore,
    build_preprocessor_zscore_sparse,
    build_preprocessor_zscore_dense_svr,
)


# ---------------------------------------------------------------------------
# Feature name sanitisation (required by XGBoost)
# ---------------------------------------------------------------------------

def sanitize_feature_names(cols):
    """
    Cleans feature names to be compatible with XGBoost's strict naming rules.
    Removes special characters and ensures uniqueness by appending a counter
    to any duplicate names that arise after cleaning.
    """
    cleaned = []
    for c in cols:
        c = str(c).strip()
        c = re.sub(r"[\n\r\t]", " ", c)            # remove newlines/tabs
        c = re.sub(r'[\\\"\'{}[\]:,]', "_", c)     # replace special chars with underscore
        c = re.sub(r"\s+", " ", c)                 # collapse multiple spaces
        cleaned.append(c)
    # Make names unique by appending __1, __2, etc. for duplicates
    seen = {}
    uniq = []
    for c in cleaned:
        if c not in seen:
            seen[c] = 0
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}__{seen[c]}")
    return uniq



# ---------------------------------------------------------------------------
# Model registry: get_pipeline_models
# Returns all sklearn-compatible models.
# ---------------------------------------------------------------------------

def get_pipeline_models(
    numeric_cols_all,
    cat_cols,
    numeric_base_cols,
    numeric_flag_cols=None,
    random_state=42,
    n_jobs_model=2,
):
    """
    Returns a dict of {model_name: (pipeline, param_dist)} for all models.

    Models included:
      - XGBoost      : tree boosting, handles NaN natively
      - RandomForest : bagged trees, mean imputation + z-score
      - HGBR         : histogram gradient boosting, handles NaN natively
      - Ridge        : L2-regularised linear regression, mean imputation + z-score
      - PCA+LR       : dimensionality reduction + LR, mean imputation + z-score
      - SVR          : support vector regression, dense z-score

    All models are sklearn-compatible and tuned via RandomizedSearchCV.
    Rare category grouping is not applied — each model's own regularisation
    handles rare OHE columns consistently across all model families.
    """
    # Derive flag columns if not provided explicitly
    if numeric_flag_cols is None:
        numeric_flag_cols = [
            c for c in numeric_cols_all
            if c.endswith("__is_NR") or c.endswith("__is_NA")
        ]

    # --- Build preprocessing pipelines ---

    # For XGBoost and HGBR: passthrough numerics (NaN handled natively), OHE cats
    pre_xgb = build_preprocessor_keep_nan_for_xgb(
        numeric_cols_all=numeric_cols_all,
        categorical_cols=cat_cols,
        numeric_base_cols=numeric_base_cols,
    )

    # For RF: mean imputation + z-score
    pre_rf = build_preprocessor_for_rf_mean_zscore(
        numeric_cols_all=numeric_cols_all,
        categorical_cols=cat_cols,
        numeric_base_cols=numeric_base_cols,
    )

    # For linear models (Ridge, PCA+LR): mean impute + z-score + OHE sparse
    pre_linear = build_preprocessor_zscore_sparse(
        numeric_base_cols=numeric_base_cols,
        numeric_flag_cols=numeric_flag_cols,
        categorical_cols=cat_cols,
    )

    # For SVR: same as linear but dense OHE + final global scaler
    pre_svr = build_preprocessor_zscore_dense_svr(
        numeric_base_cols=numeric_base_cols,
        numeric_flag_cols=numeric_flag_cols,
        categorical_cols=cat_cols,
    )

    specs = {}

    # ------------------------------------------------------------------
    # XGBoost — sequential tree boosting with regularisation
    # Handles NaN natively by learning optimal split directions for missing values.
    # Regularisation ranges pushed harder to reduce overfitting on small dataset
    # (164 training samples): max_depth capped at 5, min_child_weight up to 40,
    # stronger reg_lambda and reg_alpha ranges.
    # ------------------------------------------------------------------
    specs["XGBoost"] = (
        Pipeline([
            ("pre",   pre_xgb),
            ("model", XGBRegressor(
                random_state=random_state,
                n_jobs=n_jobs_model,
                tree_method="hist",            # histogram-based for efficiency
                objective="reg:squarederror",  # standard MSE objective
            )),
        ]),
        {
            "model__n_estimators":      [300, 600, 900, 1200, 1800],
            "model__learning_rate":     [0.01, 0.02, 0.05, 0.1],
            "model__max_depth":         [2, 3, 4, 5],              # removed 6, 8 — too deep for 164 samples
            "model__min_child_weight":  [5, 10, 20, 40, 80],       # pushed higher — prevents tiny leaf splits
            "model__subsample":         [0.6, 0.8, 1.0],
            "model__colsample_bytree":  [0.6, 0.8, 1.0],
            "model__reg_alpha":         [0.0, 0.5, 1.0, 5.0, 10.0],   # stronger L1
            "model__reg_lambda":        [1.0, 5.0, 10.0, 30.0, 50.0], # stronger L2
            "model__gamma":             [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        },
    )

    # ------------------------------------------------------------------
    # Random Forest — bagged decision trees with mean imputation
    # Cannot handle NaN natively; uses mean imputation + flag columns.
    # max_depth=None (unlimited) removed — unlimited depth memorises training
    # data on small datasets. min_samples_leaf pushed higher.
    # ------------------------------------------------------------------
    specs["RandomForest"] = (
        Pipeline([
            ("pre",   pre_rf),
            ("model", RandomForestRegressor(
                random_state=random_state,
                n_jobs=n_jobs_model,
            )),
        ]),
        {
            "model__n_estimators":     [500, 800, 1200, 2000],
            "model__max_depth":        [4, 6, 8, 10, 14],         # removed None — unlimited depth overfits
            "model__min_samples_split":[2, 5, 10, 20],
            "model__min_samples_leaf": [2, 4, 8, 16],             # pushed higher — larger leaves = less overfit
            "model__max_features":     ["sqrt", 0.3, 0.5],        # removed 0.8, 1.0 — too many features per split
            "model__bootstrap":        [True],
        },
    )

    # ------------------------------------------------------------------
    # HGBR — Histogram-based Gradient Boosting (sklearn)
    # Handles missing values natively through its histogram-based binning
    # approach, which assigns missing values to a dedicated bin during
    # split finding — similar in principle to XGBoost's native NaN handling.
    # Therefore pre_xgb is used (NaN passthrough) rather than pre_rf
    # (mean imputation), which is more principled than imputing an arbitrary
    # mean value that discards the information that the value was missing.
    # Structural regularisation via max_leaf_nodes, min_samples_leaf,
    # and l2_regularization controls model complexity.
    # ------------------------------------------------------------------
    specs["HGBR"] = (
        Pipeline([
            ("pre",   pre_xgb),   # NaN passthrough — HGBR handles missing natively
            ("model", HistGradientBoostingRegressor(random_state=random_state)),
        ]),
        {
            "model__learning_rate"    : [0.01, 0.02, 0.05],
            "model__max_depth"        : [2, 3, 4],
            "model__max_leaf_nodes"   : [15, 31, 63],
            "model__min_samples_leaf" : [10, 20, 40],
            "model__l2_regularization": [0.1, 1.0, 10.0],
        },
    )

    # ------------------------------------------------------------------
    # Ridge Regression — L2 regularisation
    # Stabilises the ill-conditioned OLS solution when n/p ratio is low.
    # Alpha controls regularisation strength: higher alpha = more shrinkage.
    # ------------------------------------------------------------------
    specs["Ridge"] = (
        Pipeline([
            ("pre",   pre_linear),
            ("model", Ridge()),
        ]),
        {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
        },
    )

    # ------------------------------------------------------------------
    # PCA + LR — dimensionality reduction before linear regression
    # PCA applied after full preprocessing (all 160+ features) to reduce
    # to n_components principal components before fitting LR.
    # This avoids the near-underdetermination problem of plain LR.
    # ------------------------------------------------------------------
    specs["PCA+LR"] = (
        Pipeline([
            ("pre",   pre_linear),
            ("pca",   PCA()),           # reduce to n_components dimensions
            ("model", LinearRegression()),
        ]),
        {"pca__n_components": [5, 10, 20, 30, 50, 80]},
    )

    # ------------------------------------------------------------------
    # SVR — Support Vector Regression with RBF kernel
    # The RBF kernel captures some non-linearity, unlike plain LR.
    # Requires a dense, fully scaled feature matrix (hence pre_svr).
    # ------------------------------------------------------------------
    specs["SVR"] = (
        Pipeline([
            ("pre",   pre_svr),    # dense OHE + global scaler
            ("model", SVR()),
        ]),
        {
            "model__C":       [0.1, 1, 10, 100],     # regularisation strength
            "model__epsilon": [0.01, 0.1, 0.5, 1.0], # epsilon-insensitive tube width
            "model__gamma":   ["scale", "auto"],      # RBF kernel bandwidth
            "model__kernel":  ["rbf"],
        },
    )

    return specs