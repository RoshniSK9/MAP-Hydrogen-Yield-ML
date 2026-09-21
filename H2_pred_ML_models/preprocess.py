# preprocess.py
# Contains all preprocessing pipelines used across the project.
#
# Three pipeline families:
#   1. keep_nan_for_xgb  — passthrough numerics (XGBoost/HGBR handle NaN natively)
#   2. mean_zscore       — mean imputation + z-score scaling (RF)
#   3. zscore_sparse     — mean imputation + z-score + OHE sparse (Ridge, PCA+LR)
#   4. zscore_dense_svr  — same as sparse but dense OHE, no final global scaler (SVR)
#
# Changes from original:
#   1. Dual flags (__is_NR, __is_NA) replaced with single __exists flag per feature
#      __exists = 1: feature exists in experiment (present or MISSING_NR)
#      __exists = 0: feature not applicable (MISSING_NA)
#   2. SVR preprocessor: removed final global StandardScaler after OHE.
#      StandardScaler applied only to numeric base columns inside ColumnTransformer.
#   3. Flag creation moved inside pipeline (AddExistsFlags transformer) so flags
#      are created after train-test split, consistent with OHE treatment.
#
# Rare category grouping is NOT applied — each model's own regularisation
# handles rare OHE columns consistently across all model families.

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# OHE helpers — handle sparse_output vs sparse API change across sklearn versions
# ---------------------------------------------------------------------------

def _ohe_sparse():
    """One-hot encoder that returns a sparse matrix."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _ohe_dense():
    """One-hot encoder that returns a dense matrix (required by SVR)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------------------------------------------------------------------------
# Custom transformer: AddExistsFlags (replaces dual __is_NR / __is_NA flags)
# ---------------------------------------------------------------------------

class AddExistsFlags(BaseEstimator, TransformerMixin):
    """
    Creates a single __exists binary flag per numeric feature.

    __exists = 1: value present OR MISSING_NR (feature exists, mean imputed if NR)
    __exists = 0: MISSING_NA (feature not applicable, value set to 0.0)

    Replaces the original dual __is_NR / __is_NA flag approach.
    Stateless: fit() just returns self.
    """

    def __init__(self, numeric_base_cols):
        self.numeric_base_cols = numeric_base_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import numpy as np
        import warnings
        X = X.copy()
        for c in self.numeric_base_cols:
            if c not in X.columns:
                continue
            col_str = X[c].astype(str)
            is_na   = col_str.eq("MISSING_NA")
            is_nr   = col_str.eq("MISSING_NR")
            # __exists = 0 for NA, 1 for NR and actual values
            X[f"{c}__exists"] = (~is_na).astype(float)
            # Set NA to 0.0, NR to NaN (for downstream mean imputation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                X[c] = X[c].replace({"MISSING_NR": float("nan"),
                                     "MISSING_NA": float("nan")})
            X[c] = X[c].where(~is_nr, other=float("nan"))
            X[c] = X[c].where(~is_na, other=0.0)
            X[c] = X[c].astype(float, errors="ignore")
        return X


# ---------------------------------------------------------------------------
# Custom transformer: ApplyNA_NR_Semantics (kept for categorical columns)
# ---------------------------------------------------------------------------

class ApplyNA_NR_Semantics(BaseEstimator, TransformerMixin):
    """
    Applies token replacement for categorical columns only.
    Numeric columns are handled by AddExistsFlags.

    For categorical columns:
      - true NaN          -> "__MISSING__"
      - NR token          -> "__NOT_REPORTED__"
      - NA token          -> "__NOT_APPLICABLE__"
    """
    def __init__(
        self,
        categorical_base_cols,
        nan_cat_token="__MISSING__",
        nr_cat_token="__NOT_REPORTED__",
        na_cat_token="__NOT_APPLICABLE__",
    ):
        self.categorical_base_cols = categorical_base_cols
        self.nan_cat_token         = nan_cat_token
        self.nr_cat_token          = nr_cat_token
        self.na_cat_token          = na_cat_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in list(self.categorical_base_cols):
            col = X[c].astype(object)
            col = col.where(~col.isna(), other=self.nan_cat_token)
            col = col.where(~col.astype(str).eq("MISSING_NR"), other=self.nr_cat_token)
            col = col.where(~col.astype(str).eq("MISSING_NA"), other=self.na_cat_token)
            X[c] = col.astype(str)
        return X


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor_keep_nan_for_xgb(numeric_cols_all, categorical_cols,
                                         numeric_base_cols):
    """
    Preprocessing for XGBoost and HGBR:
      - AddExistsFlags: creates __exists flags, handles numeric sentinels
      - ApplyNA_NR_Semantics: handles categorical sentinels
      - Numeric columns passed through as-is (XGBoost/HGBR handle NaN natively)
      - Categorical columns: constant-imputed + OHE sparse
    """
    add_flags = AddExistsFlags(numeric_base_cols)
    semantics = ApplyNA_NR_Semantics(categorical_base_cols=categorical_cols)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot",  _ohe_sparse()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols_all),
            ("cat", cat_pipe,      categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("add_flags", add_flags),
        ("semantics", semantics),
        ("ct",        ct),
    ])


def build_preprocessor_for_rf_mean_zscore(numeric_cols_all, categorical_cols,
                                           numeric_base_cols):
    """
    Preprocessing for Random Forest:
      - AddExistsFlags + ApplyNA_NR_Semantics
      - Numeric: mean imputation + z-score standardisation
      - Categorical: constant-imputed + OHE dense
    """
    add_flags = AddExistsFlags(numeric_base_cols)
    semantics = ApplyNA_NR_Semantics(categorical_base_cols=categorical_cols)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot",  _ohe_dense()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols_all),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("add_flags", add_flags),
        ("semantics", semantics),
        ("ct",        ct),
    ])


def build_preprocessor_zscore_sparse(numeric_base_cols, numeric_flag_cols,
                                      categorical_cols):
    """
    Preprocessing for Ridge and PCA+LR:
      - AddExistsFlags + ApplyNA_NR_Semantics
      - Numeric base: mean imputation + z-score scaling
      - Numeric flags (__exists): passed through unchanged
      - Categorical: OHE sparse
    """
    add_flags = AddExistsFlags(numeric_base_cols)
    semantics = ApplyNA_NR_Semantics(categorical_base_cols=categorical_cols)

    num_base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  _ohe_sparse()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num_base",  num_base_pipe,   numeric_base_cols),
            ("num_flags", "passthrough",   numeric_flag_cols),
            ("cat",       cat_pipe,        categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("add_flags", add_flags),
        ("semantics", semantics),
        ("ct",        ct),
    ])


def build_preprocessor_zscore_dense_svr(numeric_base_cols, numeric_flag_cols,
                                         categorical_cols):
    """
    Preprocessing for SVR:
      - AddExistsFlags + ApplyNA_NR_Semantics
      - Numeric base: mean imputation + z-score scaling
      - Numeric flags (__exists): passed through unchanged
      - Categorical: OHE dense — NOT scaled globally

    Note: No final global StandardScaler. StandardScaler applied only to
    numeric base columns. Scaling binary OHE columns is not recommended.
    """
    add_flags = AddExistsFlags(numeric_base_cols)
    semantics = ApplyNA_NR_Semantics(categorical_base_cols=categorical_cols)

    num_base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  _ohe_dense()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num_base",  num_base_pipe,   numeric_base_cols),
            ("num_flags", "passthrough",   numeric_flag_cols),
            ("cat",       cat_pipe,        categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("add_flags", add_flags),
        ("semantics", semantics),
        ("ct",        ct),
        # No final_scaler — scaling OHE binary columns not recommended for SVR
    ])