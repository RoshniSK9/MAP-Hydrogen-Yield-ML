# data.py
# Loads the Excel dataset, drops rows with missing target values,
# and returns base features only — NO flag columns created here.
#
# UPDATED: Flag columns (__exists) are no longer created in this file.
# They are created INSIDE the preprocessing pipeline via AddExistsFlags,
# ensuring they are created independently for training and test sets
# after the train-test split — consistent with how OneHotEncoder handles
# categorical columns.
#
# Sentinel strings (MISSING_NA, MISSING_NR) are preserved in the returned
# X so the pipeline can detect and handle them correctly.

import numpy as np
import pandas as pd

# Sentinel strings used in the Excel dataset to mark special missing types
MISSING_NA = "MISSING_NA"   # feature is not applicable to this experiment
MISSING_NR = "MISSING_NR"   # feature exists but was not reported in the paper


def load_and_clean_excel(file_path: str, target_col: str, categorical_cols: list):
    """
    Loads the Excel file and prepares X, y for modelling.

    Steps:
      1. Read the Excel file into a DataFrame.
      2. Drop rows where the target (H2 yield) is missing or a sentinel string.
      3. Classify columns into numeric base cols and categorical cols.
      4. Return X with sentinel strings preserved for pipeline handling.

    Flag columns (__exists) are NOT created here — they are created inside
    the preprocessing pipeline after the train-test split via AddExistsFlags.

    Physical meaning of sentinels:
      MISSING_NR: feature exists in experiment but was not reported
                  → __exists = 1, value gets mean imputed downstream
      MISSING_NA: feature does not apply to this experiment
                  → __exists = 0, value set to 0.0 downstream

    Returns
    -------
    X                : pd.DataFrame  — raw input features, sentinel strings preserved
    y                : pd.Series     — target hydrogen yield values
    numeric_cols_all : list          — numeric base cols + expected __exists flag col names
                                       (flag cols created by pipeline, not present in X yet)
    cat_cols         : list          — categorical column names
    numeric_base_cols: list          — only the raw numeric input features (no flags)
    """
    # --- 1. Read Excel ---
    df = pd.read_excel(file_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column not found: {target_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    # --- 2. Drop rows with missing target ---
    y_raw = df[target_col].replace({MISSING_NA: np.nan, MISSING_NR: np.nan})
    y_raw = y_raw.infer_objects(copy=False)
    y_num = pd.to_numeric(y_raw, errors="coerce")
    keep  = y_num.notna()
    df    = df.loc[keep].reset_index(drop=True)
    y     = y_num.loc[keep].astype(float).reset_index(drop=True)

    # --- 3. Build X — raw features only, NO flags ---
    X = df.drop(columns=[target_col]).copy()

    # --- 4. Classify columns ---
    cat_cols          = [c for c in categorical_cols if c in X.columns]
    numeric_base_cols = [c for c in X.columns if c not in cat_cols]

    # --- 5. Define expected flag column names ---
    # These do not exist in X yet — AddExistsFlags creates them inside
    # the pipeline after the train-test split.
    exists_cols      = [f"{c}__exists" for c in numeric_base_cols]
    numeric_cols_all = numeric_base_cols + exists_cols

    return X, y, numeric_cols_all, cat_cols, numeric_base_cols