# run.py
#
# Runs all 6 models in sequence:
#   Tree-based:    XGBoost, HGBR, Random Forest
#   Linear:        Ridge, PCA+LR
#   Kernel-based:  SVR
#
# For each model:
#   1. Hyperparameter tuning via RandomizedSearchCV (k-fold, R2 scorer)
#   2. Train/test metrics with bootstrap CIs
#   3. Parity plots (IJHE style)
#   4. SHAP analysis (appropriate explainer for each model family)
#
# All results are saved to out_root/run_<timestamp>/
#
# UPDATED:
#   - numeric_flag_cols now uses __exists naming (single flag per feature)
#   - Train-test split performed on base features only (no flags pre-created)
#   - Flag creation handled inside preprocessing pipeline via AddExistsFlags
#   - Train and test sets saved to output directory after split

import os
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import make_scorer

from .data import load_and_clean_excel
from .metrics import rmse, r2, save_train_test_metrics_csv, bootstrap_r2
from .plots import save_train_test_parity
from .models import get_pipeline_models
from .feature_importance import save_xgb_feature_importance
from .shap_analysis import (
    shap_for_sklearn_pipeline_tree_model,
    shap_for_sklearn_pipeline_linear_model,
    shap_for_sklearn_pipeline_svr,
    DEFAULT_GROUP_COLS,
)


# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

def _make_r2_scorer():
    """
    Creates a sklearn scorer using sklearn R2.
    sklearn maximises scorers, and R2 is naturally maximised (higher = better).
    """
    from sklearn.metrics import r2_score as sklearn_r2_score
    return make_scorer(sklearn_r2_score)


# ---------------------------------------------------------------------------
# Hyperparameter tuning for sklearn pipelines
# ---------------------------------------------------------------------------

def tune_pipeline_r2(
    pipe, param_dist, X_train, y_train,
    k, n_iter, random_state, n_jobs_search=1,
):
    """
    Runs k-fold RandomizedSearchCV for a sklearn Pipeline scored by R2.

    Uses refit=False to avoid redundant refitting during search, then manually
    refits the best parameters on the full training set.
    """
    r2_scorer = _make_r2_scorer()

    cv = KFold(n_splits=int(k), shuffle=True, random_state=int(random_state))

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=int(n_iter),
        scoring=r2_scorer,
        cv=cv,
        random_state=int(random_state),
        n_jobs=int(n_jobs_search),
        verbose=1,
        refit=False,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_r2  = search.best_score_

    final = clone(pipe).set_params(**best_params)
    final.fit(X_train, y_train)

    return final, best_params, float(best_cv_r2)


# ---------------------------------------------------------------------------
# Helper: evaluate and save metrics for one model
# ---------------------------------------------------------------------------

def _evaluate_and_save(
    name, model,
    X_train, y_train, pred_tr,
    X_test,  y_test,  pred_te,
    out_dir, cv_r2,
    metrics_bootstrap_B, metrics_ci,
    random_state, zoom_max,
    results_list,
):
    """
    Computes train/test metrics, saves parity plots, and appends to results_list.
    """
    tr_rmse = rmse(y_train, pred_tr)
    te_rmse = rmse(y_test,  pred_te)
    tr_r2   = r2(y_train, pred_tr)
    te_r2   = r2(y_test,  pred_te)

    print(f"CV best R2 (validation folds): {cv_r2:.4f}")
    print(f"TRAIN: RMSE={tr_rmse:.4f}, R2={tr_r2:.4f}")
    print(f"TEST : RMSE={te_rmse:.4f}, R2={te_r2:.4f}")

    ci_pct  = int(metrics_ci * 100)
    te_boot = {f"R2_ci{ci_pct}_low": float("nan"), f"R2_ci{ci_pct}_high": float("nan")}

    try:
        te_boot = bootstrap_r2(
            y_test, pred_te,
            B=int(metrics_bootstrap_B), ci=float(metrics_ci),
            random_state=int(random_state),
        )
        print(f"TEST  R2 {ci_pct}% CI: "
              f"[{te_boot[f'R2_ci{ci_pct}_low']:.4f}, "
              f"{te_boot[f'R2_ci{ci_pct}_high']:.4f}]")
    except Exception:
        print(f"[WARN] Metrics bootstrap failed for {name}:")
        traceback.print_exc()

    save_train_test_parity(
        y_train, pred_tr, y_test, pred_te,
        out_dir, name, zoom_max=zoom_max,
    )

    results_list.append({
        "model":                                    name,
        "cv_best_r2":                               cv_r2,
        "train_rmse":                               tr_rmse,
        "train_r2":                                 tr_r2,
        "test_rmse":                                te_rmse,
        "test_r2":                                  te_r2,
        f"test_r2_ci{ci_pct}_low":                  te_boot.get(f"R2_ci{ci_pct}_low",  float("nan")),
        f"test_r2_ci{ci_pct}_high":                 te_boot.get(f"R2_ci{ci_pct}_high", float("nan")),
        "best_params":                              str({}),
    })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all(
    file_path,
    target_col,
    categorical_cols,
    test_size=0.2,
    random_state=30,
    k=10,
    n_iter_search=50,
    n_jobs_model=2,
    n_jobs_search=1,
    zoom_max=80,
    out_root="results_ml_project_unified",
    metrics_bootstrap_B=2000,
    metrics_ci=0.95,
    shap_analysis=True,
    shap_max_display=27,
    shap_waterfall_row=0,
    shap_waterfall_max_display=20,
    svr_shap_sample_rows=100,
    svr_shap_background_rows=50,
):
    """
    Runs the full unified ML pipeline and saves all outputs.
    """
    # --- Create timestamped output directory ---
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"run_{run_tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(out_dir)}")

    # --- Load and clean data ---
    # Returns X with raw sentinel strings preserved (no flags created yet)
    X, y, numeric_cols_all, cat_cols, numeric_base_cols = load_and_clean_excel(
        file_path, target_col, categorical_cols
    )
    print(f"Dataset: {len(y)} samples, {len(numeric_base_cols)} numeric base cols, "
          f"{len(cat_cols)} categorical cols")

    # --- UPDATED: Derive __exists flag column names ---
    # These names are expected by the ColumnTransformer after AddExistsFlags
    # creates them inside the pipeline. They do not exist in X yet.
    numeric_flag_cols = [f"{c}__exists" for c in numeric_base_cols]
    numeric_cols_all  = numeric_base_cols + numeric_flag_cols

    # --- UPDATED: Train-test split on BASE features only ---
    # Flag columns are created inside the pipeline after the split,
    # ensuring they are fitted on training data only — consistent with
    # how OneHotEncoder handles categorical columns.
    X_train, X_test, y_train, y_test = train_test_split(
        X[numeric_base_cols + cat_cols], y,
        test_size=float(test_size),
        random_state=int(random_state),
        shuffle=True,
    )
    print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples")

    # --- Save train and test sets ---
    # Saves raw base features + target for each set before any preprocessing.
    # Useful for verifying the split and inspecting individual datapoints
    # (e.g. checking dielectric properties in Table S3).
    train_out = X_train.copy()
    train_out[target_col] = y_train.values
    train_csv = os.path.join(out_dir, "train_set.csv")
    train_out.to_csv(train_csv, index=True)
    print(f"Train set saved: {train_csv}")

    test_out = X_test.copy()
    test_out[target_col] = y_test.values
    test_csv = os.path.join(out_dir, "test_set.csv")
    test_out.to_csv(test_csv, index=True)
    print(f"Test set saved:  {test_csv}")

    # --- Baseline: predict training mean ---
    y_mean          = float(y_train.mean())
    pred_base_train = np.full(len(y_train), y_mean, dtype=float)
    pred_base_test  = np.full(len(y_test),  y_mean, dtype=float)

    print("\n=== BASELINE (predict TRAIN mean) ===")
    print(f"TRAIN: RMSE={rmse(y_train, pred_base_train):.4f}, "
          f"R2={r2(y_train, pred_base_train):.4f}")
    print(f"TEST : RMSE={rmse(y_test, pred_base_test):.4f}, "
          f"R2={r2(y_test, pred_base_test):.4f}")

    try:
        base_stats = bootstrap_r2(
            y_true=y_test, y_pred=pred_base_test,
            B=int(metrics_bootstrap_B), ci=float(metrics_ci),
            random_state=int(random_state),
        )
        print(f"BASELINE TEST R2 SD (boot): {base_stats['R2_boot_sd']:.4f}")
    except Exception:
        print("[WARN] Baseline bootstrap uncertainty failed:")
        traceback.print_exc()

    results = []

    # --- Get all model specs ---
    specs = get_pipeline_models(
        numeric_cols_all=numeric_cols_all,
        cat_cols=cat_cols,
        numeric_base_cols=numeric_base_cols,
        numeric_flag_cols=numeric_flag_cols,
        random_state=random_state,
        n_jobs_model=n_jobs_model,
    )

    _TREE_MODELS   = {"XGBoost", "RandomForest", "HGBR"}
    _LINEAR_MODELS = {"Ridge"}
    _SVR_MODELS    = {"SVR"}

    for name, (pipe, param_dist) in specs.items():
        print(f"\n\n==================== {name} ====================")

        # --- Hyperparameter tuning ---
        try:
            model, best_params, cv_r2 = tune_pipeline_r2(
                pipe, param_dist,
                X_train, y_train,
                k, n_iter_search, random_state,
                n_jobs_search=n_jobs_search,
            )
        except Exception:
            print(f"[ERROR] Tuning failed for {name}:")
            traceback.print_exc()
            continue

        # --- Predictions ---
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

        # --- Metrics and parity plots ---
        _evaluate_and_save(
            name=name, model=model,
            X_train=X_train, y_train=y_train, pred_tr=pred_tr,
            X_test=X_test,   y_test=y_test,   pred_te=pred_te,
            out_dir=out_dir, cv_r2=cv_r2,
            metrics_bootstrap_B=metrics_bootstrap_B,
            metrics_ci=metrics_ci,
            random_state=random_state,
            zoom_max=zoom_max,
            results_list=results,
        )
        results[-1]["best_params"] = str(best_params)

        # --- SHAP analysis ---
        if shap_analysis:
            try:
                if name in _TREE_MODELS:
                    shap_for_sklearn_pipeline_tree_model(
                        model_pipeline=model,
                        X=X_test,
                        out_dir=out_dir,
                        model_name=name,
                        dataset_label="test",
                        sample_rows=0,
                        random_state=random_state,
                        max_display=shap_max_display,
                        exclude_categorical_cols=DEFAULT_GROUP_COLS,
                        waterfall_row=shap_waterfall_row,
                        waterfall_max_display=shap_waterfall_max_display,
                        shap_stability=False,
                        dependence_plots=(name == "XGBoost"),
                        dependence_top_n=10,
                    )
                elif name in _LINEAR_MODELS:
                    shap_for_sklearn_pipeline_linear_model(
                        model_pipeline=model,
                        X_train=X_train,
                        X_test=X_test,
                        out_dir=out_dir,
                        model_name=name,
                        dataset_label="test",
                        sample_rows=0,
                        random_state=random_state,
                        max_display=shap_max_display,
                        exclude_categorical_cols=DEFAULT_GROUP_COLS,
                        waterfall_row=shap_waterfall_row,
                        waterfall_max_display=shap_waterfall_max_display,
                        shap_stability=False,
                    )
                elif name in _SVR_MODELS:
                    shap_for_sklearn_pipeline_svr(
                        model_pipeline=model,
                        X_train=X_train,
                        X_test=X_test,
                        out_dir=out_dir,
                        model_name=name,
                        dataset_label="test",
                        sample_rows=svr_shap_sample_rows,
                        background_rows=svr_shap_background_rows,
                        random_state=random_state,
                        max_display=shap_max_display,
                        exclude_categorical_cols=DEFAULT_GROUP_COLS,
                        waterfall_row=shap_waterfall_row,
                        waterfall_max_display=shap_waterfall_max_display,
                        shap_stability=False,
                    )
            except Exception:
                print(f"[WARN] SHAP failed for {name}:")
                traceback.print_exc()

        # --- XGBoost built-in feature importance ---
        if name == "XGBoost":
            try:
                shap_csv = os.path.join(
                    out_dir, f"shap_meanabs_{name}_test.csv"
                )
                save_xgb_feature_importance(
                    model_pipeline=model,
                    out_dir=out_dir,
                    model_name=name,
                    top_n=15,
                    shap_csv_path=shap_csv if os.path.exists(shap_csv) else None,
                )
            except Exception:
                print(f"[WARN] Feature importance failed for {name}:")
                traceback.print_exc()

    # --- Save summary results ---
    df_res  = pd.DataFrame(results).sort_values("test_r2", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(out_dir, f"summary_results_run_{run_tag}.csv")
    df_res.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print("ALL MODELS COMPLETE")
    print(f"{'='*60}")
    print(df_res[["model", "train_r2", "test_r2"]].to_string(index=False))
    print(f"\nSaved summary: {os.path.abspath(out_csv)}")
    print(f"Run directory:  {os.path.abspath(out_dir)}")

    return df_res