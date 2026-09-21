# metrics.py
# Computes RMSE and R2 (coefficient of determination) with bootstrap CIs.
# R2 is computed using sklearn.metrics.r2_score, which uses the mean of
# y_true (the evaluated set) in the denominator — consistent with the
# standard mathematical definition:
#   R2 = 1 - SS_res / SS_tot
#   SS_res = sum((y_true - y_pred)^2)
#   SS_tot = sum((y_true - mean(y_true))^2)
# During CV, y_true is the validation fold so R2 uses the validation fold mean.
# For final train/test evaluation, y_true is the respective set.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2_score


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between actual and predicted values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    """
    R2 (coefficient of determination) computed using sklearn.metrics.r2_score.
    Denominator uses mean(y_true) of the evaluated set, consistent with the
    standard mathematical definition and sklearn convention.

    R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

    Parameters
    ----------
    y_true : actual target values of the evaluated set
    y_pred : predicted target values
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(sklearn_r2_score(y_true, y_pred))


def _check_inputs(y_true, y_pred, name="set"):
    """Validate that y_true and y_pred are finite arrays of equal length."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"{name}: y_true and y_pred length mismatch.")
    if len(y_true) < 2:
        raise ValueError(f"{name}: need at least 2 samples.")
    if np.any(~np.isfinite(y_true)):
        raise ValueError(f"{name}: y_true contains NaN/Inf.")
    if np.any(~np.isfinite(y_pred)):
        raise ValueError(f"{name}: y_pred contains NaN/Inf.")
    return y_true, y_pred


def bootstrap_r2(y_true, y_pred, B=2000, ci=0.95, random_state=42):
    """
    Bootstrap resampling to estimate uncertainty of RMSE and R2.

    In each of B resamples, actual and predicted value pairs are drawn with
    replacement from the evaluated set and metrics are recomputed. The 2.5th
    and 97.5th percentiles of the resulting distributions are reported as
    the 95% CI. The model is not retrained during bootstrapping.

    R2 in each resample uses mean(y_true_resample) in the denominator,
    consistent with sklearn r2_score convention.

    Parameters
    ----------
    y_true       : actual values of the evaluated set
    y_pred       : predicted values
    B            : number of bootstrap resamples (default 2000)
    ci           : confidence interval level (default 0.95 -> 95% CI)
    random_state : reproducibility seed
    """
    y_true, y_pred = _check_inputs(y_true, y_pred, name="bootstrap")
    n   = len(y_true)
    rng = np.random.default_rng(int(random_state))

    # Point estimates
    rmse0 = rmse(y_true, y_pred)
    r2_0  = r2(y_true, y_pred)

    # Bootstrap resamples
    rmse_b = np.empty(int(B), dtype=float)
    r2_b   = np.empty(int(B), dtype=float)

    for b in range(int(B)):
        idx        = rng.integers(0, n, size=n)
        rmse_b[b]  = rmse(y_true[idx], y_pred[idx])
        r2_b[b]    = r2(y_true[idx], y_pred[idx])

    # Percentile CIs
    alpha   = (1.0 - float(ci)) / 2.0
    rmse_lo = float(np.quantile(rmse_b, alpha))
    rmse_hi = float(np.quantile(rmse_b, 1 - alpha))
    r2_lo   = float(np.quantile(r2_b,   alpha))
    r2_hi   = float(np.quantile(r2_b,   1 - alpha))

    ci_pct = int(ci * 100)
    return {
        "n":                          int(n),
        "RMSE":                       float(rmse0),
        "RMSE_boot_sd":               float(np.std(rmse_b, ddof=1)),
        f"RMSE_ci{ci_pct}_low":       rmse_lo,
        f"RMSE_ci{ci_pct}_high":      rmse_hi,
        "R2":                         float(r2_0),
        "R2_boot_sd":                 float(np.std(r2_b, ddof=1)),
        f"R2_ci{ci_pct}_low":         r2_lo,
        f"R2_ci{ci_pct}_high":        r2_hi,
    }


def save_train_test_metrics_csv(
    y_train, pred_train,
    y_test,  pred_test,
    out_csv_path,
    B=2000, ci=0.95, random_state=30,
):
    """
    Saves a 2-row CSV (train row + test row) with RMSE and R2 bootstrap stats.
    R2 uses mean(y_true) of the evaluated set in the denominator,
    consistent with sklearn r2_score convention.
    """
    tr = bootstrap_r2(y_train, pred_train, B=B, ci=ci, random_state=random_state)
    te = bootstrap_r2(y_test,  pred_test,  B=B, ci=ci, random_state=random_state)
    df = pd.DataFrame([
        {"split": "train", **tr},
        {"split": "test",  **te},
    ])
    df.to_csv(out_csv_path, index=False)
    return df