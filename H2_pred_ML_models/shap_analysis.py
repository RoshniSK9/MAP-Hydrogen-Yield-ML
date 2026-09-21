# shap_analysis.py
# Computes and saves SHAP analysis for all model families.
#
# Rules (matching the existing IJHE-style beeswarm):
#   1. Flag features (__is_NR, __is_NA, __exists) are EXCLUDED from all plots.
#      Only the 25 raw numeric input features appear on the y-axis.
#   2. Catalyst (-) and Microwave absorber (-) OHE columns are EXCLUDED.
#      Their influence is captured indirectly through the associated numeric
#      features (catalyst surface area, pore diameter, metal loading, dielectric
#      properties) — keeping the plot clean and interpretable.
#   3. NaN values are passed through to SHAP as-is so they render as grey dots
#      (SHAP's default behaviour for missing values).
#   4. The "num__" prefix added by ColumnTransformer is stripped from labels.
#   5. All plots use the same box style and font weight as the parity plots.
#
# Explainer used per model family:
#   - XGBoost, RandomForest, HGBR : TreeExplainer  (fast, exact)
#   - Ridge, PCA+LR               : LinearExplainer (fast, exact)
#   - SVR                         : KernelExplainer (slow, approximate)

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import sparse

# Categorical columns excluded from ALL SHAP plots.
DEFAULT_GROUP_COLS = ["Catalyst (-)", "Microwave absorber (-)"]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _is_flag_feature(name):
    """
    Returns True for __is_NR, __is_NA, and __exists columns — always excluded
    from plots. Also catches num_flags__-prefixed versions produced by the
    linear model ColumnTransformer.
    """
    s = str(name)
    for prefix in ["num_base__", "num_flags__", "num__", "cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.endswith("__is_NR") or s.endswith("__is_NA") or s.endswith("__exists")


def _is_excluded_categorical(name, exclude_cols):
    """
    Returns True if the feature belongs to an excluded categorical column.
    Handles all ColumnTransformer prefix variants and both sklearn OHE names
    and LightGBM native names.
    """
    s = str(name)
    for prefix in ["num_base__", "num_flags__", "num__", "cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for col in exclude_cols:
        if s == col or s.startswith(col + "_") or s.startswith(col + "__"):
            return True
    return False


def _clean_label(name):
    """
    Strips ColumnTransformer prefixes so feature labels match the original
    column names from the Excel file.
    """
    s = str(name)
    for prefix in ["num_base__", "num_flags__", "num__", "cat__"]:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _sample_rows_df(X, max_rows, random_state):
    """Randomly subsample up to max_rows rows. Returns X unchanged if max_rows <= 0."""
    max_rows = int(max_rows)
    if max_rows <= 0 or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X.iloc[idx]


def _get_feature_names_from_pipeline(pre):
    """
    Extracts feature names from a fitted sklearn Pipeline's ColumnTransformer.
    Searches all named steps for any ColumnTransformer and calls
    get_feature_names_out() on the first one found.
    """
    from sklearn.compose import ColumnTransformer as CT
    if not hasattr(pre, "named_steps"):
        return None
    for step_name, step_obj in pre.named_steps.items():
        if isinstance(step_obj, CT):
            try:
                return [str(n) for n in step_obj.get_feature_names_out()]
            except Exception:
                return None
        if hasattr(step_obj, "named_steps"):
            for inner_name, inner_obj in step_obj.named_steps.items():
                if isinstance(inner_obj, CT):
                    try:
                        return [str(n) for n in inner_obj.get_feature_names_out()]
                    except Exception:
                        return None
    return None


def _set_symmetric_xlim():
    """Centres the SHAP beeswarm x-axis symmetrically around zero."""
    ax = plt.gca()
    x0, x1 = ax.get_xlim()
    lim = max(abs(float(x0)), abs(float(x1)))
    if lim > 0:
        ax.set_xlim(-lim, lim)


def _save_mean_abs_csv(shap_vals, feature_names, out_path):
    """Saves mean |SHAP| per feature as a CSV, sorted descending."""
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(out_path, index=False)
    return df


def _filter_features(feature_names, shap_vals, X_data, exclude_cols):
    """
    Removes flag features and excluded categorical OHE columns from SHAP arrays.
    Returns cleaned (feat_labels, sv_plot, X_plot_filtered).
    """
    keep_idx = [
        i for i, fn in enumerate(feature_names)
        if not _is_flag_feature(fn)
        and not _is_excluded_categorical(fn, exclude_cols)
    ]
    feat_labels = [_clean_label(feature_names[i]) for i in keep_idx]
    sv_plot     = shap_vals[:, keep_idx]
    if isinstance(X_data, pd.DataFrame):
        X_filtered = X_data.iloc[:, keep_idx].copy()
        X_filtered.columns = feat_labels
    else:
        X_filtered = X_data[:, keep_idx]
    return feat_labels, sv_plot, X_filtered


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def _bootstrap_shap_importance_stats(shap_vals, B=2000, random_state=42):
    """Bootstrap resampling to estimate uncertainty of mean |SHAP| per feature."""
    rng  = np.random.default_rng(int(random_state))
    S    = np.asarray(shap_vals, dtype=float)
    n, p = S.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    absS  = np.abs(S)
    point = absS.mean(axis=0)
    boot  = np.empty((int(B), p), dtype=float)
    for b in range(int(B)):
        idx        = rng.integers(0, n, size=n)
        boot[b, :] = absS[idx, :].mean(axis=0)
    sd = np.std(boot, axis=0, ddof=1)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    return point, sd, lo, hi


def _bootstrap_topk_frequency(shap_vals, K=10, B=2000, random_state=42):
    """Fraction of bootstrap samples in which each feature appears in the top-K."""
    rng    = np.random.default_rng(int(random_state))
    S      = np.asarray(shap_vals, dtype=float)
    n, p   = S.shape
    absS   = np.abs(S)
    counts = np.zeros(p, dtype=int)
    for b in range(int(B)):
        idx  = rng.integers(0, n, size=n)
        topk = np.argsort(absS[idx, :].mean(axis=0))[::-1][:int(K)]
        counts[topk] += 1
    return counts / float(B)


def _bootstrap_rank_stability(shap_vals, B=500, random_state=42):
    """Mean Spearman rank correlation between full-data ranking and bootstrap rankings."""
    rng       = np.random.default_rng(int(random_state))
    S         = np.asarray(shap_vals, dtype=float)
    n, p      = S.shape
    absS      = np.abs(S)
    full_rank = np.argsort(np.argsort(-absS.mean(axis=0)))

    def _spearman(r1, r2):
        r1 = r1.astype(float) - r1.mean()
        r2 = r2.astype(float) - r2.mean()
        d  = np.sqrt((r1**2).sum()) * np.sqrt((r2**2).sum())
        return float((r1 * r2).sum() / d) if d != 0 else np.nan

    cors = []
    for _ in range(int(B)):
        idx  = rng.integers(0, n, size=n)
        rank = np.argsort(np.argsort(-absS[idx, :].mean(axis=0)))
        cors.append(_spearman(full_rank, rank))
    return float(np.nanmean(cors))


def _write_shap_stability_outputs(shap_vals, feature_names, out_dir, tag,
                                   B, topk, random_state, rank_check):
    """Saves stability CSV and rank stability TXT."""
    point, sd, lo, hi = _bootstrap_shap_importance_stats(
        shap_vals, B=B, random_state=random_state
    )
    topk_freq = _bootstrap_topk_frequency(
        shap_vals, K=topk, B=B, random_state=random_state
    )
    rho = (
        _bootstrap_rank_stability(shap_vals, B=min(500, B), random_state=random_state)
        if rank_check else np.nan
    )
    (pd.DataFrame({
        "feature":               list(feature_names),
        "mean_abs_shap":         point,
        "mean_abs_shap_boot_sd": sd,
        "ci95_low":              lo,
        "ci95_high":             hi,
        f"top{topk}_freq":       topk_freq,
    }).sort_values("mean_abs_shap", ascending=False)
      .reset_index(drop=True)
      .to_csv(os.path.join(out_dir, f"shap_stability_{tag}.csv"), index=False))
    with open(os.path.join(out_dir, f"shap_rank_stability_{tag}.txt"), "w") as f:
        f.write(f"Rank stability (mean Spearman rho vs full rank): {rho:.3f}\n")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_shap_box_style(ax):
    """All four spines visible; ticks only on bottom and left."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    ax.tick_params(bottom=True, left=True, direction="in", which="both")
    ax.tick_params(top=False, right=False, which="both")


def _format_feature_labels(feature_names):
    """
    Applies display formatting to feature labels for SHAP plots.
    Replaces plain text units with properly formatted matplotlib/LaTeX equivalents,
    and renames specific features for display purposes.
    Currently handles:
      - 'm2/g'                                 -> 'm$^2$/g'
      - 'Feedstock: Catalyst ratio (wt/wt)'   -> 'Feedstock to catalyst ratio (wt/wt)'
      - 'Feedstock:Absorber ratio (wt/wt)'    -> 'Feedstock to absorber ratio (wt/wt)'
      - 'Dielectric loss of absorber (tan δ)' -> 'Dielectric loss tangent of absorber (tan δ)'
    """
    # Display name remapping — keys are exact column names from the dataset
    _LABEL_MAP = {
        "Feedstock: Catalyst ratio (wt/wt)":   "Feedstock to catalyst ratio (wt/wt)",
        "Feedstock:Absorber ratio (wt/wt)":    "Feedstock to absorber ratio (wt/wt)",
        "Dielectric loss of absorber (tan δ)": "Dielectric loss tangent of absorber (tan δ)",
        "Reaction time (min)":                 "Isothermal time (min)",
    }
    formatted = []
    for name in feature_names:
        # Apply label remapping first
        name = _LABEL_MAP.get(name, name)
        # Then apply unit formatting
        name = name.replace("m2/g", r"m$^2$/g")
        formatted.append(name)
    return formatted


def _save_beeswarm_and_bar(shap_vals, X_plot, feature_names, out_dir, tag, max_display):
    """
    Saves beeswarm and bar SHAP summary plots.
    X_plot must contain the ORIGINAL (un-preprocessed) feature values so that
    SHAP can colour dots correctly (red = high, blue = low, grey = missing).
    """
    import shap

    display_names = _format_feature_labels(feature_names)

    _shap_font = {
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":        12,
        "axes.labelsize":   12,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "axes.labelweight": "bold",
        "font.weight":      "bold",
    }

    def _style_ax(ax):
        _apply_shap_box_style(ax)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_size(12)
        ax.yaxis.label.set_weight("bold")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(12)
            lbl.set_fontweight("bold")

    # --- beeswarm ---
    with mpl.rc_context(_shap_font):
        plt.figure()
        shap.summary_plot(
            shap_vals, X_plot,
            feature_names=display_names,
            show=False,
            max_display=int(max_display),
            plot_type="dot",
        )
        _set_symmetric_xlim()
        _style_ax(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_beeswarm_{tag}.png"), dpi=300)
        plt.close()

    # --- bar ---
    with mpl.rc_context(_shap_font):
        plt.figure()
        shap.summary_plot(
            shap_vals, X_plot,
            feature_names=display_names,
            show=False,
            max_display=int(max_display),
            plot_type="bar",
        )
        _style_ax(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_bar_{tag}.png"), dpi=300)
        plt.close()


def _save_waterfall(base_value, shap_vals_1d, data_1d, feature_names,
                    out_path, max_display=20):
    """Saves a SHAP waterfall plot for a single sample."""
    import shap
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])
    exp = shap.Explanation(
        values=np.asarray(shap_vals_1d, dtype=float),
        base_values=float(base_value),
        data=np.asarray(data_1d, dtype=object),
        feature_names=list(feature_names),
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=int(max_display), show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _save_dependence_plots(shap_vals, X_display, feature_names,
                            out_dir, tag, top_n=10):
    """
    Saves SHAP dependence plots for the top N features by mean absolute SHAP value.
    Each plot shows SHAP value vs actual feature value for one feature,
    with colour representing the feature value (matching beeswarm style).
    """
    _shap_font = {
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":        12,
        "axes.labelsize":   12,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "axes.labelweight": "bold",
        "font.weight":      "bold",
    }

    mean_abs    = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:int(top_n)]

    _ensure_dir(out_dir)

    for idx in top_indices:
        feat_name    = feature_names[idx]
        display_name = _format_feature_labels([feat_name])[0]

        if isinstance(X_display, pd.DataFrame):
            if feat_name in X_display.columns:
                feat_vals = X_display[feat_name].values.astype(float)
            else:
                continue
        else:
            feat_vals = X_display[:, idx].astype(float)

        shap_col = shap_vals[:, idx]

        with mpl.rc_context(_shap_font):
            fig, ax = plt.subplots(figsize=(5.0, 4.0))
            valid_mask = ~np.isnan(feat_vals)
            nan_mask   =  np.isnan(feat_vals)

            if nan_mask.any():
                ax.scatter(
                    feat_vals[nan_mask], shap_col[nan_mask],
                    c="grey", s=18, alpha=0.8, edgecolors="none", zorder=2,
                )
            if valid_mask.any():
                sc = ax.scatter(
                    feat_vals[valid_mask], shap_col[valid_mask],
                    c=feat_vals[valid_mask],
                    cmap="coolwarm",
                    s=18, alpha=0.8, edgecolors="none", zorder=3,
                )
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label("Feature value", fontsize=10, fontweight="bold")
                cbar.ax.tick_params(labelsize=9)

            ax.axhline(0, color="grey", linewidth=0.75, linestyle="--", zorder=1)
            ax.set_xlabel(display_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("SHAP value", fontsize=12, fontweight="bold")
            _apply_shap_box_style(ax)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(11)
                lbl.set_fontweight("bold")
            plt.tight_layout()

            safe_name = (
                feat_name
                .replace("/", "_per_").replace(" ", "_")
                .replace("(", "").replace(")", "").replace("%", "pct")
                .replace("°", "deg").replace("'", "").replace("δ", "delta")
            )
            plt.savefig(
                os.path.join(out_dir, f"shap_dependence_{tag}_{safe_name}.png"),
                dpi=300,
            )
            plt.close()


# ---------------------------------------------------------------------------
# SHAP for sklearn Pipeline tree models (XGBoost, RandomForest, HGBR)
# ---------------------------------------------------------------------------

def shap_for_sklearn_pipeline_tree_model(
    model_pipeline, X, out_dir, model_name,
    dataset_label="test", sample_rows=0, random_state=42,
    max_display=25, exclude_categorical_cols=None,
    waterfall_row=0, waterfall_max_display=20,
    dependence_plots=False, dependence_top_n=10,
    shap_stability=False, stability_B=2000, stability_topk=10,
    stability_rank_check=True, **kwargs,
):
    """
    TreeExplainer SHAP for XGBoost, RandomForest, HGBR sklearn pipelines.
    """
    import shap

    if exclude_categorical_cols is None:
        exclude_categorical_cols = DEFAULT_GROUP_COLS

    pre = model_pipeline.named_steps["pre"]
    est = model_pipeline.named_steps["model"]

    Xs = _sample_rows_df(
        X if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
        sample_rows, random_state,
    )

    Xt = pre.transform(Xs)
    feature_names = _get_feature_names_from_pipeline(pre)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(
            Xt.shape[1] if not sparse.issparse(Xt) else Xt.shape[1]
        )]

    if sparse.issparse(Xt):
        Xt_dense = Xt.toarray()
    else:
        Xt_dense = np.asarray(Xt, dtype=float)

    explainer  = shap.TreeExplainer(est)
    shap_vals  = explainer.shap_values(
        Xt if not sparse.issparse(Xt) else Xt.tocsr()
    )
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals  = np.asarray(shap_vals, dtype=float)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])

    _ensure_dir(out_dir)
    tag = f"{model_name}_{dataset_label}".replace(os.sep, "_")

    feat_labels, sv_plot, _ = _filter_features(
        feature_names, shap_vals, Xt_dense, exclude_categorical_cols
    )
    X_for_colour = _build_colour_matrix(Xs, feat_labels)

    _save_mean_abs_csv(
        sv_plot, feat_labels,
        os.path.join(out_dir, f"shap_meanabs_{tag}.csv"),
    )

    if shap_stability:
        _write_shap_stability_outputs(
            sv_plot, feat_labels, out_dir, tag,
            stability_B, stability_topk, random_state, stability_rank_check,
        )

    _save_beeswarm_and_bar(sv_plot, X_for_colour, feat_labels, out_dir, tag, max_display)

    if dependence_plots:
        _save_dependence_plots(
            sv_plot, X_for_colour, feat_labels, out_dir, tag,
            top_n=int(dependence_top_n),
        )

    if waterfall_row is not None and len(sv_plot) > 0:
        r = max(0, min(int(waterfall_row), len(sv_plot) - 1))
        _, _, Xt_filtered = _filter_features(
            feature_names, shap_vals, Xt_dense, exclude_categorical_cols
        )
        _save_waterfall(
            base_value, sv_plot[r],
            Xt_filtered[r] if isinstance(Xt_filtered, np.ndarray)
            else Xt_filtered.iloc[r].values,
            feat_labels,
            os.path.join(out_dir, f"shap_waterfall_{tag}_row{r}.png"),
            waterfall_max_display,
        )

    return {
        "model": model_name, "dataset": dataset_label,
        "n_rows": len(Xs), "n_features_shown": len(feat_labels),
    }


# ---------------------------------------------------------------------------
# SHAP for sklearn Pipeline linear models (LR, Ridge, PCA+LR)
# ---------------------------------------------------------------------------

def shap_for_sklearn_pipeline_linear_model(
    model_pipeline, X_train, X_test, out_dir, model_name,
    dataset_label="test", sample_rows=0, random_state=42,
    max_display=25, exclude_categorical_cols=None,
    waterfall_row=0, waterfall_max_display=20,
    shap_stability=False, stability_B=2000, stability_topk=10,
    stability_rank_check=True, **kwargs,
):
    """
    LinearExplainer SHAP for Ridge and PCA+LR pipelines.
    """
    import shap

    if exclude_categorical_cols is None:
        exclude_categorical_cols = DEFAULT_GROUP_COLS

    pre   = model_pipeline.named_steps["pre"]
    Xtr_t = pre.transform(X_train)
    Xs    = _sample_rows_df(
        X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test),
        sample_rows, random_state,
    )
    Xt    = pre.transform(Xs)

    if sparse.issparse(Xtr_t): Xtr_t = Xtr_t.toarray()
    if sparse.issparse(Xt):    Xt    = Xt.toarray()
    Xtr_t = np.asarray(Xtr_t, dtype=float)
    Xt    = np.asarray(Xt,    dtype=float)

    feature_names = _get_feature_names_from_pipeline(pre)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(Xt.shape[1])]

    step_names   = list(model_pipeline.named_steps.keys())
    middle_steps = [s for s in step_names if s not in ("pre", "model")]

    if middle_steps:
        reducer     = model_pipeline.named_steps[middle_steps[0]]
        Xtr_sub     = reducer.transform(Xtr_t)
        Xt_sub      = reducer.transform(Xt)
        est         = model_pipeline.named_steps["model"]
        n_comp      = Xtr_sub.shape[1]
        feat_labels = [f"Component_{i+1}" for i in range(n_comp)]
        sv_plot     = None
        Xt_plot_f   = Xt_sub
    else:
        Xtr_sub = Xtr_t
        Xt_sub  = Xt
        est     = model_pipeline.named_steps["model"]
        feat_labels = None

    explainer  = shap.LinearExplainer(
        est, Xtr_sub, feature_perturbation="interventional"
    )
    shap_vals  = np.asarray(explainer.shap_values(Xt_sub), dtype=float)
    base_value = float(explainer.expected_value)

    _ensure_dir(out_dir)
    tag = f"{model_name}_{dataset_label}".replace(os.sep, "_")

    if middle_steps:
        sv_plot      = shap_vals
        Xt_plot_f    = Xt_sub
        X_for_colour = pd.DataFrame(Xt_plot_f, columns=feat_labels)
    else:
        feat_labels, sv_plot, Xt_plot_f = _filter_features(
            feature_names, shap_vals, Xt_sub, exclude_categorical_cols
        )
        X_for_colour = _build_colour_matrix(Xs, feat_labels)

    _save_mean_abs_csv(
        sv_plot, feat_labels,
        os.path.join(out_dir, f"shap_meanabs_{tag}.csv"),
    )

    if shap_stability:
        _write_shap_stability_outputs(
            sv_plot, feat_labels, out_dir, tag,
            stability_B, stability_topk, random_state, stability_rank_check,
        )

    _save_beeswarm_and_bar(sv_plot, X_for_colour, feat_labels, out_dir, tag, max_display)

    if waterfall_row is not None and len(sv_plot) > 0:
        r        = max(0, min(int(waterfall_row), len(sv_plot) - 1))
        data_row = (
            Xt_plot_f[r]
            if isinstance(Xt_plot_f, np.ndarray)
            else Xt_plot_f.iloc[r].values
        )
        _save_waterfall(
            base_value, sv_plot[r], data_row, feat_labels,
            os.path.join(out_dir, f"shap_waterfall_{tag}_row{r}.png"),
            waterfall_max_display,
        )

    return {
        "model": model_name, "dataset": dataset_label,
        "n_rows": len(Xs), "n_features_shown": len(feat_labels),
    }


# ---------------------------------------------------------------------------
# SHAP for SVR (KernelExplainer)
# ---------------------------------------------------------------------------

def shap_for_sklearn_pipeline_svr(
    model_pipeline, X_train, X_test, out_dir, model_name,
    dataset_label="test", sample_rows=100, background_rows=50,
    random_state=42, max_display=25, exclude_categorical_cols=None,
    waterfall_row=0, waterfall_max_display=20,
    shap_stability=False, stability_B=2000, stability_topk=10,
    stability_rank_check=True, **kwargs,
):
    """
    KernelExplainer SHAP for SVR. Slow — subsamples test set by default.
    """
    import shap

    if exclude_categorical_cols is None:
        exclude_categorical_cols = DEFAULT_GROUP_COLS

    pre = model_pipeline.named_steps["pre"]
    est = model_pipeline.named_steps["model"]

    Xtr_t = pre.transform(X_train)
    if sparse.issparse(Xtr_t): Xtr_t = Xtr_t.toarray()
    Xtr_t = np.asarray(Xtr_t, dtype=float)

    rng  = np.random.default_rng(int(random_state))
    n_bg = min(int(background_rows), Xtr_t.shape[0])
    background = Xtr_t[rng.choice(Xtr_t.shape[0], size=n_bg, replace=False)]

    Xs = _sample_rows_df(
        X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test),
        sample_rows, random_state,
    )
    Xt = pre.transform(Xs)
    if sparse.issparse(Xt): Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)

    feature_names = _get_feature_names_from_pipeline(pre)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(Xt.shape[1])]

    print(f"  [SHAP-SVR] KernelExplainer: {len(Xt)} test rows, {n_bg} background rows ...")
    explainer  = shap.KernelExplainer(est.predict, background)
    shap_vals  = np.asarray(explainer.shap_values(Xt, nsamples=200), dtype=float)
    base_value = float(explainer.expected_value)

    _ensure_dir(out_dir)
    tag = f"{model_name}_{dataset_label}".replace(os.sep, "_")

    feat_labels, sv_plot, _ = _filter_features(
        feature_names, shap_vals, Xt, exclude_categorical_cols
    )
    X_for_colour = _build_colour_matrix(Xs, feat_labels)

    _save_mean_abs_csv(
        sv_plot, feat_labels,
        os.path.join(out_dir, f"shap_meanabs_{tag}.csv"),
    )

    if shap_stability:
        _write_shap_stability_outputs(
            sv_plot, feat_labels, out_dir, tag,
            stability_B, stability_topk, random_state, stability_rank_check,
        )

    _save_beeswarm_and_bar(sv_plot, X_for_colour, feat_labels, out_dir, tag, max_display)

    if waterfall_row is not None and len(sv_plot) > 0:
        r = max(0, min(int(waterfall_row), len(sv_plot) - 1))
        _, sv_f, Xt_f = _filter_features(
            feature_names, shap_vals, Xt, exclude_categorical_cols
        )
        _save_waterfall(
            base_value, sv_f[r], Xt_f[r], feat_labels,
            os.path.join(out_dir, f"shap_waterfall_{tag}_row{r}.png"),
            waterfall_max_display,
        )

    return {
        "model": model_name, "dataset": dataset_label,
        "n_rows": len(Xs), "n_features_shown": len(feat_labels),
    }


# ---------------------------------------------------------------------------
# SHAP for LightGBM native-cat wrappers
# ---------------------------------------------------------------------------

def shap_for_lightgbm_wrapper(
    lgbm_wrapper, X, out_dir, model_name,
    dataset_label="test", sample_rows=0, random_state=42,
    max_display=25, note_log_space=False,
    exclude_categorical_cols=None,
    waterfall_row=0, waterfall_max_display=20,
    shap_stability=True, stability_B=2000, stability_topk=10,
    stability_rank_check=True, **kwargs,
):
    """
    TreeExplainer SHAP for LightGBM native-categorical wrappers.
    """
    import shap

    if exclude_categorical_cols is None:
        exclude_categorical_cols = DEFAULT_GROUP_COLS

    Xs  = _sample_rows_df(
        X if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
        sample_rows, random_state,
    )
    est = lgbm_wrapper.model_

    explainer  = shap.TreeExplainer(est)
    shap_vals  = explainer.shap_values(Xs)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals  = np.asarray(shap_vals, dtype=float)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])

    _ensure_dir(out_dir)
    tag = f"{model_name}_{dataset_label}".replace(os.sep, "_")

    if note_log_space:
        with open(os.path.join(out_dir, f"shap_NOTE_{tag}.txt"), "w") as f:
            f.write("NOTE: SHAP values are in log1p(y) space. Predictions use expm1().\n")

    feature_names = list(Xs.columns)
    exclude_set   = set(exclude_categorical_cols)
    keep_cols     = [
        c for c in feature_names
        if c not in exclude_set and not _is_flag_feature(c)
    ]
    keep_idx      = [feature_names.index(c) for c in keep_cols]
    feat_labels   = keep_cols
    sv_plot       = shap_vals[:, keep_idx]

    X_for_colour = _build_colour_matrix(Xs, feat_labels)
    for c in X_for_colour.columns:
        if (str(X_for_colour[c].dtype) == "category"
                or X_for_colour[c].dtype == object):
            X_for_colour[c] = pd.to_numeric(X_for_colour[c], errors="coerce")

    _save_mean_abs_csv(
        sv_plot, feat_labels,
        os.path.join(out_dir, f"shap_meanabs_{tag}.csv"),
    )

    if shap_stability:
        _write_shap_stability_outputs(
            sv_plot, feat_labels, out_dir, tag,
            stability_B, stability_topk, random_state, stability_rank_check,
        )

    _save_beeswarm_and_bar(sv_plot, X_for_colour, feat_labels, out_dir, tag, max_display)

    if waterfall_row is not None and len(sv_plot) > 0:
        r     = max(0, min(int(waterfall_row), len(sv_plot) - 1))
        X_row = Xs[keep_cols].iloc[r].values
        _save_waterfall(
            base_value, sv_plot[r], X_row, feat_labels,
            os.path.join(out_dir, f"shap_waterfall_{tag}_row{r}.png"),
            waterfall_max_display,
        )

    return {
        "model": model_name, "dataset": dataset_label,
        "n_rows": len(Xs), "n_features_shown": len(feat_labels),
    }


# ---------------------------------------------------------------------------
# Helper: build colour matrix from original X columns
# ---------------------------------------------------------------------------

def _build_colour_matrix(X_original, feat_labels):
    """
    Builds a DataFrame of raw feature values for the SHAP beeswarm colour axis.
    For each cleaned feature label, looks up the matching column in X_original.
    NaN values are preserved so SHAP renders them as grey dots.
    If a feature label does not match any column in X_original, fills with NaN.
    """
    colour_data = {}
    for label in feat_labels:
        if label in X_original.columns:
            colour_data[label] = pd.to_numeric(
                X_original[label], errors="coerce"
            ).values
        else:
            colour_data[label] = np.full(len(X_original), np.nan)
    return pd.DataFrame(
        colour_data,
        index=X_original.index if hasattr(X_original, "index") else None,
    )