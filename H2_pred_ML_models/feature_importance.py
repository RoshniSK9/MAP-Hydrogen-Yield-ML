# feature_importance.py
# Computes and saves XGBoost built-in feature importance (gain, weight, cover).
# Called automatically by run_all() after SHAP analysis for XGBoost.

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _is_flag_or_cat(name):
    s = str(name)
    for prefix in ["num_base__", "num_flags__", "num__", "cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("__exists") or s.endswith("__is_NR") or s.endswith("__is_NA"):
        return True
    if "Catalyst" in s or "absorber" in s.lower() or "Microwave absorber" in s:
        return True
    return False


def _clean_label(name):
    for prefix in ["num_base__", "num_flags__", "num__", "cat__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


_LABEL_MAP = {
    "Feedstock: Catalyst ratio (wt/wt)":   "Feedstock to catalyst ratio (wt/wt)",
    "Feedstock:Absorber ratio (wt/wt)":    "Feedstock to absorber ratio (wt/wt)",
    "Dielectric loss of absorber (tan d)": "Dielectric loss tangent of absorber (tan d)",
    "Reaction time (min)":                 "Isothermal time (min)",
}


def _format_label(name):
    name = _clean_label(name)
    return _LABEL_MAP.get(name, name)


def _apply_box_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", width=1.2)


def save_xgb_feature_importance(
    model_pipeline,
    out_dir,
    model_name="XGBoost",
    top_n=15,
    shap_csv_path=None,
):
    _ensure_dir(out_dir)

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size":   11,
        "axes.linewidth": 1.2,
    })

    try:
        ct = model_pipeline.named_steps["pre"].named_steps["ct"]
        raw_names = ct.get_feature_names_out()
    except Exception as e:
        print(f"[WARN] Could not get feature names: {e}")
        return

    try:
        booster = model_pipeline.named_steps["model"].get_booster()
    except Exception as e:
        print(f"[WARN] Could not get XGBoost booster: {e}")
        return

    def _get_importance(importance_type):
        imp = booster.get_score(importance_type=importance_type)
        if not imp:
            return pd.Series(dtype=float)
        df = pd.Series(imp)
        df.index = [raw_names[int(i[1:])] if i.startswith("f") and i[1:].isdigit()
                    else i for i in df.index]
        df = df[~pd.Index([_is_flag_or_cat(n) for n in df.index])]
        df.index = [_format_label(n) for n in df.index]
        if df.sum() > 0:
            df = df / df.sum()
        return df.sort_values(ascending=False)

    gain   = _get_importance("gain")
    weight = _get_importance("weight")
    cover  = _get_importance("cover")

    if gain.empty:
        print(f"[WARN] No feature importance data available for {model_name}.")
        return

    df_all = pd.DataFrame({
        "Gain (normalised)":   gain,
        "Weight (normalised)": weight,
        "Cover (normalised)":  cover,
    }).fillna(0).sort_values("Gain (normalised)", ascending=False)

    csv_path = os.path.join(out_dir, f"xgb_feature_importance_{model_name}.csv")
    df_all.to_csv(csv_path)
    print(f"  Feature importance CSV saved: {csv_path}")

    top_gain = gain.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_gain) + 1.5))
    ax.barh(top_gain.index, top_gain.values,
            color="#2E75B6", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Normalised Gain", fontsize=12, fontweight="bold")
    ax.set_title(f"XGBoost Feature Importance (Gain) — Top {top_n}",
                 fontsize=12, fontweight="bold")
    _apply_box_style(ax)
    plt.tight_layout()
    gain_path = os.path.join(out_dir, f"xgb_feature_importance_gain_{model_name}.png")
    fig.savefig(gain_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved: {gain_path}")

    if shap_csv_path and os.path.exists(shap_csv_path):
        try:
            df_shap = pd.read_csv(shap_csv_path)
            feat_col = df_shap.columns[0]
            val_col  = df_shap.columns[1]
            df_shap  = df_shap.set_index(feat_col)[val_col]
            df_shap.index = df_shap.index.str.strip()
            df_shap.index = [_LABEL_MAP.get(i, i) for i in df_shap.index]
            if df_shap.sum() > 0:
                df_shap = df_shap / df_shap.sum()
            common = gain.index.intersection(df_shap.index)
            if len(common) == 0:
                print("[WARN] No common features between SHAP and XGBoost importance.")
                return
            df_compare = pd.DataFrame({
                "SHAP":         df_shap.loc[common],
                "XGBoost Gain": gain.loc[common],
            }).sort_values("SHAP", ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(df_compare))
            width = 0.38
            ax.bar(x - width/2, df_compare["SHAP"],
                   width=width, label="SHAP", color="#2E75B6",
                   edgecolor="black", linewidth=0.5)
            ax.bar(x + width/2, df_compare["XGBoost Gain"],
                   width=width, label="XGBoost Gain", color="#ED7D31",
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(df_compare.index, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Normalised Importance", fontsize=12, fontweight="bold")
            ax.set_title(f"SHAP vs XGBoost Feature Importance — {model_name}",
                         fontsize=12, fontweight="bold")
            ax.legend(frameon=False, fontsize=11)
            _apply_box_style(ax)
            plt.tight_layout()
            comp_path = os.path.join(out_dir, f"shap_vs_xgb_importance_{model_name}.png")
            fig.savefig(comp_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Comparison plot saved: {comp_path}")
        except Exception as e:
            print(f"[WARN] Comparison plot failed: {e}")

    print(f"  Feature importance complete for {model_name}.")
