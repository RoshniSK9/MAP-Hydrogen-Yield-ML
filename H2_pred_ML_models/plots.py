# plots.py
# Produces parity plots (actual vs predicted) styled for IJHE publication.
#
# Each model gets:
#   - Individual train and test parity plots (full range and zoomed)
#   - A combined side-by-side (train | test) plot
#
# All plots use:
#   - Times New Roman serif font
#   - Bold axis labels and tick labels
#   - Dashed 1:1 reference line
#   - R2 and 95% CI annotated in upper-left corner
#   - Values formatted to 2 significant figures

import os
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .metrics import bootstrap_r2

# ---------------------------------------------------------------------------
# Global IJHE rcParams — applied once at import time
# ---------------------------------------------------------------------------
_IJHE_RC = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "axes.labelweight": "bold",
    "font.weight":      "bold",
    "axes.linewidth":       0.75,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.minor.size":     2.0,
    "ytick.minor.size":     2.0,
    "lines.linewidth":      1.0,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.bottom": True,
    "axes.spines.left":   True,
    "axes.grid": False,
    "figure.dpi":         100,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}
mpl.rcParams.update(_IJHE_RC)

# Colour scheme: blue for training, red for test
_COL_TRAIN = "#2166AC"
_COL_TEST  = "#D6604D"

# Figure dimensions (inches)
_SINGLE_COL_W = 4.5    # single-panel plot width
_DOUBLE_COL_W = 9.0    # two-panel combined plot width
_ASPECT       = 1.0    # height = width * aspect ratio

# Axis range for full-range plots (0 to 100 mmol/g covers all values in current dataset)
_FIXED_MAX = 100

# Axis labels using LaTeX formatting for the units
_XLABEL = (
    r"Actual H$_2$ yield (mmol H$_2$/g$_\mathrm{feedstock}$)"
)
_YLABEL = (
    r"Predicted H$_2$ yield (mmol H$_2$/g$_\mathrm{feedstock}$)"
)


def safe_slug(s: str, maxlen: int = 180) -> str:
    """
    Creates a filesystem-safe version of a model name for use in filenames.
    Replaces special characters and limits length.
    """
    s = str(s)
    s = s.replace(os.sep, "_")
    s = re.sub(r"[\n\r\t]", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return s[:maxlen] if len(s) > maxlen else s


def _sig2(x: float) -> str:
    """
    Formats a float to 2 significant figures for axis annotations.
    Uses scientific notation for very small or very large numbers.
    """
    if not np.isfinite(x):
        return str(x)
    if x == 0.0:
        return "0.0"
    mag = np.floor(np.log10(abs(x)))
    if mag < -2 or mag >= 4:
        return f"{x:.1e}"
    decimals = max(0, int(1 - mag))
    return f"{x:.{decimals}f}"


def _axis_limits(y_true, y_pred, zoom=None):
    """
    Computes axis limits for a parity plot.
    If zoom=(xmin, xmax, ymin, ymax) is provided, uses those limits.
    Otherwise computes limits from the data with a small padding.
    """
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        return xmin, xmax, ymin, ymax
    lo  = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
    hi  = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad, lo - pad, hi + pad


def _apply_box_ticks(ax):
    """
    Applies the IJHE box-style formatting to an axes object:
      - Minor ticks on all four sides
      - All four spines visible (full box)
      - Ticks pointing inward on bottom and left only
      - Bold tick labels
    """
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # Show all four spines for the box effect
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    # Only bottom and left have actual tick marks
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(bottom=True, left=True, direction="in", which="both")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def _annotate_metrics(ax, y_true, y_pred, y_train=None, B=2000, ci=0.95,
                      random_state=42, show_ci=True):
    """
    Annotates the upper-left corner with R2 and optionally 95% CI.
    For the training panel, show_ci=False shows only the point estimate.
    For the test panel, show_ci=True computes bootstrap CI and displays it.
    """
    from .metrics import r2 as _r2
    ci_pct = int(ci * 100)

    if show_ci:
        stats      = bootstrap_r2(
            y_true, y_pred,
            B=B, ci=ci, random_state=random_state,
        )
        r2_val     = stats["R2"]
        r2_lo      = stats[f"R2_ci{ci_pct}_low"]
        r2_hi      = stats[f"R2_ci{ci_pct}_high"]
        annotation = (
            f"R² = {_sig2(r2_val)}\n"
            f"{ci_pct}% CI [{_sig2(r2_lo)}, {_sig2(r2_hi)}]"
        )
    else:
        # Training panel: point estimate only, no bootstrap
        r2_val     = _r2(y_true, y_pred)
        annotation = f"R² = {_sig2(r2_val)}"

    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        fontweight="bold",
    )


def _draw_reference_lines(ax, lo_line, hi_line):
    """
    Draws the 1:1 reference line (dashed black).
    CI band removed — R2 and CI are already annotated as text in the corner.
    """
    x = np.array([lo_line, hi_line])
    # Draw the 1:1 perfect prediction line
    ax.plot(x, x, color="black", linewidth=0.75, linestyle="--", zorder=2)


def parity_plot(
    y_true, y_pred, fname, y_train=None,
    zoom=None, color="black",
    B=2000, ci=0.95, random_state=42, show_ci=True,
):
    """
    Saves a single parity plot (actual vs predicted) to fname.

    Parameters
    ----------
    y_true     : actual target values
    y_pred     : predicted target values
    fname      : output file path
    y_train    : unused — kept for backward compatibility only
    zoom       : (xmin, xmax, ymin, ymax) axis limits, or None to auto-compute
    color      : scatter point colour
    B          : bootstrap iterations for CI (only used if show_ci=True)
    ci         : confidence interval level
    random_state : random seed
    show_ci    : if True, compute and display bootstrap CI (test plots only)
                 if False, display point estimate R2 only (training plots)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    # denominator uses std(y_true) of evaluated set, consistent with R2

    xmin, xmax, ymin, ymax = _axis_limits(y_true, y_pred, zoom)

    fig, ax = plt.subplots(figsize=(_SINGLE_COL_W, _SINGLE_COL_W * _ASPECT))
    fig.subplots_adjust(left=0.22, bottom=0.18, right=0.97, top=0.97)

    lo_line = min(xmin, ymin)
    hi_line = max(xmax, ymax)

    _draw_reference_lines(ax, lo_line, hi_line)

    ax.scatter(y_true, y_pred, s=18, color=color, edgecolors="none",
               alpha=0.80, zorder=3, linewidths=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_title("")

    _apply_box_ticks(ax)
    _annotate_metrics(ax, y_true, y_pred, y_train=None, B=B, ci=ci,
                      random_state=random_state, show_ci=show_ci)

    fig.savefig(fname)
    plt.close(fig)


def parity_plot_combined(
    y_train, pred_tr, y_test, pred_te, fname,
    zoom_train=None, zoom_test=None,
    B=2000, ci=0.95, random_state=42,
):
    """
    Saves a two-panel parity plot: (a) training set | (b) test set.
    Both panels use the same style as parity_plot() above.
    """
    y_train = np.asarray(y_train, dtype=float).ravel()
    pred_tr = np.asarray(pred_tr, dtype=float).ravel()
    y_test  = np.asarray(y_test,  dtype=float).ravel()
    pred_te = np.asarray(pred_te, dtype=float).ravel()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(_DOUBLE_COL_W, _DOUBLE_COL_W / 2),
        constrained_layout=True,
    )

    # Each panel: (axes, y_true, y_pred, zoom, colour, is_test)
    datasets = [
        (axes[0], y_train, pred_tr, zoom_train, _COL_TRAIN, False),
        (axes[1], y_test,  pred_te, zoom_test,  _COL_TEST,  True),
    ]

    for ax, yt, yp, zoom, col, is_test in datasets:
        yt = np.asarray(yt, dtype=float).ravel()
        yp = np.asarray(yp, dtype=float).ravel()

        xmin, xmax, ymin, ymax = _axis_limits(yt, yp, zoom)
        lo_line = min(xmin, ymin)
        hi_line = max(xmax, ymax)

        _draw_reference_lines(ax, lo_line, hi_line)

        ax.scatter(yt, yp, s=18, color=col, edgecolors="none",
                   alpha=0.80, zorder=3, linewidths=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(_XLABEL)
        ax.set_ylabel(_YLABEL)
        ax.set_title("")

        _apply_box_ticks(ax)
        # Test panel: show CI; training panel: point estimate only
        _annotate_metrics(ax, yt, yp, y_train=None, B=B, ci=ci,
                          random_state=random_state, show_ci=is_test)

    fig.savefig(fname)
    plt.close(fig)


def save_train_test_parity(
    y_train, pred_tr,
    y_test,  pred_te,
    out_dir, model_name,
    zoom_max=80,
    B=2000, ci=0.95, random_state=42,
):
    """
    Saves all six parity plot variants for a single model:
      1. train_full.png    — training set, full range (0 to 500)
      2. train_zoom.png    — training set, zoomed to (0 to zoom_max)
      3. test_full.png     — test set, full range
      4. test_zoom.png     — test set, zoomed
      5. combined_full.png — side-by-side full range
      6. combined_zoom.png — side-by-side zoomed

    Parameters
    ----------
    y_train    : actual training target values
    pred_tr    : predicted training values
    y_test     : actual test target values
    pred_te    : predicted test values
    out_dir    : directory to save plots
    model_name : used to construct filenames
    zoom_max   : upper limit for the zoomed plots (default 80 mmol/g)
    """
    os.makedirs(out_dir, exist_ok=True)
    slug    = safe_slug(model_name)
    zm      = (0, zoom_max, 0, zoom_max)
    full_zm = (0, _FIXED_MAX, 0, _FIXED_MAX)

    # Training set plots — point estimate only, no bootstrap CI
    parity_plot(
        y_train, pred_tr,
        fname=os.path.join(out_dir, f"{slug}_train_full.png"),
        zoom=full_zm, color=_COL_TRAIN,
        B=B, ci=ci, random_state=random_state, show_ci=False,
    )
    parity_plot(
        y_train, pred_tr,
        fname=os.path.join(out_dir, f"{slug}_train_zoom.png"),
        zoom=zm, color=_COL_TRAIN,
        B=B, ci=ci, random_state=random_state, show_ci=False,
    )

    # Test set plots — full bootstrap CI
    parity_plot(
        y_test, pred_te,
        fname=os.path.join(out_dir, f"{slug}_test_full.png"),
        zoom=full_zm, color=_COL_TEST,
        B=B, ci=ci, random_state=random_state, show_ci=True,
    )
    parity_plot(
        y_test, pred_te,
        fname=os.path.join(out_dir, f"{slug}_test_zoom.png"),
        zoom=zm, color=_COL_TEST,
        B=B, ci=ci, random_state=random_state, show_ci=True,
    )

    # Combined side-by-side plots
    parity_plot_combined(
        y_train, pred_tr, y_test, pred_te,
        fname=os.path.join(out_dir, f"{slug}_combined_full.png"),
        zoom_train=full_zm, zoom_test=full_zm,
        B=B, ci=ci, random_state=random_state,
    )
    parity_plot_combined(
        y_train, pred_tr, y_test, pred_te,
        fname=os.path.join(out_dir, f"{slug}_combined_zoom.png"),
        zoom_train=zm, zoom_test=zm,
        B=B, ci=ci, random_state=random_state,
    )