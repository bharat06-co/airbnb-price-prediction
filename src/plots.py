"""Figures for the report and slides (consistent, minimal style)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402

INK, MUTED, GRID, BLUE = "#2C2C2B", "#7D7A75", "#E6E5E3", "#2783DE"
SERIES = ["#5E9FE8", "#EAC26B", "#72BC8F", "#BF8EDA", "#DE9255"]
CITY_COLOR = {"nyc": "#2783DE", "paris": "#EAC26B", "berlin": "#72BC8F"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def price_distribution(frames: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for city, df in frames.items():
        p = df["price_usd"]
        axes[0].hist(p.clip(upper=1500), bins=60, alpha=0.55, color=CITY_COLOR[city],
                     label=cfg.CITIES[city]["name"])
        axes[1].hist(np.log(p), bins=60, alpha=0.55, color=CITY_COLOR[city])
    axes[0].set_title("Nightly price (USD, clipped at $1,500)")
    axes[1].set_title("log(price)")
    axes[0].set_xlabel("USD"); axes[1].set_xlabel("log USD")
    axes[0].legend(frameon=False)
    return _save(fig, "price_distribution.png")


def correlation_heatmap(X: np.ndarray, names):
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(np.abs(corr), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.grid(False)
    ax.set_title("|Correlation| between continuous features (NYC, before pruning)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    return _save(fig, "correlation_heatmap.png")


def pred_vs_true(panels, name, title):
    """panels: list of (subtitle, y_true, y_pred, axis_label)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (sub, yt, yp, lab) in zip(axes, panels):
        ax.scatter(yt, yp, s=4, alpha=0.35, color=BLUE, edgecolors="none")
        lo, hi = np.percentile(np.r_[yt, yp], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(sub); ax.set_xlabel(f"True {lab}"); ax.set_ylabel(f"Predicted {lab}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name)


def bar_r2(df: pd.DataFrame, label_col: str, name: str, title: str, value="test_r2",
           highlight=None, reference=None):
    """Horizontal R² bars. `highlight` / `reference` are boolean masks; by default the
    best bar is highlighted. Reference rows (e.g. an in-city upper bound) are grey."""
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 1.2))
    ax.set_axisbelow(True)
    y = np.arange(len(df))[::-1]
    vals = df[value].values
    hi = np.asarray(highlight) if highlight is not None else vals == np.nanmax(vals)
    ref = np.asarray(reference) if reference is not None else np.zeros(len(vals), bool)
    colors = ["#C9C7C3" if r else BLUE if h else "#A9CDF2" for h, r in zip(hi, ref)]
    ax.barh(y, np.clip(vals, 0, None), color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(max(v, 0) + 0.01, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(df[label_col].values)
    ax.set_xlim(0, 1); ax.set_xlabel("Test R²"); ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, name)


def feature_importance(names, importances, name="feature_importance.png", top=15):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(order))[::-1]
    ax.barh(y, np.asarray(importances)[order], color=BLUE, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("XGBoost importance (gain share)")
    ax.set_title("Top features - NYC, XGBoost, C+O+T, log label")
    ax.grid(axis="y", visible=False)
    return _save(fig, name)
