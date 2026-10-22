"""Label transforms and evaluation metrics (R^2 and MSE, as in the paper)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as cfg

# Predictions on the label scale are clipped to this range before converting back
# to dollars (exp(12) is about $160,000).
LOG_CLIP = (-10.0, 12.0)

LABELS = {
    "raw": "N/A (raw price)",
    "TH": f"Threshold (price <= ${cfg.THRESHOLD})",
    "sqrt": "Square root",
    "log": "Log",
}


def filter_for_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """The TH label drops listings above the price threshold."""
    if label == "TH":
        return df[df["price_usd"] <= cfg.THRESHOLD]
    return df


def transform_label(price, label: str):
    price = np.asarray(price, dtype=float)
    if label in ("raw", "TH"):
        return price
    if label == "sqrt":
        return np.sqrt(price)
    if label == "log":
        return np.log(price)
    raise ValueError(label)


def inverse_label(y, label: str):
    y = np.asarray(y, dtype=float)
    if label in ("raw", "TH"):
        return y
    if label == "sqrt":
        return np.clip(y, 0, None) ** 2
    if label == "log":
        return np.exp(np.clip(y, *LOG_CLIP))
    raise ValueError(label)


def scores(y_true, y_pred) -> dict:
    return {"mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def dollar_scores(price_true, y_pred, label: str) -> dict:
    """Metrics back on the dollar scale, so every label can be compared.

    `overflow_share` is the share of predictions that fell outside the clipping
    range. When most predictions overflow (e.g. a model that explodes on an
    unseen city), the dollar metrics are meaningless and are reported as NaN.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if label == "log":
        overflow = float(np.mean((y_pred < LOG_CLIP[0]) | (y_pred > LOG_CLIP[1])))
    elif label == "sqrt":
        overflow = float(np.mean(y_pred < 0))
    else:
        overflow = 0.0
    p = inverse_label(y_pred, label)
    if overflow > 0.5:
        return {"mae_usd": float("nan"), "r2_usd": float("nan"), "overflow_share": overflow}
    return {"mae_usd": float(mean_absolute_error(price_true, p)),
            "r2_usd": float(r2_score(price_true, p)), "overflow_share": overflow}


def bootstrap_r2(y_true, y_pred, rounds: int = cfg.BOOTSTRAP_ROUNDS,
                 seed: int = cfg.RANDOM_STATE):
    """95% bootstrap confidence interval of R² over test listings."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = [r2_score(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(rounds))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def interval_from_residuals(y_true, y_pred, q=cfg.INTERVAL_QUANTILES):
    """(low, high) offsets on the label scale from residual quantiles."""
    resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.quantile(resid, q[0])), float(np.quantile(resid, q[1]))


def interval_coverage(y_true, y_pred, interval) -> float:
    """Share of listings whose true value falls inside prediction + interval."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    lo, hi = interval
    return float(np.mean((y_true >= y_pred + lo) & (y_true <= y_pred + hi)))
