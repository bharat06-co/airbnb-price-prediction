"""Model zoo: baselines (Ridge, KNN), Random Forest, XGBoost and an MLP.

Owner: Bharath Hegde (models & experiments)
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config as cfg

MODEL_NAMES = {
    "linear": "Linear regression (L2)",
    "knn": "K-nearest neighbours",
    "rf": "Random forest",
    "mlp": "Neural network (MLP)",
    "xgb": "XGBoost",
}


def make_model(name: str, seed: int = cfg.RANDOM_STATE):
    if name == "linear":
        # h(x) = theta^T x, cost = ||y - h(x)||^2 + alpha * ||theta||^2
        return Ridge(alpha=1.0)
    if name == "knn":
        # unweighted KNN with Euclidean distance (sklearn default k = 5)
        return KNeighborsRegressor(n_neighbors=5, weights="uniform", n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "xgb":
        return XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
            reg_lambda=1.0, tree_method="hist", early_stopping_rounds=100,
            eval_metric="rmse", n_jobs=-1, random_state=seed,
        )
    if name == "mlp":
        # Paper section 6.2: 2 hidden layers (64, 32) with ReLU, MSE loss,
        # SGD + Nesterov momentum (mu = 0.9, lr = 0.005), L2 = 0.005, batch 64,
        # early stopping on 10% held-out training data when the validation
        # R^2 improves by < 1e-4 for 10 epochs.
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="sgd",
            learning_rate_init=0.005, momentum=0.9, nesterovs_momentum=True,
            alpha=0.005, batch_size=64, max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
            random_state=seed,
        )
        # Standardise the target so SGD is stable for every label transform.
        return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_NAMES)}")


def fit_model(model, name: str, X_train, y_train, X_val=None, y_val=None):
    """Fit a model. XGBoost uses the validation split for early stopping."""
    if name == "xgb" and X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "xgb":
        model.set_params(early_stopping_rounds=None, n_estimators=600)
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def predict(model, X):
    return np.asarray(model.predict(X), dtype=float)
