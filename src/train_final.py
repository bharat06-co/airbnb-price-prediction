"""Train and save the final models used by the demo app.

* One XGBoost model per city on all features (C+O+T) with the log label -
  the best setting from our experiments.
* One "unseen city" model trained on NYC + Paris only, with city-agnostic
  features and target-city price-level calibration (our transfer-learning
  extension). It never trains on Berlin listings; Berlin's price level is the
  mean log price of the 10% labelled Berlin sample (same as Table 8).

The nominal 80% residual band is estimated from *validation* residuals. The test
split is only used to report coverage; notably the cross-city transfer band is
not calibrated for Berlin (61.4% empirical test coverage).

Usage:
    python -m src.train_final

Owner: Bharath Hegde
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config as cfg
from .data import labelled_sample, load_city, price_level, split
from .evaluate import (dollar_scores, interval_coverage, interval_from_residuals,
                       inverse_label, scores, transform_label)
from .features import build_preprocessor, feature_names
from .models import fit_model, make_model, predict

GROUPS = "C+O+T"
LABEL = "log"


def _slim(prep):
    """Drop fitted attributes that are only needed for inspection (smaller files)."""
    for name, trans, _ in prep.transformers_:
        if name.startswith("T_"):
            vec = trans.named_steps["vectorize"]
            if hasattr(vec, "stop_words_"):
                vec.stop_words_ = None
    return prep


def _template(train: pd.DataFrame) -> dict:
    """A 'typical listing' for the city: medians / most frequent values."""
    row = {}
    for col in train.columns:
        if col in ("id", cfg.GROUP_COLUMN, "price_usd", "city"):
            continue
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            row[col] = float(s.median()) if s.notna().any() else np.nan
        elif col in cfg.TEXT:
            row[col] = ""
        elif col in cfg.LIST_FEATURES:
            row[col] = "[]"
        else:
            row[col] = str(s.mode().iat[0])
    return row


def _city_metadata(train: pd.DataFrame, prep) -> dict:
    nb = (train.groupby("neighbourhood_cleansed")
          .agg(lat=("latitude", "median"), lon=("longitude", "median"),
               group=("neighbourhood_group_cleansed", lambda s: s.mode().iat[0]),
               n=("price_usd", "size"))
          .query("n >= 5").sort_index())
    typical = (train.groupby(["neighbourhood_cleansed", "room_type"])["price_usd"]
               .agg(["median", "size"]).query("size >= 5")["median"].round(0))
    amen = None
    for name, trans, _ in prep.transformers_:
        if name == "L_amenities":
            amen = trans.vocabulary_
    return {
        "neighbourhoods": {str(k): {"lat": float(v["lat"]), "lon": float(v["lon"]),
                                    "group": str(v["group"])}
                           for k, v in nb[["lat", "lon", "group"]].to_dict("index").items()},
        "typical_price": {f"{a}|{b}": float(v) for (a, b), v in typical.items()},
        "room_types": [str(x) for x in train["room_type"].value_counts().index],
        "property_types": [str(x) for x in train["property_type"].value_counts().index[:25]],
        "amenities": [str(a) for a in (amen or [])],
        "template": _template(train),
    }


def _evaluation(y_va, pred_va, y_te, pred_te, price_te) -> dict:
    """Point metrics on test + an 80% interval fitted on validation, checked on test."""
    interval = interval_from_residuals(y_va, pred_va)
    s, d = scores(y_te, pred_te), dollar_scores(price_te, pred_te, LABEL)
    metrics = {"test_r2_log": s["r2"], "test_mse_log": s["mse"],
               "mae_usd": d["mae_usd"], "r2_usd": d["r2_usd"],
               "interval_coverage_test": interval_coverage(y_te, pred_te, interval)}
    sample = pd.DataFrame({"true": np.asarray(price_te, float),
                           "pred": inverse_label(pred_te, LABEL)})
    return dict(interval=interval, metrics=metrics,
                test_sample=sample.sample(min(1500, len(sample)), random_state=0))


def train_city(city: str) -> dict:
    t0 = time.time()
    train, val, test = split(load_city(city))
    prep = build_preprocessor(GROUPS)
    X_tr = prep.fit_transform(train)
    X_va, X_te = prep.transform(val), prep.transform(test)
    y_tr, y_va, y_te = (transform_label(d.price_usd, LABEL) for d in (train, val, test))
    model = fit_model(make_model("xgb"), "xgb", X_tr, y_tr, X_va, y_va)
    ev = _evaluation(y_va, predict(model, X_va), y_te, predict(model, X_te),
                     test.price_usd.values)
    names = feature_names(prep)
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    bundle = {
        "city": city, "city_name": cfg.CITIES[city]["name"], "kind": "in-city",
        "pipeline": Pipeline([("features", _slim(prep)), ("model", model)]),
        "label": LABEL, "groups": GROUPS,
        "metrics": {**ev["metrics"], "n_train": len(train), "n_test": len(test)},
        # 10th / 90th percentile of *validation* log-residuals -> 80% interval
        "interval": ev["interval"],
        "interval_method": "10th/90th quantiles of validation log-residuals; nominal 80%",
        "test_sample": ev["test_sample"],
        "importance": {str(k): float(v) for k, v in imp.head(20).items()},
        **_city_metadata(train, prep),
    }
    out = cfg.MODELS_DIR / f"{city}_xgb.joblib"
    joblib.dump(bundle, out, compress=3)
    m = ev["metrics"]
    print(f"{city:>6}: test R2(log)={m['test_r2_log']:.3f}  R2($)={m['r2_usd']:.3f}  "
          f"MAE=${m['mae_usd']:.0f}  80% interval coverage={m['interval_coverage_test']:.1%}  "
          f"-> {out.name} ({time.time() - t0:.0f}s)", flush=True)
    return {**bundle["metrics"], "interval_nominal_coverage": 0.8,
            "interval_method": "In-city validation log-residual 10th and 90th percentiles"}


def train_transfer() -> dict:
    """NYC + Paris -> Berlin, using city-agnostic features + target-city price level.

    Berlin information used: only the mean log price of the 10% labelled Berlin
    sample (no Berlin listing is used for training, early stopping or the interval).
    """
    t0 = time.time()
    sp = {c: split(load_city(c)) for c in cfg.CITIES}
    level = {c: price_level(s[0]) for c, s in sp.items()}
    level["berlin"] = price_level(labelled_sample(sp["berlin"][0]))
    train = pd.concat([sp["nyc"][0], sp["paris"][0]], ignore_index=True)
    val = pd.concat([sp["nyc"][1], sp["paris"][1]], ignore_index=True)
    test = sp["berlin"][2]
    prep = build_preprocessor("C+O", city_agnostic=True)
    X_tr = prep.fit_transform(train)
    X_va, X_te = prep.transform(val), prep.transform(test)
    off = lambda d: d.city.map(level).values  # noqa: E731
    y_tr, y_va, y_te = (np.log(d.price_usd.values) for d in (train, val, test))
    model = fit_model(make_model("xgb"), "xgb", X_tr, y_tr - off(train), X_va, y_va - off(val))
    # Band from NYC + Paris validation residuals: NOT calibrated for Berlin.
    ev = _evaluation(y_va, predict(model, X_va) + off(val), y_te,
                     predict(model, X_te) + off(test), test.price_usd.values)
    names = feature_names(prep)
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    bundle = {
        "city": "transfer", "city_name": "NYC + Paris → Berlin (transfer)",
        "kind": "transfer", "trained_on": ["nyc", "paris"],
        "pipeline": Pipeline([("features", prep), ("model", model)]),
        "label": "log", "city_level": level,
        "berlin_info": f"mean log price of a {cfg.TARGET_LABELLED_FRACTION:.0%} labelled "
                       "Berlin sample (one number)",
        "metrics": {**ev["metrics"], "evaluated_on": "Berlin test split",
                    "interval_warning": "Source-city residual band is not Berlin-calibrated"},
        "interval": ev["interval"],
        "interval_method": "NYC + Paris validation log-residuals; nominal 80%, uncalibrated for Berlin",
        "test_sample": ev["test_sample"],
        "importance": {str(k): float(v) for k, v in imp.head(20).items()},
    }
    out = cfg.MODELS_DIR / "transfer_xgb.joblib"
    joblib.dump(bundle, out, compress=3)
    m = ev["metrics"]
    print(f"transfer (NYC+Paris -> Berlin): R2(log)={m['test_r2_log']:.3f} "
          f"MAE=${m['mae_usd']:.0f} 80% interval coverage={m['interval_coverage_test']:.1%} "
          f"-> {out.name} ({time.time() - t0:.0f}s)")
    return {**bundle["metrics"], "interval_nominal_coverage": 0.8,
            "interval_method": "Source-city (NYC + Paris) validation log-residual quantiles; not calibrated for Berlin",
            "interval_warning": "Observed Berlin test coverage is below nominal 80%; transfer band is not Berlin-calibrated"}


def main():
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {c: train_city(c) for c in cfg.CITIES}
    summary["transfer"] = train_transfer()
    (cfg.MODELS_DIR / "model_card.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
