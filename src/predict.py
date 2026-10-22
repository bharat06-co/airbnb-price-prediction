"""Predict the nightly price of a single listing with a saved model.

Used by the Streamlit app, and usable from the command line:
    python -m src.predict --city paris --room-type "Entire home/apt" \
        --accommodates 4 --bedrooms 2 --neighbourhood "Louvre"
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data import haversine_km


@lru_cache(maxsize=None)
def load_bundle(name: str) -> dict:
    path = cfg.MODELS_DIR / f"{name}_xgb.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.train_final` first.")
    return joblib.load(path)


def make_listing(city: str, **inputs) -> pd.DataFrame:
    """Start from the city's typical listing and overwrite user inputs."""
    b = load_bundle(city)
    row = dict(b["template"])
    nb = inputs.get("neighbourhood_cleansed")
    if nb in b["neighbourhoods"]:
        info = b["neighbourhoods"][nb]
        row.update(latitude=info["lat"], longitude=info["lon"],
                   neighbourhood_group_cleansed=info["group"])
    row.update({k: v for k, v in inputs.items() if v is not None})
    amen = inputs.get("amenities")
    if isinstance(amen, (list, tuple)):
        row["amenities"] = json.dumps(list(amen))
        row["amenities_count"] = len(amen)
    row["description_length"] = len(row.get("description") or "")
    lat0, lon0 = cfg.CITIES[city]["center"]
    row["distance_to_center_km"] = float(haversine_km(row["latitude"], row["longitude"], lat0, lon0))
    row["city"] = city
    return pd.DataFrame([row])


def predict_price(city: str, listing: pd.DataFrame, use_transfer: bool = False) -> dict:
    """Return price and a validation-residual band in USD (not always 80% calibrated)."""
    if use_transfer:
        if city != "berlin":
            raise ValueError("The cross-city transfer demo is evaluated for Berlin only.")
        b = load_bundle("transfer")
        offset = b["city_level"][city]
    else:
        b = load_bundle(city)
        offset = 0.0
    log_pred = float(b["pipeline"].predict(listing)[0]) + offset
    lo, hi = b["interval"]
    return {"price": float(np.exp(log_pred)), "low": float(np.exp(log_pred + lo)),
            "high": float(np.exp(log_pred + hi)), "model": b["kind"],
            "test_band_coverage": b["metrics"].get("interval_coverage_test") }


def main():
    ap = argparse.ArgumentParser(description="Predict an Airbnb nightly price")
    ap.add_argument("--city", choices=list(cfg.CITIES), default="nyc")
    ap.add_argument("--room-type", default=None)
    ap.add_argument("--property-type", default=None)
    ap.add_argument("--neighbourhood", default=None)
    ap.add_argument("--accommodates", type=int, default=None)
    ap.add_argument("--bedrooms", type=float, default=None)
    ap.add_argument("--beds", type=float, default=None)
    ap.add_argument("--bathrooms", type=float, default=None)
    ap.add_argument("--transfer", action="store_true",
                    help="requires --city berlin; uses the NYC + Paris model (no Berlin training listings; Berlin's "
                         "price level comes from a 10%% labelled Berlin sample)")
    a = ap.parse_args()
    listing = make_listing(a.city, room_type=a.room_type, property_type=a.property_type,
                           neighbourhood_cleansed=a.neighbourhood, accommodates=a.accommodates,
                           bedrooms=a.bedrooms, beds=a.beds, bathrooms=a.bathrooms)
    out = predict_price(a.city, listing, use_transfer=a.transfer)
    band = ("UNCALIBRATED transfer band" if a.transfer else "nominal 80% residual band")
    print(f"Estimated nightly price: ${out['price']:.0f} "
          f"({band}: ${out['low']:.0f} - ${out['high']:.0f}) [{out['model']} model]")
    if a.transfer:
        print("WARNING: This source-city band covered only 61.4% of held-out Berlin "
              "prices; it is not an 80% calibrated Berlin interval.")


if __name__ == "__main__":
    main()
