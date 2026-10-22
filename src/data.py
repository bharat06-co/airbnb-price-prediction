"""Loading, cleaning and splitting the Inside Airbnb listings.

Owner: Bharat V Bhat (data cleaning & feature extraction)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as cfg

# Raw columns we actually need (keeps memory low for the large Paris file).
_RAW_COLUMNS = set(
    [c for c in cfg.CONTINUOUS if c not in {"bathrooms", "amenities_count", "description_length"}]
    + ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
       "neighbourhood_cleansed", "neighbourhood_group_cleansed", "property_type",
       "room_type", "has_availability", "bathrooms_text", "license", "amenities",
       "first_review", "last_review", "hosts_time_as_host_years",
       "hosts_time_as_host_months", "id", "host_id", "price"]
    + cfg.TEXT
)


def raw_path(city: str):
    return cfg.DATA_DIR / f"{city}_listings.csv.gz"


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0 (Inside Airbnb uses '$' even for EUR cities)."""
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
                         errors="coerce")


def parse_bathrooms(text: pd.Series) -> pd.Series:
    """'1.5 shared baths' -> 1.5, 'Half-bath' -> 0.5, missing -> NaN."""
    t = text.fillna("").str.lower()
    num = pd.to_numeric(t.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    return num.where(~t.str.contains("half-bath"), 0.5)


def bath_type(text: pd.Series) -> np.ndarray:
    t = text.fillna("").str.lower()
    return np.select([t.str.contains("shared"), t.str.contains("private")],
                     ["shared", "private"], default="standard")


def license_status(lic: pd.Series) -> np.ndarray:
    t = lic.fillna("").str.lower()
    return np.select([t.eq(""), t.str.contains("exempt")],
                     ["none", "exempt"], default="registered")


def count_list(s: pd.Series) -> pd.Series:
    def _n(x):
        try:
            return len(json.loads(x))
        except (TypeError, ValueError):
            return 0
    return s.map(_n)


def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_TAG_RE, " ", regex=True).str.strip()


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
def clean_listings(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Turn a raw Inside Airbnb listings table into a model-ready table."""
    info = cfg.CITIES[city]
    df = df.copy()

    # ---- label -------------------------------------------------------------
    price_local = parse_price(df["price"])
    df["price_usd"] = price_local * cfg.FX_TO_USD[info["currency"]]
    df = df[df["price_usd"].between(cfg.MIN_PRICE, cfg.MAX_PRICE)].copy()

    # ---- engineered continuous features ------------------------------------
    df["bathrooms"] = parse_bathrooms(df["bathrooms_text"])
    df["amenities_count"] = count_list(df["amenities"])
    df["description_length"] = df["description"].fillna("").str.len()
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)   # no reviews -> 0
    lat0, lon0 = info["center"]
    df["distance_to_center_km"] = haversine_km(df["latitude"], df["longitude"], lat0, lon0)

    # ---- engineered categorical features -----------------------------------
    df["bath_type"] = bath_type(df["bathrooms_text"])
    df["license_status"] = license_status(df["license"])
    for col in cfg.CATEGORICAL:
        df[col] = df[col].astype("object").where(df[col].notna(), "missing").astype(str)

    # ---- date features -> days since a fixed reference ---------------------
    ref = pd.Timestamp(cfg.DATE_REFERENCE)
    for col in cfg.DATE:
        df[col] = (pd.to_datetime(df[col], errors="coerce") - ref).dt.days
    df["host_tenure_months"] = (df["hosts_time_as_host_years"].fillna(0) * 12
                                + df["hosts_time_as_host_months"].fillna(0))
    df.loc[df["hosts_time_as_host_years"].isna(), "host_tenure_months"] = np.nan

    # ---- text --------------------------------------------------------------
    for col in cfg.TEXT:
        df[col] = clean_text(df[col])

    df["amenities"] = df["amenities"].fillna("[]")
    df["city"] = city
    if cfg.GROUP_COLUMN not in df.columns:      # e.g. synthetic test data
        df[cfg.GROUP_COLUMN] = df["id"]
    keep = (["id", cfg.GROUP_COLUMN, "city", "price_usd"] + cfg.CONTINUOUS + cfg.CITY_AGNOSTIC_EXTRA
            + cfg.CATEGORICAL + cfg.LIST_FEATURES + cfg.TEXT + cfg.DATE + cfg.DATE_EXTRA)
    out = df[list(dict.fromkeys(keep))].reset_index(drop=True)
    # Plain Python/NumPy object strings (not pandas' Arrow-backed strings), so the
    # saved model bundles do not need pyarrow to be unpickled.
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(object)
    return out


def load_city(city: str) -> pd.DataFrame:
    path = raw_path(city)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first.")
    raw = pd.read_csv(path, usecols=lambda c: c in _RAW_COLUMNS, low_memory=False)
    return clean_listings(raw, city)


def split(df: pd.DataFrame, seed: int = cfg.RANDOM_STATE, group_col: str | None = None):
    """70 / 20 / 10 train / validation / test split (as in the paper).

    group_col: if given (e.g. "host_id"), all rows with the same value end up in
    the same split, so near-duplicate listings of one host cannot be in both
    train and test.
    """
    tr, va, te = cfg.SPLIT
    if group_col is None:
        train, rest = train_test_split(df, train_size=tr, random_state=seed)
        val, test = train_test_split(rest, train_size=va / (va + te), random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, train_size=tr, random_state=seed)
        i_tr, i_rest = next(gss.split(df, groups=df[group_col]))
        train, rest = df.iloc[i_tr], df.iloc[i_rest]
        gss = GroupShuffleSplit(n_splits=1, train_size=va / (va + te), random_state=seed)
        i_va, i_te = next(gss.split(rest, groups=rest[group_col]))
        val, test = rest.iloc[i_va], rest.iloc[i_te]
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def labelled_sample(train: pd.DataFrame, frac: float = cfg.TARGET_LABELLED_FRACTION,
                    seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """The small labelled target-city sample used by the transfer experiments.

    It is used for few-shot training *and* for estimating the target city's price
    level, so both approaches see exactly the same Berlin labels.
    """
    return train.sample(frac=frac, random_state=seed)


def price_level(df: pd.DataFrame) -> float:
    """Mean log price of a set of listings (the 'price level' used for calibration)."""
    return float(np.log(df["price_usd"]).mean())


def load_splits(cities=("nyc", "paris", "berlin")):
    """{city: (train, val, test)} for every requested city."""
    return {c: split(load_city(c)) for c in cities}
