"""Fast sanity tests: `python -m pytest -q`"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data import bath_type, clean_listings, license_status, parse_bathrooms, parse_price
from src.evaluate import inverse_label, transform_label
from src.features import CorrelationFilter, MultiHotEncoder, build_preprocessor


def test_parse_price():
    s = pd.Series(["$1,234.00", "$85.50", None])
    out = parse_price(s)
    assert out.iloc[0] == 1234.0 and out.iloc[1] == 85.5 and np.isnan(out.iloc[2])


def test_parse_bathrooms_and_type():
    s = pd.Series(["1.5 shared baths", "Half-bath", "2 baths", None])
    assert parse_bathrooms(s).tolist()[:3] == [1.5, 0.5, 2.0]
    assert list(bath_type(s)) == ["shared", "standard", "standard", "standard"]


def test_license_status():
    s = pd.Series([None, "Exempt", "OSE-STRREG-1"])
    assert list(license_status(s)) == ["none", "exempt", "registered"]


@pytest.mark.parametrize("label", ["raw", "TH", "sqrt", "log"])
def test_label_roundtrip(label):
    p = np.array([20.0, 150.0, 999.0])
    assert np.allclose(inverse_label(transform_label(p, label), label), p)


def test_correlation_filter_drops_duplicate_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.c_[a, a * 2 + 0.001, rng.normal(size=200)]
    f = CorrelationFilter(0.9).fit(X)
    assert f.transform(X).shape[1] == 2


def test_multihot_encoder():
    s = pd.Series([json.dumps(["Wifi", "Kitchen"]), json.dumps(["Wifi"]), "[]"])
    enc = MultiHotEncoder(top_k=5).fit(s)
    out = enc.transform(s)
    assert out.shape == (3, 2) and out[0].sum() == 2 and out[2].sum() == 0
    # ties are broken alphabetically, so the vocabulary does not depend on the hash seed
    tied = pd.Series([json.dumps(["c", "a", "b"]), json.dumps(["b", "c", "a"])])
    assert MultiHotEncoder(top_k=2).fit(tied).vocabulary_ == ["a", "b"]


def _fake_raw(n=60):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.random(n) for c in cfg.CONTINUOUS})
    df["latitude"], df["longitude"] = 40.7 + rng.random(n) / 10, -74 + rng.random(n) / 10
    df["price"] = [f"${v:.2f}" for v in rng.uniform(50, 400, n)]
    df["bathrooms_text"] = "1 bath"
    df["license"] = None
    df["amenities"] = json.dumps(["Wifi", "Kitchen"])
    for c in cfg.CATEGORICAL:
        df[c] = rng.choice(["a", "b"], n)
    for c in cfg.TEXT:
        df[c] = rng.choice(["cosy flat near park", "large loft with view", "room"], n)
    df["first_review"] = "2020-01-01"
    df["last_review"] = "2026-01-01"
    df["hosts_time_as_host_years"], df["hosts_time_as_host_months"] = 3, 2
    df["id"] = range(n)
    df["host_id"] = np.arange(n) // 3
    return df


def test_end_to_end_feature_matrix():
    df = clean_listings(_fake_raw(), "nyc")
    assert df["price_usd"].between(cfg.MIN_PRICE, cfg.MAX_PRICE).all()
    prep = build_preprocessor("C+O+D")
    X = prep.fit_transform(df)
    assert X.shape[0] == len(df) and np.isfinite(X).all()
    prep_ag = build_preprocessor("C+O", city_agnostic=True).fit(df)
    assert prep_ag.transform(df).shape[0] == len(df)


def test_host_grouped_split_has_no_host_overlap():
    from src.data import split
    df = clean_listings(_fake_raw(200), "nyc")
    df["host_id"] = np.arange(len(df)) // 4          # 4 listings per host
    tr, va, te = split(df, group_col="host_id")
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(tr.host_id) & set(te.host_id)
    assert not set(tr.host_id) & set(va.host_id)
    assert not set(va.host_id) & set(te.host_id)


def test_cleaned_strings_are_not_arrow_backed():
    """Arrow-backed strings would make the saved bundles depend on pyarrow."""
    df = clean_listings(_fake_raw(), "nyc")
    for col in cfg.CATEGORICAL + cfg.TEXT + ["amenities", "city"]:
        assert df[col].dtype == object, col


def test_dollar_scores_flag_overflowing_predictions():
    from src.evaluate import dollar_scores
    price = np.array([100.0, 200.0, 300.0])
    ok = dollar_scores(price, np.log(price), "log")
    assert ok["overflow_share"] == 0 and ok["mae_usd"] < 1e-6
    bad = dollar_scores(price, np.array([-50.0, -60.0, -70.0]), "log")
    assert bad["overflow_share"] == 1 and np.isnan(bad["mae_usd"])


def test_interval_coverage():
    from src.evaluate import interval_coverage, interval_from_residuals
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    pred = np.zeros_like(y)
    iv = interval_from_residuals(y, pred)
    assert abs(interval_coverage(y, pred, iv) - 0.8) < 0.02


@pytest.mark.parametrize("name", ["nyc", "paris", "berlin", "transfer"])
def test_saved_bundle_loads_and_predicts(name):
    """The shipped demo models must load and give a sensible price."""
    path = cfg.MODELS_DIR / f"{name}_xgb.joblib"
    if not path.exists():
        pytest.skip(f"{path.name} not built - run `python -m src.train_final`")
    from src.predict import load_bundle, make_listing, predict_price
    city = "berlin" if name == "transfer" else name
    if not (cfg.MODELS_DIR / f"{city}_xgb.joblib").exists():
        pytest.skip("in-city bundle needed for the listing template")
    bundle = load_bundle(name)
    assert {"pipeline", "metrics", "interval"} <= set(bundle)
    nb = next(iter(load_bundle(city)["neighbourhoods"]))
    listing = make_listing(city, neighbourhood_cleansed=nb, room_type="Entire home/apt",
                           accommodates=2, bedrooms=1)
    out = predict_price(city, listing, use_transfer=(name == "transfer"))
    assert 10 <= out["price"] <= 10_000
    assert out["low"] < out["price"] < out["high"]
