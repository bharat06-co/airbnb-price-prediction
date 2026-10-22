"""Feature engineering: turns a cleaned listings table into a numeric matrix.

Feature groups (same naming as the paper):
    C  continuous   -> mean-impute, standardise, drop highly correlated columns
    I  interactions -> 2-degree interaction terms of C (optional)
    O  categorical  -> one-hot encoding + multi-hot "amenities" list feature
    T  text         -> TF-IDF (uni+bi-grams) -> truncated SVD (50 dims / column)
    D  date         -> days since 2008-01-01, mean-impute, standardise

Owner: Bharat V Bhat (feature extraction & engineering)
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import config as cfg


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one column of every pair whose |Pearson r| exceeds `threshold`.

    Fitted on the training data only, so no information leaks from test data.
    """

    def __init__(self, threshold: float = cfg.CORRELATION_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):   # constant columns
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        keep = []
        for j in range(n):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        self.n_features_in_ = n
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)[self.keep_]


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Encode a JSON list column (e.g. amenities) as top-k binary indicators.

    This is the "dictionary building and mapping" step from the paper.
    """

    def __init__(self, top_k: int = cfg.TOP_K_AMENITIES, prefix: str = "amenity"):
        self.top_k = top_k
        self.prefix = prefix

    @staticmethod
    def _parse(x):
        try:
            return [str(a).strip().lower() for a in json.loads(x)]
        except (TypeError, ValueError):
            return []

    def _series(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return pd.Series(np.asarray(X).ravel())

    def fit(self, X, y=None):
        counts = Counter(a for lst in self._series(X).map(self._parse) for a in set(lst))
        # Most frequent first, ties alphabetically. counts.most_common() would break
        # ties by insertion order, which comes from iterating a set of strings and so
        # changes with Python's hash seed on every run (column order, and with it the
        # XGBoost / MLP results, were not reproducible).
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.vocabulary_ = [a for a, _ in ranked[: self.top_k]]
        self._index = {a: i for i, a in enumerate(self.vocabulary_)}
        return self

    def transform(self, X):
        lists = self._series(X).map(self._parse)
        out = np.zeros((len(lists), len(self.vocabulary_)), dtype=np.float32)
        index = {a: i for i, a in enumerate(self.vocabulary_)}
        for r, lst in enumerate(lists):
            for a in lst:
                j = index.get(a)
                if j is not None:
                    out[r, j] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix}={a}" for a in self.vocabulary_], dtype=object)


class TextToDense(BaseEstimator, TransformerMixin):
    """Make sure a text column arrives as a 1-D array of strings."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return pd.Series(np.asarray(X).ravel()).fillna("").astype(str).values


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def continuous_columns(city_agnostic: bool = False):
    cols = list(cfg.CONTINUOUS)
    if city_agnostic:
        cols = [c for c in cols if c not in cfg.CITY_SPECIFIC_CONTINUOUS]
        cols += cfg.CITY_AGNOSTIC_EXTRA
    return cols


def categorical_columns(city_agnostic: bool = False):
    cols = list(cfg.CATEGORICAL)
    if city_agnostic:
        cols = [c for c in cols if c not in cfg.CITY_SPECIFIC_CATEGORICAL]
    return cols


def _text_vectorizer(mode: str):
    common = dict(max_features=cfg.TFIDF_MAX_FEATURES, min_df=3,
                  strip_accents="unicode", lowercase=True)
    if mode == "count_1":
        return CountVectorizer(ngram_range=(1, 1), **common)
    if mode == "tfidf_1":
        return TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, **common)
    if mode == "tfidf_12":
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, **common)
    raise ValueError(f"unknown text mode {mode!r}")


def build_preprocessor(groups: str = "C+O+T", text_mode: str = "tfidf_12",
                       city_agnostic: bool = False) -> ColumnTransformer:
    """Build an (unfitted) ColumnTransformer for a combination of feature groups.

    groups: '+'-separated subset of {C, I, O, T, D}, e.g. "C+O+T" or "C+I".
    city_agnostic: drop latitude/longitude/neighbourhood, add distance to centre.
    """
    parts = set(groups.split("+"))
    unknown = parts - set("CIOTD")
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}")
    transformers = []

    if "C" in parts or "I" in parts:
        steps = [("impute", SimpleImputer(strategy="mean")),
                 ("scale", StandardScaler()),
                 ("decorrelate", CorrelationFilter())]
        if "I" in parts:
            steps += [("interact", PolynomialFeatures(degree=2, interaction_only=True,
                                                      include_bias=False)),
                      ("rescale", StandardScaler())]
        transformers.append(("C", Pipeline(steps), continuous_columns(city_agnostic)))

    if "O" in parts:
        transformers.append(("O", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                min_frequency=10, sparse_output=False),
                             categorical_columns(city_agnostic)))
        for col in cfg.LIST_FEATURES:
            transformers.append((f"L_{col}", MultiHotEncoder(prefix=col), [col]))

    if "T" in parts:
        for col in cfg.TEXT:
            pipe = Pipeline([
                ("to_text", TextToDense()),
                ("vectorize", _text_vectorizer(text_mode)),
                ("svd", TruncatedSVD(n_components=cfg.SVD_DIM,
                                     random_state=cfg.RANDOM_STATE)),
            ])
            transformers.append((f"T_{col}", pipe, [col]))

    if "D" in parts:
        transformers.append(("D", Pipeline([("impute", SimpleImputer(strategy="mean")),
                                            ("scale", StandardScaler())]),
                             cfg.DATE + cfg.DATE_EXTRA))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0,
                             verbose_feature_names_out=True)


def feature_names(preprocessor: ColumnTransformer):
    """Readable feature names (SVD components are named <column>_svd<k>)."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name.startswith("T_"):
            names += [f"{cols[0]}_svd{k}" for k in range(cfg.SVD_DIM)]
        elif name == "C":
            fitted = list(cols)
            filt = trans.named_steps["decorrelate"]
            kept = list(np.asarray(fitted, dtype=object)[filt.keep_])
            if "interact" in trans.named_steps:
                kept = list(trans.named_steps["interact"].get_feature_names_out(kept))
            names += kept
        else:
            names += list(trans.get_feature_names_out(cols))
    return names
