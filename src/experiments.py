"""Run every experiment from the paper (Tables 1-7), our extension (Table 8) and a
host-grouped robustness check (Table 9).

Usage:
    python -m src.experiments            # full run (~10-15 min on a laptop)
    python -m src.experiments --quick    # 25% sample, for a fast smoke test
    python -m src.experiments --tables T1,T6

Results are written to results/tables/*.csv, results/RESULTS.md and
results/figures/*.png.

Owner: Bharath Hegde (experiments, neural network & transfer learning)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import plots
from .data import labelled_sample, load_city, price_level, split
from .evaluate import (LABELS, bootstrap_r2, dollar_scores, filter_for_label, scores,
                       transform_label)
from .features import build_preprocessor, feature_names
from .models import MODEL_NAMES, fit_model, make_model, predict

TABLE_TITLES = {  # noqa: E501
    "T1": "Table 1 - Continuous features with different models (NYC, TH label)",
    "T2": "Table 2 - XGBoost with different label transformations (NYC)",
    "T3": "Table 3 - XGBoost with different text representations (NYC, TH label)",
    "T4": "Table 4 - XGBoost with date / categorical / text features (NYC)",
    "T5": "Table 5 - XGBoost with all features on the other cities",
    "T6": "Table 6 - Neural network: individual vs combined datasets",
    "T7": "Table 7 - Transfer learning: train elsewhere, test on unseen Berlin",
    "T8": "Table 8 - Our extension: city-agnostic features, target-city price-level calibration "
          "and few-shot transfer (Berlin test split)",
    "T9": "Table 9 - Robustness: random split vs host-grouped split (XGBoost, C+O+T, log)",
}

# Short notes printed under some tables in RESULTS.md.
TABLE_NOTES = {
    "T6": "Rows marked *(combined model)* evaluate the NYC + Paris model on each city's test "
          "split separately. The pooled R² is not comparable with single-city R², because "
          "it also gets credit for the price gap between the two cities.",
    "T7": "*n/a*: almost every prediction overflows (log price far outside [-10, 12]), so "
          "dollar metrics are meaningless.",
    "T8": "**Berlin info used** states exactly which Berlin labels each row uses. *Price level* "
          "= the mean log price of the same 10% labelled Berlin sample that the few-shot rows "
          "train on (one number). No row uses Berlin's validation or test labels. The 95% CI is "
          "a bootstrap over the 816 Berlin test listings.",
    "T9": "Host-grouped split: every host's listings are in exactly one of train / validation "
          "/ test, so near-identical listings of multi-listing hosts cannot appear in both "
          "train and test.",
}

CITY_LABEL = {"nyc": "NYC", "paris": "Paris", "berlin": "Berlin"}


class Runner:
    """Holds the splits and caches fitted feature matrices between runs."""

    def __init__(self, quick: bool = False):
        self.splits = {}
        self.frames = {}
        for city in cfg.CITIES:
            df = load_city(city)
            if quick:
                df = df.sample(frac=0.25, random_state=cfg.RANDOM_STATE)
            self.frames[city] = df
            self.splits[city] = split(df)
        self.group_splits = {}
        # The labelled Berlin sample (10% of Berlin's training split). It is the
        # ONLY source of Berlin labels in the transfer experiments: few-shot rows
        # train on it, and calibrated rows use its mean log price.
        self.berlin_fewshot = labelled_sample(self.splits["berlin"][0])
        # Price level (mean log price) used by target-city price-level calibration.
        # Source cities: their own training split. Berlin: the 10% sample only.
        self.city_level = {c: price_level(s[0]) for c, s in self.splits.items()}
        self.city_level["berlin"] = price_level(self.berlin_fewshot)
        # Sensitivity check only: Berlin's level from its full training split.
        self.city_level_full = {c: price_level(s[0]) for c, s in self.splits.items()}
        self._cache = {}
        self.rows = []
        self.artifacts = {}

    # ---- datasets ----------------------------------------------------------
    def grouped(self, city):
        """Host-grouped 70/20/10 split of one city (Table 9)."""
        if city not in self.group_splits:
            self.group_splits[city] = split(self.frames[city], group_col=cfg.GROUP_COLUMN)
        return self.group_splits[city]

    def data(self, key: str):
        """'nyc', 'nyc+paris', 'nyc+paris+berlin10', 'berlin10', 'nyc@host' -> train, val, test.

        'berlin10' = the 10% labelled Berlin sample. On its own it is split 80/20
        into train / early-stopping data, so no other Berlin labels are used.
        '@host' = use the host-grouped split instead of the random one.
        """
        grouped = key.endswith("@host")
        parts = key.replace("@host", "").split("+")
        if parts == ["berlin10"]:
            tr, va = train_test_split(self.berlin_fewshot, train_size=0.8,
                                      random_state=cfg.RANDOM_STATE)
            return (tr.reset_index(drop=True), va.reset_index(drop=True),
                    self.splits["berlin"][2])
        cities = [p for p in parts if p in cfg.CITIES]
        src = {c: (self.grouped(c) if grouped else self.splits[c]) for c in cities}
        tr = [src[c][0] for c in cities]
        va = [src[c][1] for c in cities]
        te = [src[c][2] for c in cities]
        if "berlin10" in parts:
            tr.append(self.berlin_fewshot)
        return (pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True),
                pd.concat(te, ignore_index=True))

    def test_set(self, train_key: str, test: str):
        if test == "test":
            return self.data(train_key)[2]
        if test == "berlin_all":
            return pd.concat(self.splits["berlin"], ignore_index=True)
        if test == "berlin_test":
            return self.splits["berlin"][2]
        raise ValueError(test)

    # ---- features ----------------------------------------------------------
    def features(self, train_key, test, groups, label, text_mode, city_agnostic):
        th = label == "TH"
        key = (train_key, test, groups, th, text_mode, city_agnostic)
        if key not in self._cache:
            train, val, _ = self.data(train_key)
            test_df = self.test_set(train_key, test)
            train, val, test_df = (filter_for_label(d, label) for d in (train, val, test_df))
            prep = build_preprocessor(groups, text_mode=text_mode, city_agnostic=city_agnostic)
            X_tr = prep.fit_transform(train)
            X_va, X_te = prep.transform(val), prep.transform(test_df)
            self._cache[key] = dict(prep=prep, X=(X_tr, X_va, X_te),
                                    price=(train.price_usd.values, val.price_usd.values,
                                           test_df.price_usd.values),
                                    city=(train.city.values, val.city.values,
                                          test_df.city.values))
        return self._cache[key]

    # ---- one experiment ----------------------------------------------------
    def run(self, table, train_key, groups, label, model, *, test="test",
            text_mode="tfidf_12", city_agnostic=False, feature_label=None,
            data_label=None, keep=None, calibrate=False, berlin_info="",
            ci=False, per_city=False):
        """Fit one model and record its metrics.

        calibrate: False, True (Berlin level from the 10% labelled sample) or
                   "full" (Berlin level from its full training split; sensitivity check).
        ci:        add a 95% bootstrap interval for the test R².
        per_city:  also report the metrics on each city's part of the test set.
        """
        t0 = time.time()
        f = self.features(train_key, test, groups, label, text_mode, city_agnostic)
        X_tr, X_va, X_te = f["X"]
        p_tr, p_va, p_te = f["price"]
        y_tr, y_va, y_te = (transform_label(p, label) for p in (p_tr, p_va, p_te))
        if calibrate:
            # learn prices *relative to each city's average*, then add the
            # target city's average back at prediction time
            assert label == "log", "price-level calibration is defined for the log label"
            level = self.city_level_full if calibrate == "full" else self.city_level
            off_tr, off_va, off_te = (np.array([level[c] for c in cities])
                                      for cities in f["city"])
        else:
            off_tr = off_va = off_te = 0.0
        est = fit_model(make_model(model), model, X_tr, y_tr - off_tr, X_va, y_va - off_va)
        pred_tr, pred_te = predict(est, X_tr) + off_tr, predict(est, X_te) + off_te
        s_tr, s_te = scores(y_tr, pred_tr), scores(y_te, pred_te)
        d_te = dollar_scores(p_te, pred_te, label)
        ci_lo, ci_hi = bootstrap_r2(y_te, pred_te) if ci else (np.nan, np.nan)
        row = {
            "table": table, "model": MODEL_NAMES[model],
            "train_data": data_label or train_key.upper().replace("+", " + "),
            "test_data": {"test": "same-city test split", "berlin_all": "Berlin (all, unseen)",
                          "berlin_test": "Berlin test split"}[test],
            "features": feature_label or groups, "label": label,
            "berlin_info": berlin_info,
            "train_mse": s_tr["mse"], "train_r2": s_tr["r2"],
            "test_mse": s_te["mse"], "test_r2": s_te["r2"],
            "test_r2_ci_low": ci_lo, "test_r2_ci_high": ci_hi,
            "test_mae_usd": d_te["mae_usd"], "test_r2_usd": d_te["r2_usd"],
            "pred_overflow_share": d_te["overflow_share"],
            "n_train": len(y_tr), "n_test": len(y_te), "n_features": X_tr.shape[1],
            "seconds": round(time.time() - t0, 1),
        }
        self.rows.append(row)
        print(f"[{table}] {row['model']:<26} {row['train_data']:<22} {row['features']:<14} "
              f"{label:<5} train R2={s_tr['r2']:.3f} test R2={s_te['r2']:.3f} "
              f"MAE=${d_te['mae_usd']:.0f} ({row['seconds']}s)", flush=True)
        if per_city:
            test_cities = f["city"][2]
            for c in pd.unique(test_cities):
                m = test_cities == c
                s_c = scores(y_te[m], np.asarray(pred_te)[m])
                d_c = dollar_scores(p_te[m], np.asarray(pred_te)[m], label)
                self.rows.append({**row, "test_data": f"{CITY_LABEL[c]} test split (combined model)",
                                  "test_mse": s_c["mse"], "test_r2": s_c["r2"],
                                  "test_mae_usd": d_c["mae_usd"], "test_r2_usd": d_c["r2_usd"],
                                  "pred_overflow_share": d_c["overflow_share"],
                                  "n_test": int(m.sum())})
        if keep:
            self.artifacts[keep] = dict(y_true=y_te, y_pred=pred_te, model=est, prep=f["prep"])
        return row


# --------------------------------------------------------------------------- #
# Experiment definitions
# --------------------------------------------------------------------------- #
def table1(r):
    for m in ["linear", "knn", "rf", "mlp", "xgb"]:
        r.run("T1", "nyc", "C", "TH", m)


def table2(r):
    r.run("T2", "nyc", "C", "raw", "xgb", keep="nyc_raw")
    r.run("T2", "nyc", "C", "sqrt", "xgb")
    r.run("T2", "nyc", "C", "log", "xgb", keep="nyc_log")
    r.run("T2", "nyc", "C+I", "log", "xgb")


def table3(r):
    for mode, lab in [("count_1", "Text - unigram count"), ("tfidf_1", "Text - unigram tf-idf"),
                      ("tfidf_12", "Text - uni+bigram tf-idf")]:
        r.run("T3", "nyc", "T", "TH", "xgb", text_mode=mode, feature_label=lab)


def table4(r):
    r.run("T4", "nyc", "D", "TH", "xgb")
    r.run("T4", "nyc", "O", "TH", "xgb")
    r.run("T4", "nyc", "C+O", "TH", "xgb")            # text ablation: C+O vs C+O+T
    r.run("T4", "nyc", "C+O+T", "TH", "xgb")
    r.run("T4", "nyc", "C+O", "log", "xgb")
    r.run("T4", "nyc", "C+O+T", "log", "xgb", keep="nyc_best")
    r.run("T4", "nyc", "C+O+T+D", "log", "xgb")


def table5(r):
    for city in ["paris", "berlin"]:
        r.run("T5", city, "C+O+T", "TH", "xgb")
        r.run("T5", city, "C+O", "log", "xgb")        # text ablation
        r.run("T5", city, "C+O+T", "log", "xgb")


def table6(r):
    r.run("T6", "nyc", "C+O", "log", "mlp")
    r.run("T6", "paris", "C+O", "log", "mlp")
    # pooled test R² plus the same model scored on each city's test split
    r.run("T6", "nyc+paris", "C+O", "log", "mlp", per_city=True)


def table7(r):
    none = "none"
    r.run("T7", "nyc", "C+O", "log", "mlp", test="berlin_all", data_label="Train on NYC only",
          keep="tl_nyc", berlin_info=none)
    r.run("T7", "paris", "C+O", "log", "mlp", test="berlin_all",
          data_label="Train on Paris only", berlin_info=none)
    r.run("T7", "nyc+paris", "C+O", "log", "mlp", test="berlin_all",
          data_label="Train on NYC + Paris", keep="tl_both", berlin_info=none)


def table8(r):
    t = "berlin_test"
    kw = dict(test=t, ci=True)
    NONE = "none"
    LEVEL = "price level of 10% sample (1 number)"
    ROWS = "10% sample as training rows"
    BOTH = "10% sample: rows + price level"
    # (a) zero-shot: no Berlin labels at all
    r.run("T8", "nyc", "C+O", "log", "mlp", city_agnostic=True, data_label="NYC only",
          feature_label="C+O city-agnostic", berlin_info=NONE, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "mlp", data_label="NYC + Paris",
          feature_label="C+O (paper)", berlin_info=NONE, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "mlp", city_agnostic=True, data_label="NYC + Paris",
          feature_label="C+O city-agnostic", berlin_info=NONE, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "xgb", city_agnostic=True, data_label="NYC + Paris",
          feature_label="C+O city-agnostic", berlin_info=NONE, **kw)
    # (b) target-city price-level calibration: one number from the 10% Berlin sample
    r.run("T8", "nyc", "C+O", "log", "xgb", city_agnostic=True, calibrate=True,
          data_label="NYC only", feature_label="C+O agnostic + price level",
          berlin_info=LEVEL, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "mlp", calibrate=True, data_label="NYC + Paris",
          feature_label="C+O (paper) + price level", berlin_info=LEVEL, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "xgb", calibrate=True, data_label="NYC + Paris",
          feature_label="C+O (paper) + price level", berlin_info=LEVEL, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "mlp", city_agnostic=True, calibrate=True,
          data_label="NYC + Paris", feature_label="C+O agnostic + price level",
          berlin_info=LEVEL, **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "xgb", city_agnostic=True, calibrate=True,
          data_label="NYC + Paris", feature_label="C+O agnostic + price level",
          berlin_info=LEVEL, keep="tl_calibrated", **kw)
    r.run("T8", "nyc+paris", "C+O", "log", "xgb", city_agnostic=True, calibrate="full",
          data_label="NYC + Paris", feature_label="C+O agnostic + price level (full-train mean)",
          berlin_info="price level of full Berlin train split (sensitivity check)", **kw)
    # (c) few-shot: the same 10% Berlin sample used as training rows
    r.run("T8", "nyc+paris+berlin10", "C+O", "log", "mlp", city_agnostic=True,
          data_label="NYC + Paris + 10% Berlin", feature_label="C+O city-agnostic",
          berlin_info=ROWS, **kw)
    r.run("T8", "nyc+paris+berlin10", "C+O", "log", "xgb", city_agnostic=True,
          data_label="NYC + Paris + 10% Berlin", feature_label="C+O city-agnostic",
          berlin_info=ROWS, keep="tl_fewshot", **kw)
    r.run("T8", "nyc+paris+berlin10", "C+O", "log", "xgb", city_agnostic=True, calibrate=True,
          data_label="NYC + Paris + 10% Berlin", feature_label="C+O agnostic + price level",
          berlin_info=BOTH, **kw)
    r.run("T8", "berlin10", "C+O", "log", "xgb", data_label="10% Berlin sample only",
          feature_label="C+O", berlin_info=ROWS, **kw)
    # (d) in-city reference (upper bound): the full Berlin training split
    r.run("T8", "berlin", "C+O", "log", "xgb", test="test", ci=True,
          data_label="Berlin only (in-city reference)", feature_label="C+O",
          berlin_info="full Berlin train + validation splits")


def table9(r):
    for city in ["nyc", "paris", "berlin"]:
        r.run("T9", city, "C+O+T", "log", "xgb", data_label=f"{CITY_LABEL[city]}, random split")
        r.run("T9", f"{city}@host", "C+O+T", "log", "xgb",
              data_label=f"{CITY_LABEL[city]}, host-grouped split")


TABLES = {"T1": table1, "T2": table2, "T3": table3, "T4": table4,
          "T5": table5, "T6": table6, "T7": table7, "T8": table8, "T9": table9}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_usd(v, digits=0):
    return "n/a" if pd.isna(v) else f"{v:.{digits}f}"


def _markdown(df: pd.DataFrame, table: str = "") -> str:
    transfer = table in ("T7", "T8")
    with_ci = df.get("test_r2_ci_low", pd.Series(dtype=float)).notna().any()
    head = ["Model", "Train data", "Test data", "Features", "Label"]
    if transfer:
        head.append("Berlin info used")
    head += ["Train MSE", "Train R²", "Test MSE", "Test R²"]
    if with_ci:
        head.append("95% CI")
    head += ["Test R² ($)", "Test MAE ($)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = df["test_r2"].max()
    for _, r in df.iterrows():
        vals = [str(r[c]) for c in ["model", "train_data", "test_data", "features", "label"]]
        if transfer:
            info = r.get("berlin_info", "")
            vals.append("" if pd.isna(info) else str(info))
        vals += [f"{r['train_mse']:.4g}", f"{r['train_r2']:.3f}", f"{r['test_mse']:.4g}",
                 f"**{r['test_r2']:.3f}**" if r["test_r2"] == best else f"{r['test_r2']:.3f}"]
        if with_ci:
            vals.append("" if pd.isna(r["test_r2_ci_low"])
                        else f"[{r['test_r2_ci_low']:.3f}, {r['test_r2_ci_high']:.3f}]")
        vals += [_fmt_usd(r["test_r2_usd"], 3), _fmt_usd(r["test_mae_usd"])]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_results(r: Runner):
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(r.rows)
    old = cfg.RESULTS_DIR / "all_results.csv"
    if old.exists():  # partial re-run: keep the tables that were not re-run
        prev = pd.read_csv(old)
        res = pd.concat([prev[~prev.table.isin(res.table.unique())], res], ignore_index=True)
    res = res.sort_values("table", kind="stable").reset_index(drop=True)
    res.to_csv(cfg.RESULTS_DIR / "all_results.csv", index=False)
    md = ["# Results", "",
          "All metrics are computed on the held-out data. MSE and R² are on the scale of the "
          "label used for training (as in the paper). **R² on different labels (raw, sqrt, log) "
          "is not directly comparable**, so *Test R² ($)* and *Test MAE ($)* convert the "
          "predictions back to US dollars. TH rows are also evaluated only on listings <= "
          f"${cfg.THRESHOLD}.", "",
          f"Label key: " + "; ".join(f"**{k}** = {v}" for k, v in LABELS.items()) + ".", ""]
    for t, g in res.groupby("table", sort=True):
        g.to_csv(cfg.TABLES_DIR / f"{t}.csv", index=False)
        md += [f"## {TABLE_TITLES[t]}", "", _markdown(g, t), ""]
        if t in TABLE_NOTES:
            md += [TABLE_NOTES[t], ""]
    (cfg.RESULTS_DIR / "RESULTS.md").write_text("\n".join(md))
    return res


def make_figures(r: Runner, res: pd.DataFrame):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frames = {c: pd.concat(s) for c, s in r.splits.items()}
    plots.price_distribution(frames)

    nyc_train = r.splits["nyc"][0]
    from .features import continuous_columns
    cols = continuous_columns()
    X = nyc_train[cols].fillna(nyc_train[cols].mean()).values
    plots.correlation_heatmap(X, cols)

    a = r.artifacts
    if "nyc_raw" in a and "nyc_log" in a:
        plots.pred_vs_true(
            [("(a) No transformation", a["nyc_raw"]["y_true"], a["nyc_raw"]["y_pred"], "price ($)"),
             ("(b) Log transformation", a["nyc_log"]["y_true"], a["nyc_log"]["y_pred"], "log price")],
            "label_transformation.png", "XGBoost on continuous features (NYC test set)")
    if "tl_nyc" in a and "tl_both" in a:
        plots.pred_vs_true(
            [("(a) Train NYC -> test Berlin", a["tl_nyc"]["y_true"], a["tl_nyc"]["y_pred"], "log price"),
             ("(b) Train NYC + Paris -> test Berlin", a["tl_both"]["y_true"], a["tl_both"]["y_pred"], "log price")],
            "transfer_learning.png", "Neural network transfer to an unseen city")
    if "nyc_best" in a:
        est, prep = a["nyc_best"]["model"], a["nyc_best"]["prep"]
        names = feature_names(prep)
        plots.feature_importance(names, est.feature_importances_)
        plots.pred_vs_true([("XGBoost, C+O+T, log label", a["nyc_best"]["y_true"],
                             a["nyc_best"]["y_pred"], "log price")],
                           "best_model_nyc.png", "Best NYC model on the test set")
    t1 = res[res.table == "T1"]
    if len(t1):
        plots.bar_r2(t1, "model", "model_comparison.png",
                     "Test R² by model - NYC, continuous features, TH label")
    t8 = res[res.table == "T8"]
    if len(t8):
        t8 = t8.assign(lbl=t8["model"].str.replace(" (MLP)", "").str.replace("Neural network", "NN")
                       + " | " + t8["train_data"] + " | " + t8["features"])
        ours = ((t8["model"] == "XGBoost") & (t8["train_data"] == "NYC + Paris")
                & (t8["features"] == "C+O agnostic + price level")).values
        plots.bar_r2(t8, "lbl", "transfer_extension.png",
                     "Predicting Berlin prices (Berlin test split; see 'Berlin info used' in T8)",
                     highlight=ours,
                     reference=t8["train_data"].str.startswith("Berlin only").values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="use a 25%% sample of every city")
    ap.add_argument("--tables", default=",".join(TABLES), help="comma-separated, e.g. T1,T7")
    args = ap.parse_args()

    t0 = time.time()
    r = Runner(quick=args.quick)
    for t in args.tables.split(","):
        print(f"\n=== {TABLE_TITLES[t]} ===", flush=True)
        TABLES[t](r)
    res = write_results(r)
    make_figures(r, res)
    summary = {
        "n_listings": {c: int(sum(len(s) for s in sp)) for c, sp in r.splits.items()},
        "runtime_minutes": round((time.time() - t0) / 60, 1),
        "quick": args.quick,
    }
    (cfg.RESULTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {summary['runtime_minutes']} min -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
