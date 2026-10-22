# Predicting Airbnb Listing Price Across Different Cities

**UE24CS352A – Machine Learning · Mini-project · Problem statement #71**

| Team member | SRN | PRN |
|---|---|---|
| Bharat V Bhat | PES1UG24CS113 | PES1202403301 |
| Bharath Hegde | PES1UG24CS114 | PES1202400374 |

Reference paper: Luo, Zhou & Zhou, *Predicting Airbnb Listing Price Across Different Cities*, Stanford CS229, 2019
([PDF](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647491.pdf)).

We predict the nightly price of Airbnb listings in **New York City, Paris and Berlin**. We
re-implement the paper's whole pipeline on fresh **September 2026** Inside Airbnb data, and we
test zero-shot transfer to Berlin (no Berlin labels), and then study **target-city price-level
calibration**, which uses Berlin's mean log price **estimated from 571 labelled Berlin training
listings**. The calibrated extension is not strictly zero-shot; the zero-shot results are separate.

---

## Headline results

**In-city** (XGBoost, all features C+O+T, log label, random 70/20/10 split):

| City | Test R² (log price) | Test R² ($) | Test MAE |
|---|---|---|---|
| NYC | **0.888** | 0.723 | $72 |
| Paris | **0.818** | 0.703 | $99 |
| Berlin | **0.763** | 0.576 | $48 |

**Transfer to Berlin** (all rows scored on the same Berlin test split, 816 listings):

| Trained on | Model | Berlin labels used | Test R² (log) | Test MAE |
|---|---|---|---|---|
| NYC + Paris, paper features (zero-shot) | Neural network | none | 0.189 | $109 |
| NYC + Paris, city-agnostic features (zero-shot) | XGBoost | none | 0.111 | $111 |
| NYC + Paris, city-agnostic + **price-level calibration** | XGBoost | mean log price of a 10% Berlin sample (1 number) | **0.526** | $65 |
| NYC + Paris + the same 10% Berlin sample as training rows | XGBoost | 10% sample (571 listings) | 0.489 | $82 |
| Berlin only (in-city upper bound) | XGBoost | full Berlin training split | 0.777 | $46 |

* A **log transform of the label** is the single most useful choice. On the label scale R² goes from 0.654 to 0.865, but those two numbers are on different scales. Converted back to dollars, R² goes from 0.654 to 0.721 and MAE from $103 to $80.
* **XGBoost is the best model** in the controlled comparison (Table 1: NYC, same features), followed by random forest and then the neural network.
* A model trained on a single city **fails completely** on Berlin (R² ≪ 0). Training on NYC + Paris together works much better, as the paper found.
* **Our extension:** a major part of cross-city error is consistent with a different *price level* across cities. Calibrating with one number, Berlin's mean log price from a 10% labelled sample, gives the largest gain (XGBoost 0.111 → 0.526). Training on that same sample instead gives a similar 0.489 (the 95% confidence intervals overlap). City-agnostic features on their own do *not* help (0.189 → 0.171 for the NN), but for XGBoost they work together with calibration (paper features + price level 0.410 vs city-agnostic + price level 0.526; the separately bootstrapped CIs do not overlap, suggesting an improvement but not replacing a paired test). For the NN the two are about equal (0.502 vs 0.493). See Table 8 for the full ablation.
* **Honest caveat on transfer:** a model trained on the 10% Berlin sample *alone* (571 listings) already reaches 0.641, the same as the best transfer model (sample rows + price level, 0.632; CIs overlap). Transfer pays off when only the price level is known, not once a few hundred local labels exist.
* **Caveat (Table 9):** many hosts list several near-identical units. With a split that keeps each host's listings on one side, in-city R² drops to about 0.847 (NYC), 0.779 (Paris) and 0.752 (Berlin). The random-split numbers above are therefore somewhat optimistic.

All tables are in [`results/RESULTS.md`](results/RESULTS.md) and all figures are in [`results/figures/`](results/figures).
The 2-page write-up is [`docs/Problem71_Writeup.pdf`](docs/Problem71_Writeup.pdf).

---

## Repository structure

```
airbnb-price-prediction/
├── app.py                  # Streamlit demo (live demonstration)
├── requirements.txt
├── scripts/
│   └── download_data.py    # downloads the 3 Inside Airbnb files
├── src/
│   ├── config.py           # paths, URLs, feature lists, constants
│   ├── data.py             # loading, cleaning, 70/20/10 split
│   ├── features.py         # C / I / O / T / D feature groups (sklearn pipelines)
│   ├── models.py           # Ridge, KNN, Random Forest, XGBoost, MLP
│   ├── evaluate.py         # label transforms, R² / MSE / MAE, bootstrap CIs, intervals
│   ├── experiments.py      # Tables 1-9 + figures
│   ├── train_final.py      # trains + saves the models used by the app
│   ├── predict.py          # single-listing prediction (app + CLI)
│   └── plots.py            # figures
├── tests/test_pipeline.py  # unit, end-to-end and saved-model loading tests
├── docs/
│   ├── Problem71_Writeup.pdf   # 2-page write-up
│   ├── QA_Prep.md              # viva / Q&A preparation + demo script
│   └── writeup/                # write-up source (HTML) + its figures
├── data/raw/               # *.csv.gz go here (git-ignored)
├── models/                 # saved model bundles (*.joblib) + model_card.json
└── results/                # RESULTS.md, all_results.csv, tables/, figures/
```

---

## Setup

Use **Python 3.11 or newer** (required by the pinned scikit-learn/pandas versions).
A normal laptop CPU is sufficient; no GPU is needed. Clone this repository or unzip the project archive,
open the `airbnb-price-prediction/` folder, and run:

```bash
cd airbnb-price-prediction
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, clone the team's **private** GitHub repository using its URL, after the team
has granted you collaborator access. GitHub publication steps are in
[`GITHUB_SETUP.md`](GITHUB_SETUP.md).

The versions in `requirements.txt` are pinned because the saved models in `models/` are
pickled scikit-learn / XGBoost / pandas objects. With other versions, rebuild them with
`python -m src.train_final`.

### Data

**The three original, compressed September 2026 snapshots are included in this repository**
under `data/raw/`, for exact reproduction of the published results. No data download is
necessary for the demonstration or for the experiments. Every raw snapshot is below GitHub's
100 MB per-file limit, although the repository clone is correspondingly larger.

To fetch these from the source again instead, run:

```bash
python scripts/download_data.py      # skips files already present
```

The files come from [Inside Airbnb](https://insideairbnb.com/get-the-data/) (CC BY 4.0):
14 September (NYC), 16 September (Paris), 19 September (Berlin), 2026. Attribution and
snapshot information are in [`data/README.md`](data/README.md). Inside Airbnb retains only
recent snapshots online; **do not silently substitute a newer snapshot when reproducing the
reported results**, since the figures and tests may then differ.

## How to run

| Step | Command | Time* |
|---|---|---|
| Run the unit tests | `python -m pytest -q` | 2 s |
| Run every experiment (Tables 1–9 + figures) | `python -m src.experiments` | ~15 min |
| Quick smoke test on a 25% sample | `python -m src.experiments --quick` | ~2 min |
| Run only some tables | `python -m src.experiments --tables T1,T7` | – |
| Train + save the demo models | `python -m src.train_final` | ~3 min |
| Launch the demo app | `streamlit run app.py` | – |
| Predict from the command line | `python -m src.predict --city paris --room-type "Entire home/apt" --accommodates 4 --bedrooms 2 --neighbourhood Louvre` | 2 s |
| Rebuild the write-up figures | `python docs/writeup/make_figures.py` | 10 s |

\*Measured on a 2-core machine with 4 GB of RAM.

The trained models are already in `models/`, so the demo works straight away without retraining.

---

## Method

**Data.** We use the Inside Airbnb *detailed listings* files. We keep listings that have a price
between \$10 and \$10,000. Each file has 90 columns. Paris and Berlin prices are converted from
EUR to USD at a fixed rate of 1 EUR = 1.15 USD (set in `config.py`). Each city is split
70 / 20 / 10 into train / validation / test with seed 42. Table 9 repeats the main model with a
host-grouped split (`host_id`), where no host has listings in both train and test.

| City | Raw listings | With a valid price (used) |
|---|---|---|
| New York City | 29,918 | 19,396 |
| Paris | 77,149 | 43,495 |
| Berlin | 12,376 | 8,156 |

**Labels** (same as the paper): `raw` price, `TH` (only listings with price ≤ \$1000), `sqrt`, and `log`.

**Feature groups** (`src/features.py`). Every step is a scikit-learn transformer that is fitted on the training split only.

* **C – continuous (33):** 39 candidates, including size, minimum/maximum nights, availability, reviews, review scores, host listing counts, number of amenities and description length. Missing values get the mean, values are standardised, and one feature of every pair with |r| > 0.9 is dropped.
* **I – interactions:** degree-2 interaction terms of C.
* **O – categorical:** one-hot encoding of room/property type, neighbourhood, superhost, bathroom type, licence status and so on, plus a multi-hot vector of the 100 most common amenities.
* **T – text:** TF-IDF (unigrams + bigrams) on the title, description and host bio, reduced to 50 dimensions each with truncated SVD.
* **D – dates:** first and last review as days since 2008, plus host tenure.
* **City-agnostic variant (our extension):** latitude, longitude and neighbourhood are dropped, and *distance to the city centre* is added.

**Models** (`src/models.py`): Ridge regression and KNN (baselines), Random Forest, XGBoost
(validation early stopping), and an MLP with the paper's set-up. The MLP has 2 hidden layers of
64 and 32 units with ReLU. It is trained with SGD with Nesterov momentum (0.9), lr 0.005, L2 penalty 0.005 and
batch size 64. Training stops early when the validation R² improves by less than 1e-4 for 10 epochs.

**Metrics:** R² and MSE on the label scale, as in the paper, plus R² and MAE in dollars so that
different label transforms can be compared. Berlin transfer results also get a 95% bootstrap
confidence interval, because the Berlin test split is small (816 listings).

**Target-city price-level calibration (our extension).** The model learns each listing's log
price *minus its city's mean log price*. For Berlin we add back Berlin's mean log price. This is
**not** a zero-label experiment: Berlin's mean is estimated from a labelled 10% sample of
Berlin's training split (571 listings). The few-shot rows in Table 8 train on exactly the same
sample, so both approaches use the same Berlin labels. The zero-shot rows (no Berlin labels at
all) are reported separately. A sensitivity row shows that estimating the mean from the full
Berlin training split instead changes the result very little.

**Indicative price range (nominal 80%).** The demo uses the 10th and 90th percentiles of
*validation* log-residuals. On held-out test listings, observed coverage is 82.4% (NYC),
81.4% (Paris), and 82.1% (Berlin in-city). **The transfer band is not calibrated for Berlin:**
it comes from NYC + Paris residuals and covers only **61.4%** of Berlin's test prices. The
app and CLI explicitly warn about this; do not interpret it as an 80% Berlin interval.
All coverages are empirical marginal test-set figures, not guarantees for individual listings.

## Results (full numbers in `results/RESULTS.md`)

| Table | Finding |
|---|---|
| T1 Models (NYC, C, TH) | XGBoost 0.787 > Random Forest 0.778 > MLP 0.712 > KNN 0.659 > Ridge 0.466 |
| T2 Label transform | raw 0.654 → sqrt 0.809 → **log 0.865** on the label scale; in dollars R² 0.654 → 0.717 → 0.721. Interaction terms barely help (0.866) |
| T3 Text representation | unigram TF-IDF 0.587 ≈ uni+bigram TF-IDF 0.574 > counts 0.558 |
| T4 Feature groups | dates alone 0.154 (weak), categorical 0.718, C+O 0.811 → C+O+T 0.813 (TH); C+O 0.888 → C+O+T **0.888** (log), so text adds almost nothing |
| T5 Other cities | Paris 0.818, Berlin 0.763 (C+O+T, log). Without text: Paris 0.820, Berlin 0.777, so text does not help and slightly hurts in Berlin |
| T6 NN individual vs combined | NYC 0.835, Paris 0.774. The NYC+Paris model scores 0.804 on the pooled test set, and on each city separately 0.841 (NYC) and 0.773 (Paris): combining helps NYC slightly and leaves Paris unchanged |
| T7 Transfer to all of Berlin (NN, zero-shot) | NYC-only (−44,916) and Paris-only (−36,389) fail completely: their predictions overflow, so dollar metrics are n/a. **NYC+Paris 0.237** |
| T8 Our extension (Berlin test split) | Zero-shot: NN paper features 0.189, city-agnostic NN 0.171 / XGB 0.111. Price level (1 number from the 10% sample): XGB paper features 0.410, XGB city-agnostic **0.526** (95% CI 0.477–0.573), NN 0.493. 10% sample as training rows: XGB 0.489 (0.416–0.550, similar); rows + level 0.632; Berlin sample alone 0.641. Full Berlin model: 0.777. From NYC alone even calibration fails (−0.41) |
| T9 Host-grouped split | NYC 0.888 → 0.847, Paris 0.818 → 0.779, Berlin 0.763 → 0.752 (XGB, C+O+T, log) |

**Why does a single-city model collapse on Berlin?** The features are standardised with NYC's
statistics. Berlin's latitude (52.5°) is then about 200 standard deviations from NYC's mean
(40.7°, sd ≈ 0.06°). A ReLU network extrapolates linearly, so its predictions explode. When the
model is trained on NYC + Paris together, the standardised spread of latitude and longitude
becomes much wider. Berlin is still *outside* the training range (it lies north and east of
Paris), but only one or two standard deviations away instead of 200. That is largely why the
paper saw combined training "generalise". Much of the remaining error is a price-level bias,
which calibration removes (0.111 → 0.526 for XGBoost).

### Differences from the 2019 paper

* The 2026 Inside Airbnb schema no longer includes `security_deposit`, `cleaning_fee`, `transit`, `summary` or other text fields, and `host_since` and `neighborhood_overview` are empty. So we have 3 text fields instead of 12, and host tenure comes from `hosts_time_as_host_*`.
* The new `price_quote_*` and `estimated_revenue_l365d` columns contain the price itself, so they are **excluded to avoid label leakage**.
* In these snapshots, nightly prices are higher than in the 2019 paper, so the TH cut-off is \$1000 instead of \$500.
* The neural network uses scikit-learn's `MLPRegressor`, which gives exactly the paper's optimiser, regularisation and early-stopping rule.
* The paper's text-feature recipe (uni- + bigram TF-IDF) is kept for comparability, although unigrams scored slightly higher in our Table 3.

### Limitations

* Random splits put listings of the same host in train and test. Table 9 shows how much this inflates the scores.
* One snapshot per city, so there is no seasonality. Prices are listed prices, not booked prices.
* One seed. Berlin's test split is small, so read differences of a few hundredths between Berlin rows against the bootstrap confidence intervals in Table 8.

---

## Contributions

| Member | Work |
|---|---|
| **Bharat V Bhat** | Data pipeline and feature engineering: `data.py`, `features.py`, `config.py`, `scripts/download_data.py`. Data cleaning, label transforms, text TF-IDF/SVD and amenity encoding, correlation pruning, EDA figures, baselines (Ridge, KNN), Tables 2–4, unit tests. Co-wrote the report and slides. |
| **Bharath Hegde** | Models and experiments: `models.py`, `experiments.py`, `train_final.py`, `predict.py`, `app.py`. Random Forest, XGBoost and MLP, cross-city and transfer-learning experiments (Tables 1, 5–9), the price-level calibration extension, and the Streamlit demo. Co-wrote the report and slides. |

## References

1. Luo, Zhou, Zhou. *Predicting Airbnb Listing Price Across Different Cities.* CS229, Stanford, 2019.
2. Inside Airbnb – http://insideairbnb.com (data, CC BY 4.0).
3. Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
4. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 2011.
