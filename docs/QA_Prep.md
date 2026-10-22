# Viva / Q&A preparation – Problem statement #71

*Predicting Airbnb Listing Price Across Different Cities* · UE24CS352A Machine Learning

Short, honest answers to the questions the panel is most likely to ask. Every number
comes from `results/RESULTS.md` (seed 42, test split, R² on the label scale unless noted).

---

## 1. Problem and data

**Q. What problem are you solving?**
Regression: predict the nightly price (USD) of an Airbnb listing from its attributes, in NYC,
Paris and Berlin. Second question: can a model trained on some cities price listings in a city it
has never seen (Berlin)? This follows Luo, Zhou & Zhou (Stanford CS229, 2019).

**Q. Why is this useful?**
New hosts need a fair starting price; Airbnb/competitors need pricing suggestions in new markets
where they have little data. The transfer question matters exactly in that "new market" case.

**Q. Where does the data come from? Why not the Kaggle file?**
Inside Airbnb *detailed listings* (CC BY 4.0), September 2026 snapshots: NYC (2026-09-14),
Paris (2026-09-16), Berlin (2026-09-19). The popular Kaggle NYC file has only ~16 columns; the
paper used the ~96-column detailed files, which only Inside Airbnb provides.

**Q. How many rows?**
Raw 29,918 / 77,149 / 12,376. With a valid price (\$10–\$10,000): 19,396 / 43,495 / 8,156.
About 40% of raw 2026 rows lack a usable price; we exclude them without assuming they are inactive, rather than
impute the label.

**Q. How did you split?**
Per city, random 70 / 20 / 10 train / validation / test, seed 42 (same as the paper). All
preprocessing (imputation means, scalers, TF-IDF vocabulary, SVD, top-100 amenities, correlation
pruning) is fitted on the **training split only** to avoid leakage.

**Q. Did you have any leakage problems?**
Yes, and we caught them. The 2026 schema adds `price_quote_*` and `estimated_revenue_l365d`,
which are computed from the price. Including them would give near-perfect R². We exclude them
explicitly in `config.py`.

**Q. Is there leakage between train and test through hosts?**
Partly, yes. Many hosts list several near-identical units. With a random split, 58% of NYC test
listings have a host who also has listings in train, and 26% have a description identical to a
training listing. Table 9 repeats the main model with a host-grouped split (no host on both sides):
R² drops from 0.888 to 0.847 (NYC), 0.818 to 0.779 (Paris) and 0.763 to 0.752
(Berlin). So the random-split numbers are somewhat optimistic. We kept the random split as the main
protocol because it is what the paper used.

**Q. Currency?**
Paris/Berlin are in EUR; we convert with a fixed 1 EUR = 1.15 USD so all cities share one scale.
A fixed rate doesn't change R² within a city; it only matters for cross-city training.

## 2. Features

**Q. What are the feature groups?**
- **C (continuous, 33 after pruning):** accommodates, bedrooms, beds, bathrooms, min/max nights,
  availability, review counts and scores, host listing counts, number of amenities, description
  length, lat/long, … Mean imputation + standardisation; one of every pair with |r| > 0.9 dropped.
- **I:** degree-2 interactions of C.
- **O (categorical):** one-hot room type, property type, neighbourhood, superhost, bathroom
  type, licence status, …, plus multi-hot top-100 amenities.
- **T (text):** TF-IDF (uni+bigram) on name, description, host_about → TruncatedSVD to 50 dims each.
- **D (dates):** days since first/last review, host tenure.

**Q. Why TruncatedSVD and not PCA on text?**
TF-IDF matrices are sparse; SVD works directly on sparse input without centring (that's LSA).
PCA would densify a huge matrix.

**Q. Why remove correlated features?**
Following the paper; it stabilises the linear/NN models (multicollinearity) and makes
importances easier to read. Trees don't need it but aren't hurt.

**Q. Which features matter most?**
From XGBoost gain importance in NYC: shared bathroom, licence status (exempt / registered),
entire home vs. private room, Manhattan, minimum nights, the "building staff" amenity, then size
(accommodates, bedrooms, bathrooms). Licence status and minimum nights matter because of NYC's
short-term rental law: unregistered listings must take 30+ night stays, which changes the price.
See `results/figures/feature_importance.png`.

**Q. Did text help?**
Hardly at all. NYC: C+O 0.888 → C+O+T 0.888 (log) and 0.811 → 0.813 (TH). Paris: 0.820 → 0.818.
In Berlin text slightly *hurts* (C+O 0.777 → C+O+T 0.763). We keep C+O+T as the main model because it is the paper's set-up. In Table 3 unigram TF-IDF (0.587) beat
raw counts (0.558) and bigrams didn't add more (0.574); we kept uni+bigrams to match the paper.
In 2026 only 3 text fields are filled (the paper had 12), so text carries less signal.

**Q. Dates?**
Dates alone give R² 0.154; adding D to C+O+T changes nothing (0.888 → 0.887). Same conclusion as the paper.

## 3. Labels and metrics

**Q. Why log price?**
Prices are heavily right-skewed with a long tail of luxury listings. Squared error on raw price
is dominated by a few outliers. Log makes the distribution near-normal and turns the error into a
relative (percentage) error. XGBoost NYC C: raw 0.654 → sqrt 0.809 → log 0.865.

**Q. Are R² values on log and raw price comparable?**
Not directly – they are computed on different scales. That's why we also report **MAE in dollars**
for every run (predictions are converted back with exp), and R² in dollars. In dollars the label-transform gain
is smaller: raw 0.654 → sqrt 0.717 → log 0.721 (MAE \$103 → \$84 → \$80). C+O+T TH has a lower MAE
(\$53) than log (\$72) partly because TH removes expensive listings from the test set too.

**Q. What is the TH label?**
Drop listings above \$1000 (paper used \$500 in 2019; prices roughly doubled). It removes ~3–4% of
NYC/Paris listings.

**Q. Why R²?**
It is scale-free, so it can be compared across cities, and it is what the paper reports. We also
keep MSE (paper) and MAE (interpretable).

## 4. Models

**Q. Which models and why?**
Ridge and KNN as baselines, Random Forest, XGBoost, and an MLP – the same five as the paper.
Table 1 (NYC, C, TH): XGBoost 0.787 > RF 0.778 > MLP 0.712 > KNN 0.659 > Ridge 0.466.

**Q. Why does XGBoost win?**
Tabular data with mixed types, non-linear effects and interactions (e.g. bedrooms × room type ×
location). Gradient-boosted trees capture these without scaling and are robust to irrelevant
one-hot columns. NNs usually need more data/tuning on tabular data.

**Q. XGBoost hyperparameters?**
learning rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.6, min_child_weight 3,
L2 1.0, up to 3000 trees with **early stopping (100 rounds) on the validation split**.

**Q. Train R² is 0.93–0.999 but test is ~0.8–0.89. Is it overfitting?**
There is a train–test gap, which is normal for boosted trees with depth 6 and thousands of
features. The *test* score is what matters and it's measured on untouched data. The number of
trees is chosen by early stopping on the validation set, so boosting stops as soon as validation
error stops improving; subsampling (0.8 rows, 0.6 columns) and L2 also regularise. Next step
would be tuning depth / min_child_weight on the validation split.

**Q. Describe the neural network.**
As in the paper's final set-up: 2 hidden layers (64, 32), ReLU, MSE loss, SGD with Nesterov
momentum 0.9, lr 0.005, L2 0.005, batch 64, early stopping when validation R² improves < 1e-4 for
10 epochs. Implemented with scikit-learn `MLPRegressor` (target standardised for stable SGD).

**Q. Why not PyTorch/TensorFlow?**
MLPRegressor implements exactly this architecture and optimiser, runs on CPU in seconds, and keeps
the dependency list small. The paper's net is small enough that a framework gives no benefit.

**Q. KNN – what k, what distance?**
k = 5, uniform weights, Euclidean distance on the standardised features.

## 5. Cross-city results and transfer

**Q. Results in each city?**
XGBoost, C+O+T, log label: NYC **0.888** (MAE \$72), Paris **0.818** (\$99), Berlin **0.763** (\$48).
Paper (2019 data): NYC 0.740, Paris 0.704.

**Q. Why are your numbers higher than the paper's?**
Fresher data (rows without usable price are excluded), more amenities/review fields,
better XGBoost settings with early stopping, and a stricter sanity filter on price. We can't claim
it's purely a better method – the dataset changed. Part of the gap is also same-host near-duplicates that a random split puts
in both train and test (Table 9).

**Q. Does combining cities help (Table 6)?**
MLP C+O log: NYC 0.835, Paris 0.774. The combined model gets 0.804 on the pooled test set, but a
pooled R² is not comparable: it also gets credit for the price gap between NYC and Paris. Scored on
each city separately, the combined model gets 0.841 (NYC) and 0.773 (Paris), so combining helps NYC a
little (0.835 → 0.841) and leaves Paris unchanged (0.774 → 0.773). The paper reported only the pooled
number (0.773).

**Q. Transfer to Berlin (Table 7)?**
Trained on NYC only: R² = −44,916; Paris only: −36,389; NYC+Paris: **0.237** (MAE \$106).
The paper got −3.5 / −17.4 / 0.514. Same qualitative story: single-city models fail, combined works.
For the single-city models the MAE is shown as n/a: almost all predictions overflow and are clipped
to about \$0, so the old "\$191" was just Berlin's mean price.

**Q. Why does a single-city model give R² of −40,000?**
Lat/long are standardised with NYC's mean and sd. Berlin's latitude (52.5°) is ~200 standard
deviations away from NYC's (40.7°, sd ≈ 0.06°). A ReLU network is piecewise linear, so it
extrapolates linearly and the outputs explode. Combined NYC+Paris training spreads lat/long so
Berlin is "only" one or two sd away, although it is still outside the training range. So the paper's "generalisation" is partly an artefact of feature
scaling, not of learning city-independent pricing.

**Q. Your extension?**
Two changes (Table 8, tested on Berlin's test split):
1. **City-agnostic features** – drop lat/long/neighbourhood and add *distance to city centre*.
2. **Target-city price-level calibration** – the model learns log price **minus its city's mean
   log price**; for Berlin we add Berlin's mean log price back. That mean is estimated from a
   labelled 10% sample of Berlin's training split (571 listings).

Results (trained on NYC + Paris, same Berlin test split, same model where possible):
- Zero-shot, no Berlin labels: NN paper features 0.189, NN city-agnostic 0.171, XGB city-agnostic 0.111.
  So city-agnostic features alone do **not** help.
- Price level only: XGB paper features 0.410, XGB city-agnostic **0.526** (95% CI 0.477–0.573),
  NN city-agnostic 0.493 (NN paper features + level: 0.502). For XGBoost the two parts work
  together: city-agnostic features alone don't help, but with calibration they beat the paper
  features (separate 95% bootstrap CIs do not overlap; a paired-difference CI would be stronger). For the NN the two are about equal.
- Same 10% sample as training rows: XGB 0.489 (CI 0.416–0.550). That is *similar to* calibration
  (CIs overlap), not clearly worse. Rows + price level 0.632; that sample alone 0.641.
  Full Berlin model: 0.777 (upper bound).

**Q. If 571 Berlin labels alone give 0.641, why transfer at all?**
Fair point, and we say so. Once the 10% sample is available, a Berlin-only model is as good as the
best transfer model (0.641 vs 0.632, CIs overlap). Transfer pays off when only the price level is
known: one number lifts XGBoost from 0.111 (zero-shot) to 0.526. A price level needs far fewer
labels than a model (or could come from public statistics), but we only tested the 10% sample.

**Q. Does the extension work from a single source city?**
No. Trained on NYC only, the city-agnostic NN gets −1.38 and XGB + price level −0.41 on Berlin.
It no longer explodes (−44,916 before), but one city's market structure doesn't carry over. For
example, 73% of priced NYC listings require a stay of 30+ nights (NYC's short-term rental rules),
versus 19% in Berlin and 13% in Paris, so NYC-learned relations between minimum stay and price are
wrong for Berlin. Several source cities are needed.

**Q. Isn't using Berlin's mean price cheating?**
It is not zero-shot, and we say so. The mean comes from a labelled 10% sample of Berlin's
*training* split, never from validation or test. To compare fairly, the few-shot models train on
exactly the same 10% sample, so both approaches use the same Berlin labels. The zero-shot rows
are reported separately. A sensitivity row that uses the full Berlin training mean instead gives
0.526, the same as with the sample (0.526), so the estimate is stable even from 571 listings.

**Q. Is the demo's price interval guaranteed to contain 80% of Berlin prices?**
No. The band uses the 10th and 90th percentiles of validation log-residuals and has a **nominal**
80% target. Observed held-out coverage is 82.4% / 81.4% / 82.1% for NYC / Paris / Berlin
in-city models. The transfer band comes from NYC + Paris validation residuals and covers only
**61.4% of Berlin test prices**. It is too narrow for Berlin and the app warns explicitly.
The band does not use Berlin test labels to set its width, and none of these are guarantees for
an individual listing. Recalibrating on separate local calibration data is future work.

**Q. Why is the remaining gap (0.53 vs 0.78) still there?**
Different cities value features differently (e.g. how much an extra bedroom is worth, which
neighbourhood types are premium, different property-type mix and regulations). A shared model can
learn general structure, but not city-specific slopes without local labels.

## 6. Code and repo

**Q. Walk us through the code.**
`config.py` (all constants) → `data.py` (load, clean, split) → `features.py` (ColumnTransformer
with C/I/O/T/D groups) → `models.py` (model factory) → `evaluate.py` (label transforms, metrics) →
`experiments.py` (Tables 1–9, figures, RESULTS.md) → `train_final.py` (saves models) →
`predict.py` (inference) → `app.py` (Streamlit demo). Tests in `tests/test_pipeline.py`.

**Q. How do you guarantee reproducibility?**
Fixed seed 42 everywhere, pinned requirement versions, all constants in `config.py`, a single
command (`python -m src.experiments`) regenerates every table and figure, and preprocessing is a
fitted sklearn pipeline saved together with the model in the `.joblib` bundle. While re-running we
found one hidden source of randomness: the top-100 amenity list broke ties between equally common
amenities in an order that depended on Python's hash seed, so Berlin's R² moved by ~0.002 between
runs. Ties are now broken alphabetically (and a test checks it), so every run gives identical numbers.

**Q. What do the tests check?**
Price parsing ("$1,234.00" → 1234), bathroom text parsing and type, licence status, label
transform round-trips (raw/TH/sqrt/log), the correlation filter dropping a duplicate column, the
multi-hot amenity encoder, and an end-to-end clean → feature-matrix run on synthetic listings
(finite values, right shape, city-agnostic variant), the host-grouped split (no host on two sides),
plain (non-Arrow) strings, overflow handling in the dollar metrics, interval coverage, and that
every saved model bundle loads and gives a sensible price. 18 tests, a few seconds.

**Q. How does the demo work?**
`streamlit run app.py`. Pick a city, enter listing details (neighbourhood, room/property type,
guests, bedrooms, beds, bathrooms, minimum nights, review score, superhost, amenities, title,
description), and the saved XGBoost bundle (preprocessor + model in one `.joblib`) predicts the
nightly price with a residual-based indicative range and the neighbourhood median for comparison. Other tabs show model performance and all experiment tables.
A Berlin toggle switches to the NYC+Paris transfer model. The sidebar and the *Model performance*
tab then show that model's own test results.

## 7. Limitations / future work

- One snapshot per city; prices are seasonal. Use calendar data / multiple snapshots.
- Listing price ≠ booked price (no transaction data).
- Fixed exchange rate; no cost-of-living or demand features.
- Hyper-parameters lightly tuned; could use Bayesian search / LightGBM / CatBoost.
- Transfer: try domain adaptation, city embeddings, or hierarchical models; more source cities.
- Images of listings (CNN features) were not used.

## 8. Demo script (≈3 minutes)

1. `python -m pytest -q` – 18 tests pass.
2. `streamlit run app.py` → *Predict a price* tab: NYC, Midtown, Entire home/apt, 4 guests, 2 bedrooms → show price.
   Change to Private room → the price drops (about 18% for the same size). Room type is one of the
   top features, together with bathroom type and licence status.
3. Switch to Paris and Berlin with the same inputs; then turn on *Use the unseen-city model* for
   Berlin and compare with the Berlin-trained model; the performance tab switches with it.
4. *Model performance* tab: predicted vs actual plot and feature importance.
5. *Experiments* tab: Tables 1–9; point at Table 7 (collapse) and Table 8 (our fix).
6. Optional CLI: `python -m src.predict --city paris --room-type "Entire home/apt" --accommodates 4 --bedrooms 2 --neighbourhood Louvre`.
