# Results

All metrics are computed on the held-out data. MSE and R² are on the scale of the label used for training (as in the paper). **R² on different labels (raw, sqrt, log) is not directly comparable**, so *Test R² ($)* and *Test MAE ($)* convert the predictions back to US dollars. TH rows are also evaluated only on listings <= $1000.

Label key: **raw** = N/A (raw price); **TH** = Threshold (price <= $1000); **sqrt** = Square root; **log** = Log.

## Table 1 - Continuous features with different models (NYC, TH label)

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear regression (L2) | NYC | same-city test split | C | TH | 2.117e+04 | 0.444 | 2.128e+04 | 0.466 | 0.466 | 102 |
| K-nearest neighbours | NYC | same-city test split | C | TH | 9486 | 0.751 | 1.358e+04 | 0.659 | 0.659 | 76 |
| Random forest | NYC | same-city test split | C | TH | 1331 | 0.965 | 8870 | 0.778 | 0.778 | 58 |
| Neural network (MLP) | NYC | same-city test split | C | TH | 9280 | 0.756 | 1.146e+04 | 0.712 | 0.712 | 69 |
| XGBoost | NYC | same-city test split | C | TH | 1154 | 0.970 | 8485 | **0.787** | 0.787 | 58 |

## Table 2 - XGBoost with different label transformations (NYC)

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | NYC | same-city test split | C | raw | 2.24e+04 | 0.899 | 5.944e+04 | 0.654 | 0.654 | 103 |
| XGBoost | NYC | same-city test split | C | sqrt | 2.627 | 0.960 | 11.99 | 0.809 | 0.717 | 84 |
| XGBoost | NYC | same-city test split | C | log | 0.02217 | 0.971 | 0.1021 | 0.865 | 0.721 | 80 |
| XGBoost | NYC | same-city test split | C+I | log | 0.0157 | 0.979 | 0.1011 | **0.866** | 0.725 | 79 |

## Table 3 - XGBoost with different text representations (NYC, TH label)

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | NYC | same-city test split | Text - unigram count | TH | 2111 | 0.945 | 1.762e+04 | 0.558 | 0.558 | 90 |
| XGBoost | NYC | same-city test split | Text - unigram tf-idf | TH | 2211 | 0.942 | 1.646e+04 | **0.587** | 0.587 | 87 |
| XGBoost | NYC | same-city test split | Text - uni+bigram tf-idf | TH | 1975 | 0.948 | 1.697e+04 | 0.574 | 0.574 | 88 |

## Table 4 - XGBoost with date / categorical / text features (NYC)

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | NYC | same-city test split | D | TH | 3.105e+04 | 0.185 | 3.373e+04 | 0.154 | 0.154 | 134 |
| XGBoost | NYC | same-city test split | O | TH | 4710 | 0.876 | 1.123e+04 | 0.718 | 0.718 | 69 |
| XGBoost | NYC | same-city test split | C+O | TH | 1702 | 0.955 | 7533 | 0.811 | 0.811 | 53 |
| XGBoost | NYC | same-city test split | C+O+T | TH | 432.1 | 0.989 | 7456 | 0.813 | 0.813 | 53 |
| XGBoost | NYC | same-city test split | C+O | log | 0.02059 | 0.973 | 0.08447 | **0.888** | 0.732 | 73 |
| XGBoost | NYC | same-city test split | C+O+T | log | 0.0006249 | 0.999 | 0.08504 | 0.888 | 0.723 | 72 |
| XGBoost | NYC | same-city test split | C+O+T+D | log | 0.001266 | 0.998 | 0.08566 | 0.887 | 0.744 | 73 |

## Table 5 - XGBoost with all features on the other cities

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | PARIS | same-city test split | C+O+T | TH | 1825 | 0.950 | 1.012e+04 | 0.726 | 0.726 | 67 |
| XGBoost | PARIS | same-city test split | C+O | log | 0.03098 | 0.946 | 0.1031 | **0.820** | 0.728 | 97 |
| XGBoost | PARIS | same-city test split | C+O+T | log | 0.01039 | 0.982 | 0.104 | 0.818 | 0.703 | 99 |
| XGBoost | BERLIN | same-city test split | C+O+T | TH | 302.4 | 0.982 | 5277 | 0.659 | 0.659 | 46 |
| XGBoost | BERLIN | same-city test split | C+O | log | 0.01186 | 0.974 | 0.1008 | 0.777 | 0.670 | 46 |
| XGBoost | BERLIN | same-city test split | C+O+T | log | 0.01453 | 0.968 | 0.107 | 0.763 | 0.576 | 48 |

## Table 6 - Neural network: individual vs combined datasets

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural network (MLP) | NYC | same-city test split | C+O | log | 0.09851 | 0.869 | 0.1246 | 0.835 | 0.506 | 93 |
| Neural network (MLP) | PARIS | same-city test split | C+O | log | 0.09512 | 0.835 | 0.129 | 0.774 | 0.638 | 111 |
| Neural network (MLP) | NYC + PARIS | same-city test split | C+O | log | 0.1148 | 0.823 | 0.1268 | 0.804 | 0.611 | 107 |
| Neural network (MLP) | NYC + PARIS | NYC test split (combined model) | C+O | log | 0.1148 | 0.823 | 0.12 | **0.841** | 0.580 | 91 |
| Neural network (MLP) | NYC + PARIS | Paris test split (combined model) | C+O | log | 0.1148 | 0.823 | 0.1298 | 0.773 | 0.618 | 114 |

Rows marked *(combined model)* evaluate the NYC + Paris model on each city's test split separately. The pooled R² is not comparable with single-city R², because it also gets credit for the price gap between the two cities.

## Table 7 - Transfer learning: train elsewhere, test on unseen Berlin

| Model | Train data | Test data | Features | Label | Berlin info used | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Neural network (MLP) | Train on NYC only | Berlin (all, unseen) | C+O | log | none | 0.09851 | 0.869 | 2.049e+04 | -44916.350 | n/a | n/a |
| Neural network (MLP) | Train on Paris only | Berlin (all, unseen) | C+O | log | none | 0.09512 | 0.835 | 1.66e+04 | -36388.718 | n/a | n/a |
| Neural network (MLP) | Train on NYC + Paris | Berlin (all, unseen) | C+O | log | none | 0.1148 | 0.823 | 0.3478 | **0.237** | -0.085 | 106 |

*n/a*: almost every prediction overflows (log price far outside [-10, 12]), so dollar metrics are meaningless.

## Table 8 - Our extension: city-agnostic features, target-city price-level calibration and few-shot transfer (Berlin test split)

| Model | Train data | Test data | Features | Label | Berlin info used | Train MSE | Train R² | Test MSE | Test R² | 95% CI | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Neural network (MLP) | NYC only | Berlin test split | C+O city-agnostic | log | none | 0.1173 | 0.844 | 1.072 | -1.377 | [-1.654, -1.122] | -4.057 | 274 |
| Neural network (MLP) | NYC + Paris | Berlin test split | C+O (paper) | log | none | 0.1148 | 0.823 | 0.3659 | 0.189 | [0.075, 0.296] | -0.374 | 109 |
| Neural network (MLP) | NYC + Paris | Berlin test split | C+O city-agnostic | log | none | 0.1216 | 0.813 | 0.3737 | 0.171 | [0.072, 0.266] | -0.198 | 112 |
| XGBoost | NYC + Paris | Berlin test split | C+O city-agnostic | log | none | 0.03936 | 0.939 | 0.4012 | 0.111 | [0.001, 0.201] | -0.023 | 111 |
| XGBoost | NYC only | Berlin test split | C+O agnostic + price level | log | price level of 10% sample (1 number) | 0.02137 | 0.972 | 0.6353 | -0.409 | [-0.577, -0.259] | -1.088 | 168 |
| Neural network (MLP) | NYC + Paris | Berlin test split | C+O (paper) + price level | log | price level of 10% sample (1 number) | 0.1055 | 0.837 | 0.2246 | 0.502 | [0.433, 0.567] | 0.400 | 67 |
| XGBoost | NYC + Paris | Berlin test split | C+O (paper) + price level | log | price level of 10% sample (1 number) | 0.0357 | 0.945 | 0.2659 | 0.410 | [0.341, 0.476] | 0.297 | 71 |
| Neural network (MLP) | NYC + Paris | Berlin test split | C+O agnostic + price level | log | price level of 10% sample (1 number) | 0.131 | 0.798 | 0.2286 | 0.493 | [0.436, 0.547] | 0.364 | 67 |
| XGBoost | NYC + Paris | Berlin test split | C+O agnostic + price level | log | price level of 10% sample (1 number) | 0.04447 | 0.932 | 0.2139 | 0.526 | [0.477, 0.573] | 0.398 | 65 |
| XGBoost | NYC + Paris | Berlin test split | C+O agnostic + price level (full-train mean) | log | price level of full Berlin train split (sensitivity check) | 0.04447 | 0.932 | 0.2136 | 0.526 | [0.478, 0.573] | 0.401 | 65 |
| Neural network (MLP) | NYC + Paris + 10% Berlin | Berlin test split | C+O city-agnostic | log | 10% sample as training rows | 0.1251 | 0.807 | 0.2741 | 0.392 | [0.318, 0.460] | 0.029 | 93 |
| XGBoost | NYC + Paris + 10% Berlin | Berlin test split | C+O city-agnostic | log | 10% sample as training rows | 0.03973 | 0.939 | 0.2307 | 0.489 | [0.416, 0.550] | 0.276 | 82 |
| XGBoost | NYC + Paris + 10% Berlin | Berlin test split | C+O agnostic + price level | log | 10% sample: rows + price level | 0.03278 | 0.950 | 0.1658 | 0.632 | [0.588, 0.677] | 0.459 | 58 |
| XGBoost | 10% Berlin sample only | Berlin test split | C+O | log | 10% sample as training rows | 0.0004976 | 0.999 | 0.1619 | 0.641 | [0.588, 0.688] | 0.448 | 57 |
| XGBoost | Berlin only (in-city reference) | same-city test split | C+O | log | full Berlin train + validation splits | 0.01186 | 0.974 | 0.1008 | **0.777** | [0.742, 0.810] | 0.670 | 46 |

**Berlin info used** states exactly which Berlin labels each row uses. *Price level* = the mean log price of the same 10% labelled Berlin sample that the few-shot rows train on (one number). No row uses Berlin's validation or test labels. The 95% CI is a bootstrap over the 816 Berlin test listings.

## Table 9 - Robustness: random split vs host-grouped split (XGBoost, C+O+T, log)

| Model | Train data | Test data | Features | Label | Train MSE | Train R² | Test MSE | Test R² | Test R² ($) | Test MAE ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | NYC, random split | same-city test split | C+O+T | log | 0.0006249 | 0.999 | 0.08504 | **0.888** | 0.723 | 72 |
| XGBoost | NYC, host-grouped split | same-city test split | C+O+T | log | 0.01428 | 0.980 | 0.1141 | 0.847 | 0.590 | 81 |
| XGBoost | Paris, random split | same-city test split | C+O+T | log | 0.01039 | 0.982 | 0.104 | 0.818 | 0.703 | 99 |
| XGBoost | Paris, host-grouped split | same-city test split | C+O+T | log | 0.02224 | 0.962 | 0.1232 | 0.779 | 0.501 | 110 |
| XGBoost | Berlin, random split | same-city test split | C+O+T | log | 0.01453 | 0.968 | 0.107 | 0.763 | 0.576 | 48 |
| XGBoost | Berlin, host-grouped split | same-city test split | C+O+T | log | 0.01422 | 0.969 | 0.1139 | 0.752 | 0.627 | 53 |

Host-grouped split: every host's listings are in exactly one of train / validation / test, so near-identical listings of multi-listing hosts cannot appear in both train and test.
