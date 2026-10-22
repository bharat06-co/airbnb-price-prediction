"""Central configuration: paths, data sources, feature lists and constants.

Every "magic number" in the project lives here so that experiments are
reproducible and easy to tweak.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Data sources (Inside Airbnb "detailed listings", CC BY 4.0)
# --------------------------------------------------------------------------- #
CITIES = {
    "nyc": {
        "name": "New York City",
        "url": "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz",
        "currency": "USD",
        "center": (40.7580, -73.9855),   # Times Square / Midtown
    },
    "paris": {
        "name": "Paris",
        "url": "https://data.insideairbnb.com/france/ile-de-france/paris/2026-09-16/data/listings.csv.gz",
        "currency": "EUR",
        "center": (48.8566, 2.3522),     # Hotel de Ville
    },
    "berlin": {
        "name": "Berlin",
        "url": "https://data.insideairbnb.com/germany/be/berlin/2026-09-19/data/listings.csv.gz",
        "currency": "EUR",
        "center": (52.5200, 13.4050),    # Mitte / Alexanderplatz
    },
}

# Inside Airbnb publishes prices in local currency. Convert everything to USD
# with one fixed, approximate rate so that cities are comparable.
FX_TO_USD = {"USD": 1.0, "EUR": 1.15}

# --------------------------------------------------------------------------- #
# Label settings
# --------------------------------------------------------------------------- #
MIN_PRICE = 10        # sanity filter: drop obviously broken listings
MAX_PRICE = 10_000    # sanity filter: drop placeholder / absurd prices
THRESHOLD = 1_000     # "TH" label: keep listings with price <= $1000/night
# The 2019 paper used a $500 cut-off (~1% of listings). 2026 prices are about
# twice as high, so $1000 plays the same role (removes ~3-4% in NYC/Paris).

SPLIT = (0.7, 0.2, 0.1)  # train : validation : test (same as the paper)

# Robustness check (Table 9): many hosts own several near-identical listings. A
# random split puts some of them in train and some in test, which flatters the
# score. The grouped split keeps every host's listings on one side.
GROUP_COLUMN = "host_id"

# Transfer experiments: share of Berlin's *training* split that is treated as
# "labelled target-city data". The same sample is used both for few-shot
# training and to estimate Berlin's price level for calibration, so the two
# approaches use the same information budget.
TARGET_LABELLED_FRACTION = 0.10

BOOTSTRAP_ROUNDS = 1000   # 95% confidence intervals for the Berlin test results
INTERVAL_QUANTILES = (0.1, 0.9)   # 80% prediction interval (from validation residuals)

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
# C: continuous features (before correlation pruning)
CONTINUOUS = [
    "host_listings_count",
    "latitude", "longitude",
    "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "maximum_nights",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90",
    "availability_365", "availability_eoy",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "number_of_reviews_ly", "estimated_occupancy_l365d", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    # engineered
    "amenities_count", "description_length",
]
# Continuous features tied to one city's geography. They are removed in the
# "city-agnostic" feature set (our transfer-learning extension) and replaced
# by a distance that means the same thing in every city.
CITY_SPECIFIC_CONTINUOUS = ["latitude", "longitude"]
CITY_AGNOSTIC_EXTRA = ["distance_to_center_km"]

# O: categorical features (one-hot) + list features (multi-hot)
CATEGORICAL = [
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "property_type", "room_type", "has_availability",
    "bath_type", "license_status",
]
CITY_SPECIFIC_CATEGORICAL = ["neighbourhood_cleansed", "neighbourhood_group_cleansed"]
LIST_FEATURES = ["amenities"]
TOP_K_AMENITIES = 100

# T: free-text features (TF-IDF -> truncated SVD, 50 dims each)
TEXT = ["name", "description", "host_about"]
SVD_DIM = 50
TFIDF_MAX_FEATURES = 5000

# D: date features, converted to "days since 2008-01-01" (Airbnb's founding)
DATE = ["first_review", "last_review"]
DATE_REFERENCE = "2008-01-01"
DATE_EXTRA = ["host_tenure_months"]

CORRELATION_THRESHOLD = 0.9  # drop one feature of every pair with |r| > 0.9

# Columns that leak the label and must never be used as features.
LEAKY_COLUMNS = [
    "price_quote_checkin_date", "price_quote_checkout_date",
    "price_quote_total_price", "price_quote_price_per_night",
    "price_quote_raw", "estimated_revenue_l365d",
]
