"""Streamlit demo: predict the nightly price of an Airbnb listing.

Run:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import config as cfg
from src.predict import load_bundle, make_listing, predict_price

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏠", layout="wide")

CITY_KEYS = list(cfg.CITIES)


@st.cache_resource
def bundle(name):
    return load_bundle(name)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Settings")
city = st.sidebar.radio("City", CITY_KEYS, format_func=lambda c: cfg.CITIES[c]["name"])
b = bundle(city)
use_transfer = False
if city == "berlin":
    use_transfer = st.sidebar.toggle(
        "Use the transfer model (NYC + Paris)",
        help="Trained only on NYC + Paris listings with city-agnostic features. The only "
             "Berlin information it uses is Berlin's average log price, estimated from a "
             "10% labelled sample of Berlin listings (target-city price-level calibration).")
# the model whose numbers are shown everywhere below
active = bundle("transfer") if use_transfer else b
info = active["metrics"]
st.sidebar.markdown("**Model quality (held-out test set)**")
st.sidebar.metric("R² on log price", f"{info['test_r2_log']:.3f}")
st.sidebar.metric("Mean absolute error", f"${info['mae_usd']:.0f}")
if "interval_coverage_test" in info:
    st.sidebar.metric("Nominal 80% band: observed test coverage",
                      f"{info['interval_coverage_test']:.1%}")
    if use_transfer:
        st.sidebar.warning("The transfer price band covered only 61.4% of held-out "
                           "Berlin prices. It is not calibrated to 80% coverage in Berlin.")
st.sidebar.caption("Model: XGBoost, continuous + categorical + text features, log label."
                   if not use_transfer else
                   "Model: XGBoost trained on NYC + Paris only, city-agnostic features, "
                   "Berlin price level from a 10% labelled Berlin sample.")

st.title("Airbnb nightly price predictor")
st.caption("UE24CS352A Machine Learning mini-project · Problem 71: Predicting Airbnb "
           "listing price across different cities · Bharat V Bhat & Bharath Hegde")

tab_predict, tab_perf, tab_exp = st.tabs(["Predict a price", "Model performance", "Experiments"])

# --------------------------------------------------------------------------- #
# Tab 1 - prediction form
# --------------------------------------------------------------------------- #
with tab_predict:
    t = b["template"]
    with st.form("listing"):
        c1, c2, c3 = st.columns(3)
        nbs = sorted(b["neighbourhoods"])
        default_nb = {"nyc": "Midtown", "paris": "Louvre", "berlin": "Alexanderplatz"}.get(city)
        nb = c1.selectbox("Neighbourhood", nbs,
                          index=nbs.index(default_nb) if default_nb in nbs else 0)
        room = c1.selectbox("Room type", b["room_types"])
        ptype = c1.selectbox("Property type", b["property_types"])
        bath_type = c1.selectbox("Bathroom type", ["standard", "private", "shared"])

        acc = c2.slider("Guests (accommodates)", 1, 16, 2)
        bedrooms = c2.number_input("Bedrooms", 0, 10, 1)
        beds = c2.number_input("Beds", 0, 16, 1)
        baths = c2.number_input("Bathrooms", 0.0, 8.0, 1.0, step=0.5)

        min_nights = c3.number_input("Minimum nights", 1, 365, int(t["minimum_nights"]))
        rating = c3.slider("Review score (out of 5)", 0.0, 5.0, 4.8, 0.05)
        n_reviews = c3.number_input("Number of reviews", 0, 3000, 20)
        superhost = c3.checkbox("Host is a Superhost")

        amen_all = b["amenities"]
        amenities = st.multiselect("Amenities", amen_all, default=amen_all[:15],
                                   format_func=lambda a: a.capitalize())
        title = st.text_input("Listing title", "Bright, cosy apartment close to the metro")
        desc = st.text_area("Description",
                            "Sunny apartment with a fully equipped kitchen, fast wifi and a "
                            "comfortable queen bed. Five minutes walk from the subway.")
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        listing = make_listing(
            city, neighbourhood_cleansed=nb, room_type=room, property_type=ptype,
            bath_type=bath_type, accommodates=acc, bedrooms=bedrooms, beds=beds,
            bathrooms=baths, minimum_nights=min_nights, review_scores_rating=rating,
            number_of_reviews=n_reviews, host_is_superhost="t" if superhost else "f",
            amenities=amenities, name=title, description=desc)
        out = predict_price(city, listing, use_transfer=use_transfer)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated nightly price", f"${out['price']:,.0f}")
        m2.metric("Source-city band (uncalibrated)" if use_transfer
                  else "Indicative price range (nominal 80%)",
                  f"${out['low']:,.0f} – ${out['high']:,.0f}")
        typical = b["typical_price"].get(f"{nb}|{room}")
        if typical:
            m3.metric(f"Median of all {room.lower()} listings in {nb}", f"${typical:,.0f}",
                      help="All sizes and minimum stays, so it can differ a lot from the "
                           "estimate for this specific listing.")
        if cfg.CITIES[city]["currency"] == "EUR":
            st.caption(f"≈ €{out['price'] / cfg.FX_TO_USD['EUR']:,.0f} per night "
                       f"(1 EUR = {cfg.FX_TO_USD['EUR']} USD)")
        if use_transfer:
            st.info("The model is trained only on NYC + Paris listings, but its Berlin price "
                    "level is estimated from 571 labelled Berlin training listings. It is not "
                    "strictly zero-shot.")
            st.warning("**Range limitation:** these bounds come from NYC + Paris validation "
                       "residuals and captured **61.4%**, not 80%, of held-out Berlin prices. "
                       "Treat the band as illustrative, not a calibrated Berlin interval.")

# --------------------------------------------------------------------------- #
# Tab 2 - performance of the selected model (in-city or transfer)
# --------------------------------------------------------------------------- #
with tab_perf:
    title = (f"Transfer model (NYC + Paris → Berlin test split)" if use_transfer
             else f"{b['city_name']} model (test set)")
    if "test_sample" not in active or "importance" not in active:
        st.warning("This model bundle has no saved evaluation data. "
                   "Run `python -m src.train_final` to rebuild the models.")
    else:
        st.subheader(f"{title}: predicted vs actual price")
        sample = active["test_sample"]
        lim = [10, 5000]
        scatter = alt.Chart(sample).mark_circle(size=12, opacity=0.4, color="#2783DE").encode(
            x=alt.X("true", scale=alt.Scale(type="log", domain=lim), title="Actual price ($)"),
            y=alt.Y("pred", scale=alt.Scale(type="log", domain=lim), title="Predicted price ($)"))
        diag = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(
            color="#2C2C2B", strokeDash=[4, 4]).encode(x="x", y="y")
        st.altair_chart(scatter + diag, width="stretch")
        st.subheader(f"Most important features ({'transfer' if use_transfer else 'in-city'} model)")
        imp = pd.DataFrame({"feature": list(active["importance"]),
                            "gain": list(active["importance"].values())})
        st.altair_chart(alt.Chart(imp).mark_bar(color="#2783DE").encode(
            x=alt.X("gain", title="Importance (gain share)"),
            y=alt.Y("feature", sort="-x", title=None)), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 3 - experiment results
# --------------------------------------------------------------------------- #
with tab_exp:
    results = cfg.RESULTS_DIR / "RESULTS.md"
    if results.exists():
        st.markdown(results.read_text())
        for fig in ["model_comparison.png", "label_transformation.png",
                    "transfer_learning.png", "transfer_extension.png"]:
            path = cfg.FIGURES_DIR / fig
            if path.exists():
                st.image(str(path))
    else:
        st.warning("Run `python -m src.experiments` to generate the results.")
