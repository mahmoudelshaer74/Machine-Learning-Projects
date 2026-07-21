"""
streamlit_app.py
=================
Interactive app for the NYC Taxi Trip Duration project.

Two tabs:
  1. EDA — key charts computed live from train.csv via processing.py.
  2. Predict — enter a trip's pickup/dropoff info and get a predicted
     duration from the trained pipeline (model.joblib, produced by
     train_model.py). No target-derived input is ever requested from
     the user (there is no "speed" or "duration" field to fill in) —
     this mirrors real deployment, where trip_duration is unknown.

Run:
    streamlit run streamlit_app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from processing import load_data, domain_filter, haversine_distance, calculate_bearing, manhattan_distance

st.set_page_config(page_title="NYC Taxi Trip Duration", layout="wide")


@st.cache_data
def get_clean_data(file_path="train.csv"):
    df = load_data(file_path)
    return domain_filter(df)


@st.cache_resource
def get_model(model_path="model.joblib"):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def build_single_trip_features(pickup_dt, pickup_lat, pickup_lon,
                                dropoff_lat, dropoff_lon, passenger_count, vendor_id,
                                store_and_fwd_flag, nums_col, cat_col):
    """Build one feature row identical in shape to training features,
    from prediction-time-only inputs (no trip_duration involved)."""
    row = {
        "vendor_id": vendor_id,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_lon, "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon, "dropoff_latitude": dropoff_lat,
        "store_and_fwd_flag": store_and_fwd_flag,
    }
    hour = pickup_dt.hour
    weekday = pickup_dt.weekday()
    row["Hour"] = hour
    row["Weekday"] = weekday
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["is_weekend"] = int(weekday >= 5)
    row["is_rush_hour"] = int(hour in [7, 8, 9, 16, 17, 18, 19])
    row["is_night"] = int(hour in [0, 1, 2, 3, 4, 5])

    dist = haversine_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    row["distance"] = dist
    row["Manhattan_distance"] = manhattan_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    row["Bearing"] = calculate_bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    row["abs_lat_diff"] = abs(dropoff_lat - pickup_lat)
    row["abs_lon_diff"] = abs(dropoff_lon - pickup_lon)
    row["center_lat"] = (pickup_lat + dropoff_lat) / 2
    row["center_lon"] = (pickup_lon + dropoff_lon) / 2
    row["distance_x_rush"] = dist * row["is_rush_hour"]
    row["distance_x_hour"] = dist * hour

    full = pd.DataFrame([row])
    return full[nums_col + cat_col]


st.title("🚖 NYC Taxi Trip Duration")
tab_eda, tab_predict = st.tabs(["📊 EDA", "🔮 Predict"])

with tab_eda:
    st.subheader("Exploratory analysis (train.csv required in the app folder)")
    if not os.path.exists("train.csv"):
        st.info("Place train.csv next to this app to enable live EDA.")
    else:
        df = get_clean_data("train.csv")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(np.log1p(df["trip_duration"]), bins=50, kde=True, ax=ax)
            ax.set_title("Log(1 + Trip Duration)")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x="Hour", y="trip_duration", estimator="median", ax=ax)
            ax.set_title("Median Duration by Hour")
            st.pyplot(fig)
        st.dataframe(df.head(20))

with tab_predict:
    st.subheader("Predict a trip's duration")
    bundle = get_model("model.joblib")
    if bundle is None:
        st.warning("model.joblib not found. Run `python train_model.py` first.")
    else:
        pipeline, nums_col, cat_col = bundle["pipeline"], bundle["nums_col"], bundle["cat_col"]

        c1, c2 = st.columns(2)
        with c1:
            pickup_date = st.date_input("Pickup date")
            pickup_time = st.time_input("Pickup time")
            vendor_id = st.selectbox("Vendor", [1, 2])
            passenger_count = st.slider("Passenger count", 1, 6, 1)
            store_and_fwd_flag = st.selectbox("Store & forward flag", ["N", "Y"])
        with c2:
            pickup_lat = st.number_input("Pickup latitude", value=40.7589, format="%.6f")
            pickup_lon = st.number_input("Pickup longitude", value=-73.9851, format="%.6f")
            dropoff_lat = st.number_input("Dropoff latitude", value=40.7306, format="%.6f")
            dropoff_lon = st.number_input("Dropoff longitude", value=-73.9352, format="%.6f")

        if st.button("Predict duration"):
            pickup_dt = pd.Timestamp.combine(pickup_date, pickup_time)
            X_single = build_single_trip_features(
                pickup_dt, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                passenger_count, vendor_id, store_and_fwd_flag, nums_col, cat_col,
            )
            pred_log = pipeline.predict(X_single)[0]
            pred_log = np.clip(pred_log, np.log1p(30), np.log1p(4 * 60 * 60))
            pred_seconds = np.expm1(pred_log)
            minutes = pred_seconds / 60
            st.success(f"Predicted trip duration: **{minutes:.1f} minutes** (~{pred_seconds:.0f} seconds)")
