"""
eda.py
======
Exploratory Data Analysis for the NYC Taxi Trip Duration dataset.
Pure analysis / visualization — no modeling here (see processing.py
for cleaning/features and streamlit_app.py for the interactive app).

Run:
    python eda.py
Figures are saved under ./eda_figures/ instead of only being shown,
so they can be reused (e.g. in the Streamlit app or a report).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from processing import load_data, add_features, domain_filter

FIG_DIR = "eda_figures"
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=120, bbox_inches="tight")
    plt.close()


def column_info(df, col):
    print(f"Column: {col}")
    print(f"  dtype: {df[col].dtype}")
    print(f"  unique: {df[col].nunique()}")
    print(f"  missing: {df[col].isnull().sum()}")
    print(f"  min/max: {df[col].min()} / {df[col].max()}")
    print(f"  std: {df[col].std():.3f}  skew: {df[col].skew():.3f}")


def plot_target_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["trip_duration"], bins=50, kde=True, ax=ax[0])
    ax[0].set_title("Trip Duration (seconds)")
    sns.histplot(np.log1p(df["trip_duration"]), bins=50, kde=True, ax=ax[1], color="green")
    ax[1].set_title("Log(1 + Trip Duration)")
    savefig("target_distribution.png")


def plot_groupby_stats(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    vendor_group = df.groupby("vendor_id")["trip_duration"].median().reset_index()
    sns.barplot(data=vendor_group, x="vendor_id", y="trip_duration", ax=axes[0])
    axes[0].set_title("Median Trip Duration by Vendor")

    passenger_group = df.groupby("passenger_count")["trip_duration"].median().reset_index()
    sns.barplot(data=passenger_group, x="passenger_count", y="trip_duration", ax=axes[1])
    axes[1].set_title("Median Trip Duration by Passenger Count")
    savefig("groupby_stats.png")


def plot_temporal_patterns(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.lineplot(data=df, x="Hour", y="trip_duration", estimator="median", ax=axes[0, 0])
    axes[0, 0].set_title("Median Trip Duration by Hour")

    sns.boxplot(data=df, x="is_rush_hour", y="trip_duration", ax=axes[0, 1])
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["No", "Yes"])
    axes[0, 1].set_title("Trip Duration: Rush Hour vs Normal")

    sns.barplot(data=df, x="Weekday", y="trip_duration", estimator="median",
                errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Median Trip Duration by Day of Week (0=Monday)")

    sns.violinplot(data=df, x="is_weekend", y="trip_duration", ax=axes[1, 1])
    axes[1, 1].set_title("Trip Duration: Weekend vs Weekday")
    savefig("temporal_patterns.png")


def plot_geo_patterns(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["pickup_longitude"], df["pickup_latitude"], alpha=0.1, s=1)
    ax.set_xlim(df["pickup_longitude"].quantile(0.01), df["pickup_longitude"].quantile(0.99))
    ax.set_ylim(df["pickup_latitude"].quantile(0.01), df["pickup_latitude"].quantile(0.99))
    ax.set_title("Pickup Density in NYC")
    savefig("pickup_density.png")


def plot_correlation(df):
    # Only correlate genuine model FEATURES against the target — never
    # include Speed_Kmh/Log_trip_duration here, they are perfectly (or
    # near-perfectly) correlated with trip_duration by construction and
    # would be misleading, not informative.
    cols = ["trip_duration", "distance", "Manhattan_distance", "Bearing",
            "Hour", "is_rush_hour", "is_weekend", "abs_lat_diff", "abs_lon_diff"]
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (leakage columns excluded on purpose)")
    savefig("correlation_matrix.png")


def summarize(df) -> dict:
    return {
        "shape": list(df.shape),
        "missing_percent": (df.isna().mean().sort_values() * 100).round(3).to_dict(),
        "target_seconds": df["trip_duration"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_dict(),
        "distance_km": df["distance"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_dict(),
        "median_duration_by_hour": df.groupby("Hour")["trip_duration"].median().round().to_dict(),
        "median_duration_by_weekday": df.groupby("Weekday")["trip_duration"].median().round().to_dict(),
        "median_duration_by_vendor": df.groupby("vendor_id")["trip_duration"].median().round().to_dict(),
        "store_and_fwd_counts": df["store_and_fwd_flag"].value_counts(dropna=False).to_dict(),
    }


def run_eda(file_path: str = "train.csv"):
    df = load_data(file_path)
    df = domain_filter(df)  # add_features + physical cleaning, computed ONCE

    for col in ["vendor_id", "trip_duration", "passenger_count"]:
        column_info(df, col)
        print()

    info = summarize(df)
    print(info)

    plot_target_distribution(df)
    plot_groupby_stats(df)
    plot_temporal_patterns(df)
    plot_geo_patterns(df)
    plot_correlation(df)
    print(f"Figures saved to ./{FIG_DIR}/")
    return df, info


if __name__ == "__main__":
    run_eda("train.csv")
