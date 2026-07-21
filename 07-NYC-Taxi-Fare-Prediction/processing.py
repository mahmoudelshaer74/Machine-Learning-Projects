

import numpy as np
import pandas as pd

RANDOM_STATE = 42
NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 41.05
NYC_LON_MIN, NYC_LON_MAX = -74.35, -73.55

# Columns that are derived from (or equal to) the target and must
# NEVER be used as model features.
LEAKY_COLUMNS = ["trip_duration", "Log_trip_duration", "Speed_Kmh", "speed_kmh"]

TARGET = "trip_duration"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw CSV and drop columns unavailable at prediction time."""
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    # dropoff_datetime is not known at prediction time (a real deployment
    # only knows the pickup time), so it must never be used as a feature.
    if "dropoff_datetime" in df.columns:
        df = df.drop(columns=["dropoff_datetime"])
    return df


# --------------------------------------------------------------------------
# Geometry / time feature engineering
# --------------------------------------------------------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    earth_radius = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    theta = 2 * np.arcsin(np.sqrt(a))
    return earth_radius * theta


def calculate_bearing(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlng = lng2 - lng1
    x = np.sin(dlng) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def manhattan_distance(lon1, lat1, lon2, lat2):
    lat_distance = haversine_distance(lon1, lat1, lon1, lat2)
    lon_distance = haversine_distance(lon1, lat1, lon2, lat1)
    return lat_distance + lon_distance


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time and geometry features that are all computable at prediction
    time (pickup info only — no use of trip_duration or dropoff time).
    """
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    dt = df["pickup_datetime"]

    df["Hour"] = dt.dt.hour
    df["Weekday"] = dt.dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["is_weekend"] = (df["Weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_night"] = df["Hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    df["distance"] = haversine_distance(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"],
    )
    df["Manhattan_distance"] = manhattan_distance(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"],
    )
    df["Bearing"] = calculate_bearing(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["center_lat"] = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2
    df["center_lon"] = (df["pickup_longitude"] + df["dropoff_longitude"]) / 2
    df["distance_x_rush"] = df["distance"] * df["is_rush_hour"]
    df["distance_x_hour"] = df["distance"] * df["Hour"]
    return df


def domain_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove physically impossible / clearly bad records using FIXED,
    domain-based thresholds (not statistics computed from the data),
    so the same rule can be applied identically to train and test
    without leaking information between them.

    NOTE: this filter uses trip_duration to drop obviously corrupted
    labels (e.g. 1-second trips, 20-hour trips, implied speeds of
    600 km/h). This is standard label-cleaning, not feature leakage,
    because trip_duration is never turned into an input feature here.
    """
    df = add_features(df)
    speed_kmh = df["distance"] / (df["trip_duration"] / 3600)

    mask = (
        df["trip_duration"].between(30, 4 * 60 * 60)
        & df["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)
        & df["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)
        & speed_kmh.between(1, 120)
        & df["passenger_count"].between(1, 6)
        & df["distance"].between(0.1, 100)
    )
    return df.loc[mask].copy()


def get_feature_columns(df: pd.DataFrame):
    """
    Split columns into numeric / categorical feature lists, explicitly
    excluding the target and any target-derived (leaky) column.
    """
    usable = [c for c in df.columns if c not in LEAKY_COLUMNS + ["id", "pickup_datetime"]]
    nums_col = [c for c in usable if df[c].dtype in ["int64", "float64", "int32", "float32"]]
    cat_col = [c for c in usable if df[c].dtype in ["object", "category"]]
    return nums_col, cat_col


def time_based_split(df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2):
    """
    Chronological split (sorted by pickup_datetime): the model is trained
    on the past and evaluated on the future, matching the TimeSeriesSplit
    strategy used during cross-validation and matching how the model
    would actually be used in production.
    """
    df_sorted = df.sort_values("pickup_datetime").reset_index(drop=True)
    nums_col, cat_col = get_feature_columns(df_sorted)

    X = df_sorted[nums_col + cat_col]
    y = df_sorted[target]
    y_log = np.log1p(y)

    split = int(len(X) * (1 - test_size))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    y_tr_log, y_te_log = y_log.iloc[:split], y_log.iloc[split:]

    return {
        "X_tr": X_tr, "X_te": X_te,
        "y_tr": y_tr, "y_te": y_te,
        "y_tr_log": y_tr_log, "y_te_log": y_te_log,
        "nums_col": nums_col, "cat_col": cat_col,
    }


def run_pipeline(file_path: str = "train.csv", test_size: float = 0.2) -> dict:
    """End-to-end: load -> clean/engineer -> leakage-safe split."""
    df = load_data(file_path)
    df_clean = domain_filter(df)
    return time_based_split(df_clean, TARGET, test_size)


if __name__ == "__main__":
    data = run_pipeline("train.csv")
    print("Train shape:", data["X_tr"].shape, "Test shape:", data["X_te"].shape)
    print("Numeric features:", data["nums_col"])
    print("Categorical features:", data["cat_col"])
