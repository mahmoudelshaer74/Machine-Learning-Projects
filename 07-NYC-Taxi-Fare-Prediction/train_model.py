"""
train_model.py
===============
Trains and compares the recommended regressors for NYC Taxi Trip Duration,
using the leakage-safe pipeline from processing.py, then saves the best
model to model.joblib for use by streamlit_app.py.

Why these models (see the write-up for full reasoning):
  - HistGradientBoostingRegressor: strong baseline, native NaN/categorical
    handling, fast on ~1M rows, no external dependency.
  - LightGBM: usually the best accuracy/speed trade-off on this exact
    Kaggle dataset in public leaderboards; used if installed.
  - Random Forest: useful sanity-check baseline, but slower and usually
    a bit weaker than boosting for this problem — kept only for comparison.
"""

import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from processing import run_pipeline, RANDOM_STATE

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_pipeline(model, nums_col, cat_col):
    processor = ColumnTransformer(transformers=[
        ("num", "passthrough", nums_col),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_col),
    ])
    return Pipeline(steps=[("processor", processor), ("regressor", model)])


def evaluate(name, pipeline, X_te, y_te_log, y_te):
    y_pred_log = pipeline.predict(X_te)
    # Clip in LOG space using the same physical bounds used to clean the
    # labels (30s .. 4h), then invert the log transform.
    min_log, max_log = np.log1p(30), np.log1p(4 * 60 * 60)
    y_pred_log = np.clip(y_pred_log, min_log, max_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2_log = r2_score(y_te_log, y_pred_log)
    print(f"\n--- {name} ---")
    print(f"R2 (seconds): {r2:.4f}   RMSE: {rmse:.1f}s   MAE: {mae:.1f}s   R2 (log space): {r2_log:.4f}")
    return {"name": name, "pipeline": pipeline, "r2": r2, "rmse": rmse, "mae": mae}


def main(file_path="train.csv"):
    data = run_pipeline(file_path)
    X_tr, X_te = data["X_tr"], data["X_te"]
    y_tr_log, y_te_log = data["y_tr_log"], data["y_te_log"]
    y_te = data["y_te"]
    nums_col, cat_col = data["nums_col"], data["cat_col"]
    tscv = TimeSeriesSplit(n_splits=3)

    results = []

    # --- HistGradientBoostingRegressor (small randomized search) ---
    hgb_pipeline = build_pipeline(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE), nums_col, cat_col
    )
    hgb_grid = {
        "regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__max_depth": [6, 8, 10, None],
        "regressor__max_iter": [200, 300],
        "regressor__l2_regularization": [0.0, 0.5, 1.0],
    }
    hgb_search = RandomizedSearchCV(
        hgb_pipeline, hgb_grid, n_iter=8, cv=tscv, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    hgb_search.fit(X_tr, y_tr_log)
    print("Best HGB params:", hgb_search.best_params_)
    results.append(evaluate("HistGradientBoosting", hgb_search.best_estimator_, X_te, y_te_log, y_te))

    # --- LightGBM (if available) — typically the strongest option here ---
    if HAS_LGBM:
        lgbm_pipeline = build_pipeline(
            LGBMRegressor(random_state=RANDOM_STATE, n_estimators=600,
                          learning_rate=0.05, num_leaves=64, subsample=0.8,
                          colsample_bytree=0.8),
            nums_col, cat_col,
        )
        lgbm_pipeline.fit(X_tr, y_tr_log)
        results.append(evaluate("LightGBM", lgbm_pipeline, X_te, y_te_log, y_te))
    else:
        print("\nlightgbm not installed — skipping (pip install lightgbm to include it).")

    # --- Random Forest baseline for comparison ---
    rf_pipeline = build_pipeline(
        RandomForestRegressor(n_estimators=300, max_depth=16, n_jobs=-1,
                               random_state=RANDOM_STATE),
        nums_col, cat_col,
    )
    rf_pipeline.fit(X_tr, y_tr_log)
    results.append(evaluate("RandomForest", rf_pipeline, X_te, y_te_log, y_te))

    best = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best['name']} (R2={best['r2']:.4f})")
    joblib.dump(
        {"pipeline": best["pipeline"], "nums_col": nums_col, "cat_col": cat_col},
        "model.joblib",
    )
    print("Saved best pipeline to model.joblib")
    return best


if __name__ == "__main__":
    main("train.csv")
