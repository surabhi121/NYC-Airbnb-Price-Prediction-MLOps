"""Shared data loading, cleaning and model-building code.

Every training entry point (train.py, track_experiments.py,
orchestrate_training.py, scripts/ci_seed_model.py) imports from here so the
cleaning rules and preprocessing can never drift apart.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path("data/AB_NYC_2019.csv")

TARGET = "price"
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
# neighbourhood has 221 distinct values: a high-cardinality categorical.
# One-hot still works; handle_unknown="ignore" zeroes any unseen value at predict time.
CATEGORICAL_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Cleaning thresholds (see implementation.md, Tasks 2 and 3):
# - price == 0: 11 rows in the 2019 file, data errors rather than free listings.
# - price > 800: ~the 99th percentile ($799); 420 extreme outliers up to $10,000.
MAX_PRICE = 800
RANDOM_STATE = 42


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV. With no path, honour $DATA_PATH (CI uses the sample)."""
    path = path or os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # reviews_per_month is missing exactly when number_of_reviews == 0:
    # no reviews means a review rate of 0, not the average rate.
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
    df = df[(df[TARGET] > 0) & (df[TARGET] <= MAX_PRICE)]
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """80/20 split -> (X_train, X_test, y_train, y_test)."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE
    )


def build_model(regressor) -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    # Price is heavily right-skewed, so the regressor is fit on log1p(price).
    # TransformedTargetRegressor applies expm1 inside .predict(), so every
    # caller (evaluation, the API, tests) gets dollars back automatically.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """RMSE / MAE / R² on the original dollar scale."""
    predictions = model.predict(X)
    return {
        "rmse": float(root_mean_squared_error(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }
