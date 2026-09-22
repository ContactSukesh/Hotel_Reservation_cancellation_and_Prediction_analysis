"""
preprocessor.py
Shared preprocessing helpers used by both model_trainer.py and app.py.
This ensures identical transformations at training time and inference time.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

NUMERICAL_COLS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
]

TARGET_COL = "booking_status"

# All feature columns in order (no Booking_ID, no target)
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Possible categorical values observed in the dataset (for consistent OHE)
MEAL_PLAN_CATEGORIES = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
ROOM_TYPE_CATEGORIES = [
    "Room_Type 1", "Room_Type 2", "Room_Type 3",
    "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7",
]
MARKET_SEGMENT_CATEGORIES = [
    "Aviation", "Complementary", "Corporate", "Offline", "Online"
]

CATEGORICAL_CATEGORIES = [
    MEAL_PLAN_CATEGORIES,
    ROOM_TYPE_CATEGORIES,
    MARKET_SEGMENT_CATEGORIES,
]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted scikit-learn ColumnTransformer that:
    - Standard-scales all numerical columns
    - One-hot encodes all categorical columns (with known categories to avoid
      surprises at inference time)
    """
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(
            categories=CATEGORICAL_CATEGORIES,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def load_data(csv_path: str):
    """
    Load the CSV, drop Booking_ID, encode the target, and return (X, y, df).

    Returns
    -------
    X : pd.DataFrame  — feature columns only
    y : pd.Series     — binary target (1 = Canceled, 0 = Not_Canceled)
    df : pd.DataFrame — full cleaned dataframe (including target)
    """
    df = pd.read_csv(csv_path)

    # Drop identifier column
    df = df.drop(columns=["Booking_ID"], errors="ignore")

    # Encode target
    df[TARGET_COL] = (df[TARGET_COL] == "Canceled").astype(int)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def preprocess_input(raw_dict: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Convert a single booking dict into a numpy array ready for model.predict().

    Parameters
    ----------
    raw_dict : dict
        Keys must match FEATURE_COLS.  Missing keys default to 0 / empty string.
    preprocessor : fitted ColumnTransformer

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = {col: raw_dict.get(col, 0) for col in NUMERICAL_COLS}
    row.update({col: raw_dict.get(col, "") for col in CATEGORICAL_COLS})
    df = pd.DataFrame([row])
    return preprocessor.transform(df)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Return ordered list of feature names after transformation.
    Useful for building the feature-importance mapping.
    """
    num_names = NUMERICAL_COLS
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(CATEGORICAL_COLS)
    )
    return num_names + cat_names
