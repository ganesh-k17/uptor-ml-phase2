import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "MedHouseVal"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "california_model.joblib"
SCALER_PATH = MODEL_DIR / "california_scaler.joblib"
FEATURES_PATH = MODEL_DIR / "california_features.joblib"

MODEL = None
SCALER = None
FEATURE_COLUMNS = None


def load_california_data():
    housing_bundle = fetch_california_housing(as_frame=True)
    df = housing_bundle.frame.copy()
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def train_california_model():
    global MODEL, SCALER, FEATURE_COLUMNS

    x, y = load_california_data()
    FEATURE_COLUMNS = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation="relu",
        max_iter=500,
        random_state=42,
        verbose=False,
    )

    model.fit(x_train_scaled, y_train)

    MODEL = model
    SCALER = scaler

    joblib.dump(MODEL, MODEL_PATH)
    joblib.dump(SCALER, SCALER_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    return MODEL, SCALER, FEATURE_COLUMNS


def get_california_model():
    global MODEL, SCALER, FEATURE_COLUMNS

    if MODEL is None or SCALER is None:
        if all(path.exists() for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
            MODEL = joblib.load(MODEL_PATH)
            SCALER = joblib.load(SCALER_PATH)
            FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
        else:
            train_california_model()

    return MODEL, SCALER, FEATURE_COLUMNS


def prepare_input_frame(data, feature_columns):
    if isinstance(data, dict):
        frame = pd.DataFrame([data], columns=feature_columns)
    elif isinstance(data, (list, tuple, np.ndarray)):
        if len(data) != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} values")
        frame = pd.DataFrame([data], columns=feature_columns)
    else:
        raise TypeError("Input must be a dictionary or a list/tuple of feature values")

    frame = frame.reindex(columns=feature_columns)
    return frame.astype(float)


def predict_california_value(data):
    model, scaler, feature_columns = get_california_model()

    frame = prepare_input_frame(data, feature_columns)

    scaled = scaler.transform(frame)
    prediction = model.predict(scaled)[0]
    return float(prediction)


if __name__ == "__main__":
    get_california_model()
    print("California housing model ready")
    print("Feature columns:", FEATURE_COLUMNS)