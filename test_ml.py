# test_ml.py
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# Ensure repo root is on the path (so `ml` package imports work)
sys.path.append(str(Path(__file__).resolve().parent))

from ml.data import process_data  # noqa: E402
from ml.model import train_model, inference, compute_model_metrics  # noqa: E402


# -------------------- Fixtures --------------------
@pytest.fixture(scope="session")
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    assert data_path.exists(), f"Dataset not found at {data_path}"
    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def processed_splits(raw_df, cat_features):
    # Simple deterministic 80/20 split by index parity (avoids extra imports)
    idx = np.arange(len(raw_df))
    train_df = raw_df[idx % 5 != 0].reset_index(drop=True)
    test_df = raw_df[idx % 5 == 0].reset_index(drop=True)

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


# -------------------- Tests --------------------
def test_process_data_outputs_are_arrays(processed_splits):
    X_train, y_train, X_test, y_test, *_ = processed_splits
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_train_model_returns_expected_type(processed_splits):
    X_train, y_train, *_ = processed_splits
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")
    assert isinstance(model, RandomForestClassifier)


def test_inference_len_matches(processed_splits):
    X_train, y_train, X_test, y_test, *_ = processed_splits
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(y_test)


def test_metrics_within_expected_range(processed_splits):
    X_train, y_train, X_test, y_test, *_ = processed_splits
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, f1 = compute_model_metrics(y_test, preds)
    for m in (p, r, f1):
        assert 0.0 <= m <= 1.0


def test_train_test_sizes_reasonable(processed_splits):
    X_train, _, X_test, _, *_ = processed_splits
    n_total = len(X_train) + len(X_test)
    assert n_total >= 1000
    ratio = len(X_train) / n_total
    assert 0.70 <= ratio <= 0.90
