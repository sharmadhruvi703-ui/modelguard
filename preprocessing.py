"""Preprocessing utilities shared by training and evaluation."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"
ID_COLUMNS = ["customer_id"]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS + [TARGET]]


def align_features(frame: pd.DataFrame, expected_features: Iterable[str]) -> pd.DataFrame:
    """Return a copy with the expected schema; absent columns become all-NaN."""
    result = frame.copy()
    for col in expected_features:
        if col not in result:
            result[col] = np.nan
    return result[list(expected_features)]


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")
