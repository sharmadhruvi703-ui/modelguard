"""Model factory functions."""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor


def build_models(frame):
    preprocessor = build_preprocessor(frame)
    return {
        "logistic_regression": Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        "random_forest": Pipeline([("preprocess", build_preprocessor(frame)), ("model", RandomForestClassifier(n_estimators=180, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))]),
    }
