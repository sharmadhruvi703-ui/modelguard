"""Model fitting and metric calculation."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .models import build_models
from .preprocessing import TARGET, ID_COLUMNS, align_features, feature_columns

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def evaluate_predictions(y_true, predictions, probabilities):
    return {"accuracy": accuracy_score(y_true, predictions), "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0), "f1": f1_score(y_true, predictions, zero_division=0), "roc_auc": roc_auc_score(y_true, probabilities)}

def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, seed=42):
    expected = feature_columns(train_df)
    x_train = align_features(train_df, expected)
    x_test = align_features(test_df, expected)
    y_train, y_test = train_df[TARGET], test_df[TARGET]
    results = {}
    for name, model in build_models(x_train).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        results[name] = evaluate_predictions(y_test, predictions, probabilities)
    return results

def split_data(df, seed=42):
    train, test = train_test_split(df, test_size=.25, stratify=df[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)
