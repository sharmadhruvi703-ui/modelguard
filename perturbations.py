"""Controlled, reproducible data perturbations."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _copy(df): return df.copy(deep=True)
def _features(df): return [c for c in df.select_dtypes(include="number").columns if c != "churned"]

def add_missing_values(df, severity=.10, seed=42):
    out, rng = _copy(df), np.random.default_rng(seed)
    cols = [c for c in out.columns if c not in ["customer_id", "churned"]]
    for col in cols:
        mask = rng.random(len(out)) < severity
        out[col] = out[col].mask(mask, np.nan)
    return out

def add_outliers(df, severity=.10, seed=42):
    out, rng = _copy(df), np.random.default_rng(seed)
    cols = _features(out)
    count = max(1, int(len(out) * severity))
    for col in cols:
        out[col] = out[col].astype(float)
        idx = rng.choice(len(out), count, replace=False)
        values = out.iloc[idx][col].astype(float)
        out.loc[out.index[idx], col] = values * rng.uniform(3, 6, count)
    return out

def add_noise(df, severity=.10, seed=42):
    out, rng = _copy(df), np.random.default_rng(seed)
    for col in _features(out):
        scale = out[col].std(skipna=True) or 1
        out[col] = out[col] + rng.normal(0, severity * scale, len(out))
    return out

def introduce_distribution_shift(df, severity=.10, seed=42):
    out, rng = _copy(df), np.random.default_rng(seed)
    for col in _features(out):
        scale = out[col].std(skipna=True) or 1
        out[col] = out[col] + severity * scale * 2
    for col in [c for c in ["contract_type", "internet_service", "region", "support_plan", "device_type"] if c in out]:
        categories = out[col].dropna().unique()
        if len(categories) > 1:
            out[col] = out[col].map(lambda x: rng.choice(categories) if rng.random() < severity else x)
    return out

def create_duplicates(df, severity=.10, seed=42):
    out = _copy(df)
    count = max(1, int(len(out) * severity))
    duplicates = out.sample(count, random_state=seed)
    return pd.concat([out, duplicates], ignore_index=True)

def corrupt_categories(df, severity=.10, seed=42):
    out, rng = _copy(df), np.random.default_rng(seed)
    cols = [c for c in out.select_dtypes(exclude="number").columns if c != "customer_id"]
    for col in cols:
        mask = rng.random(len(out)) < severity
        out.loc[mask, col] = "__unseen_category__"
    return out

def remove_features(df, severity=.10, seed=42):
    out = _copy(df)
    candidates = [c for c in out.columns if c not in ["customer_id", "churned"]]
    count = max(1, int(round(len(candidates) * severity)))
    rng = np.random.default_rng(seed)
    return out.drop(columns=rng.choice(candidates, count, replace=False).tolist())

def change_feature_scale(df, severity=.10, seed=42):
    out = _copy(df)
    factor = 1 + 5 * severity
    for col in _features(out): out[col] = out[col] * factor
    return out

PERTURBATIONS = {
    "missing_values": add_missing_values, "outliers": add_outliers, "feature_noise": add_noise,
    "distribution_shift": introduce_distribution_shift, "duplicates": create_duplicates,
    "corrupt_categories": corrupt_categories, "missing_features": remove_features,
    "feature_scale_change": change_feature_scale,
}
