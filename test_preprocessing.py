import pandas as pd
from data.generate_data import generate_dataset
from src.preprocessing import align_features, build_preprocessor, feature_columns

def test_generation_shape_and_target():
    df = generate_dataset(500, 42)
    assert len(df) == 500 and "churned" in df and df.churned.nunique() == 2

def test_generation_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(300, 3), generate_dataset(300, 3))

def test_align_restores_missing_column():
    df = generate_dataset(100, 1); cols = feature_columns(df); reduced = df.drop(columns=[cols[0]])
    aligned = align_features(reduced, cols)
    assert list(aligned.columns) == cols and aligned[cols[0]].isna().all()

def test_preprocessor_builds():
    df = generate_dataset(100, 1); transformer = build_preprocessor(df.drop(columns=["customer_id", "churned"]))
    assert transformer is not None
