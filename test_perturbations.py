import pandas as pd
from data.generate_data import generate_dataset
from src.perturbations import *

def test_perturbations_do_not_mutate_input():
    df=generate_dataset(200, 2); original=df.copy(deep=True)
    for fn in [add_missing_values, add_outliers, add_noise, introduce_distribution_shift, create_duplicates, corrupt_categories, remove_features, change_feature_scale]:
        changed=fn(df, severity=.15, seed=9); pd.testing.assert_frame_equal(df, original); assert len(changed) >= len(df) if fn is create_duplicates else len(changed) == len(df)

def test_unknown_categories_are_created():
    changed=corrupt_categories(generate_dataset(200, 2), .25, 9)
    assert "__unseen_category__" in changed.to_numpy()

def test_missing_values_increase():
    df=generate_dataset(500, 2); changed=add_missing_values(df, .2, 9)
    assert changed.isna().sum().sum() > df.isna().sum().sum()
