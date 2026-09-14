from data.generate_data import generate_dataset
from src.evaluation import split_data, train_and_evaluate
from src.metrics import degradation, robustness_score
from src.perturbations import corrupt_categories, remove_features

def test_training_and_metrics():
    train,test=split_data(generate_dataset(500, 4), 4); result=train_and_evaluate(train,test)
    assert set(result) == {"logistic_regression", "random_forest"}
    assert all(0 <= value <= 1 for values in result.values() for value in values.values())

def test_unseen_categories_and_missing_features_are_evaluable():
    train,test=split_data(generate_dataset(500, 4), 4)
    assert train_and_evaluate(train, corrupt_categories(test, .2, 4))
    assert train_and_evaluate(train, remove_features(test, .2, 4))

def test_metric_calculation():
    assert degradation(.8, .6) == .2
    assert robustness_score(.8, [.8, .6]) == 87.5
