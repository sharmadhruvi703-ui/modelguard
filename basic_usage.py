"""Minimal ModelGuard usage example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_data import generate_dataset
from src.evaluation import split_data, train_and_evaluate
from src.perturbations import add_missing_values

dataset = generate_dataset(rows=1000, seed=7)
train, test = split_data(dataset, seed=7)
print("Clean:", train_and_evaluate(train, test))
print("Missing values:", train_and_evaluate(train, add_missing_values(test, severity=.15, seed=7)))
