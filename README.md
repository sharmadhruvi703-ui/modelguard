# ModelGuard

## Overview

**ModelGuard** is an independent, reproducible Python project that tests how reliably machine-learning models behave when incoming data becomes messy, incomplete, or different from the data used for training. It uses a synthetic customer-churn dataset so the entire experiment can run locally without downloads or private data.

## Why This Project?

A model can score well on a clean holdout set and still be vulnerable to ordinary data-pipeline problems. This project turns that concern into a small, inspectable experiment framework: establish a clean baseline, apply controlled perturbations, measure the change, and generate a report from the measured results.

## Problem and approach

The central question is: **how fragile is a model when production data stops looking like training data?** The project generates structured data, splits it into train and test sets, trains two pipelines, evaluates a clean baseline, and then reruns evaluation for each perturbation at 5%, 15%, and 30% severity.

## Dataset

The generator creates 9,000 synthetic customer records with numerical usage and billing variables, categorical service information, account identifiers, modest natural missingness, and an imbalanced churn target. Relationships are probabilistic rather than deterministic. The data is suitable for experimentation only; it is not a representation of any real organization or customer population.

## Models

The baseline comparison includes class-weighted logistic regression and a class-weighted random forest. Both models use a scikit-learn `Pipeline`. Numeric fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded with `handle_unknown="ignore"`.

## Data perturbations

The suite includes missing values, numeric outliers, additive feature noise, distribution shift, duplicate records, unseen categorical values, missing feature columns, and feature-scale changes. Each function copies its input, accepts a severity and seed, and returns a modified DataFrame.

## Experimental setup

The default run uses seed 42, a stratified 75/25 split, three severities, and five classification metrics: accuracy, precision, recall, F1, and ROC-AUC. Results are written to CSV and JSON, while Matplotlib produces two PNG charts.

## Installation

```bash
git clone <your-repository-url>
cd modelguard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

```bash
python data/generate_data.py --output data/customer_churn.csv --rows 9000 --seed 42
python experiments/run_experiments.py
pytest -q
```

If the CSV does not exist, the experiment runner creates the default dataset in memory. Explicit generation is recommended when you want to inspect or version the input artifact.

## Results

The experiment runner calculates all metrics; it does not use hardcoded example values. Inspect `experiments/results/metrics.csv`, `experiments/results/baseline.json`, and the charts after running it. The generated narrative report is at `reports/model_reliability_report.md`.

## Robustness score

This project-defined summary is:

> `Robustness Score = 100 × mean(F1 across all perturbed runs) / clean baseline F1`

It is intentionally described as a comparison aid rather than a scientifically authoritative reliability metric. A score near 100 means less average F1 degradation within this specific experiment grid; it does not guarantee production performance.

## Example usage

```python
from data.generate_data import generate_dataset
from src.evaluation import split_data, train_and_evaluate
from src.perturbations import add_missing_values

df = generate_dataset(rows=1000, seed=7)
train, test = split_data(df, seed=7)
print(train_and_evaluate(train, test))
print(train_and_evaluate(train, add_missing_values(test, severity=.15, seed=7)))
```

## Findings and limitations

The generated report identifies the most damaging scenario for the actual run and compares both models. Findings are specific to the synthetic data-generating process, perturbation definitions, model settings, and random seed. The study does not establish universal claims about model reliability. It also does not cover temporal drift, calibration, monitoring latency, compositional failures, retraining strategy, or real production data.

## Future improvements

Useful extensions include time-ordered validation, configurable perturbation compositions, calibration curves, confidence intervals from repeated seeds, a command-line configuration file, and support for user-supplied tabular datasets with explicit schema validation.

## Project structure

```text
modelguard/
├── data/                  # Synthetic dataset generator
├── src/                   # Preprocessing, models, perturbations, evaluation
├── experiments/           # Experiment runner and generated artifacts
├── reports/               # Programmatically generated Markdown report
├── notebooks/             # Optional exploratory notebook
├── tests/                 # pytest coverage of core behavior
└── examples/              # Small usage example
```

## License

MIT.
