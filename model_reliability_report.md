# ModelGuard Reliability Report

This report was generated programmatically from the experiment results. It describes an engineering exploration on synthetic customer data, not a universal claim about model behavior.

## Baseline results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.721 | 0.142 | 0.731 | 0.238 | 0.787 |
| random_forest | 0.924 | 0.291 | 0.187 | 0.227 | 0.774 |

## Perturbations and degradation

The table below reports F1 degradation from each model's clean baseline. Positive values indicate a drop.

| Scenario | Severity | Logistic regression degradation | Random forest degradation |
|---|---:|---:|---:|
| missing_values | 5% | 0.013 | 0.036 |
| missing_values | 15% | 0.024 | 0.057 |
| missing_values | 30% | 0.051 | 0.052 |
| outliers | 5% | 0.039 | -0.015 |
| outliers | 15% | 0.057 | 0.014 |
| outliers | 30% | 0.070 | 0.081 |
| feature_noise | 5% | 0.004 | 0.026 |
| feature_noise | 15% | 0.003 | 0.044 |
| feature_noise | 30% | 0.013 | 0.064 |
| distribution_shift | 5% | -0.004 | 0.010 |
| distribution_shift | 15% | 0.007 | 0.037 |
| distribution_shift | 30% | -0.002 | 0.067 |
| duplicates | 5% | -0.000 | 0.009 |
| duplicates | 15% | -0.001 | 0.017 |
| duplicates | 30% | -0.002 | 0.011 |
| corrupt_categories | 5% | 0.008 | 0.009 |
| corrupt_categories | 15% | 0.009 | 0.004 |
| corrupt_categories | 30% | 0.015 | 0.016 |
| missing_features | 5% | 0.057 | -0.007 |
| missing_features | 15% | 0.059 | 0.000 |
| missing_features | 30% | 0.046 | -0.024 |
| feature_scale_change | 5% | 0.001 | 0.041 |
| feature_scale_change | 15% | -0.008 | 0.186 |
| feature_scale_change | 30% | -0.005 | 0.184 |

## Model comparison

The most damaging perturbation by mean F1 was **feature_scale_change** in this run. Results vary with the chosen seed, generated data, and model settings.

## Robustness score

This project-defined score is `100 × mean(perturbed F1) / baseline F1`, averaged across all tested perturbations and severities. It is a compact comparison aid, not a calibrated reliability guarantee.

| Model | Robustness score |
|---|---:|
| logistic_regression | 92.1/100 |
| random_forest | 83.1/100 |

## Key observations

- The baseline F1 values were measured before the controlled perturbations.
- The ranking of perturbations is dataset- and model-dependent; **feature_scale_change** had the lowest mean F1 here.
- Preprocessing handled missing values and unseen categories without changing the experiment code.
- Duplicate rows and missing features represent data-pipeline problems rather than changes to the underlying customer behavior.

## Limitations

The data is synthetic, the perturbations are isolated rather than compositional, and the robustness score only summarizes this experiment grid. Production monitoring, temporal validation, calibration, and causal analysis are outside the scope of this small project.
