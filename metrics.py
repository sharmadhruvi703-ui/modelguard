"""Metrics and project-defined robustness summaries."""
from __future__ import annotations
import numpy as np

def robustness_score(baseline: float, perturbed: list[float]) -> float:
    """Project-defined score: 100 * average perturbed/baseline performance."""
    if baseline <= 0 or not perturbed: return 0.0
    return float(np.clip(100 * np.mean(perturbed) / baseline, 0, 100))

def degradation(baseline: float, value: float) -> float:
    return round(float(baseline - value), 12)
