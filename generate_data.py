"""Generate a reproducible synthetic customer-churn dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(rows: int = 9000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic but structured customer dataset."""
    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    age = np.clip(rng.normal(42, 13, rows), 18, 85).round().astype(int)
    monthly_charges = np.clip(rng.normal(72, 25, rows), 20, 180).round(2)
    data = pd.DataFrame({
        "customer_id": [f"C{n:06d}" for n in range(rows)],
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_tickets": rng.poisson(1.5, rows),
        "monthly_logins": np.clip(rng.normal(16, 7, rows), 0, 45).round(1),
        "avg_session_minutes": np.clip(rng.normal(31, 12, rows), 3, 90).round(1),
        "satisfaction_score": np.clip(rng.normal(7.1, 1.8, rows), 1, 10).round(1),
        "payment_failures": rng.poisson(0.35, rows),
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55, .28, .17]),
        "internet_service": rng.choice(["DSL", "Fiber", "None"], rows, p=[.35, .52, .13]),
        "region": rng.choice(["North", "South", "East", "West"], rows),
        "support_plan": rng.choice(["Basic", "Premium", "None"], rows, p=[.43, .22, .35]),
        "device_type": rng.choice(["Mobile", "Desktop", "Tablet"], rows, p=[.51, .34, .15]),
    })
    logit = (
        -2.1 + 0.95 * (data.monthly_charges > 95) + 0.85 * (data.contract_type == "Month-to-month")
        - 0.035 * data.tenure_months + 0.28 * data.support_tickets
        - 0.22 * data.satisfaction_score + 0.42 * data.payment_failures
        + 0.35 * (data.internet_service == "Fiber") - 0.18 * (data.support_plan == "Premium")
        + 0.018 * (data.monthly_logins < 8) * data.monthly_charges
        + rng.normal(0, 0.35, rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    data["churned"] = rng.binomial(1, probability)
    # Missingness is present but modest; the experiment suite adds controlled stress.
    for col, rate in [("satisfaction_score", .025), ("avg_session_minutes", .018), ("support_plan", .012)]:
        mask = rng.random(rows) < rate
        data.loc[mask, col] = np.nan
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/customer_churn.csv"))
    parser.add_argument("--rows", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Wrote {len(generate_dataset(args.rows, args.seed)):,} rows to {args.output}")

if __name__ == "__main__":
    main()
