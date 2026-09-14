# Data

The dataset is generated locally and is intentionally not committed. Run:

```bash
python data/generate_data.py --output data/customer_churn.csv --rows 9000 --seed 42
```

The generator creates account, usage, service, billing, and satisfaction variables with a binary churn target. It is synthetic and should not be treated as a real customer dataset.
