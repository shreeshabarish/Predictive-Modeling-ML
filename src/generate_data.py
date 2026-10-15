"""Generate a reproducible synthetic customer churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"

def main():
    rng = np.random.default_rng(42)
    n = 1200
    tenure = rng.integers(1, 73, size=n)
    monthly = np.clip(rng.normal(68, 18, size=n), 18, 125).round(2)
    tickets = np.clip(rng.poisson(1.6, size=n), 0, 9)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n, p=[0.58, 0.25, 0.17]
    )
    internet = rng.choice(["fiber", "dsl", "none"], size=n, p=[0.48, 0.38, 0.14])
    paperless = rng.choice(["yes", "no"], size=n, p=[0.68, 0.32])

    # A synthetic probability rule creates a learnable, but imperfect, target.
    logit = (
        -1.05
        - 0.035 * (tenure - 24)
        + 0.027 * (monthly - 65)
        + 0.42 * tickets
        + np.where(contract == "month-to-month", 1.05, 0.0)
        + np.where(contract == "one-year", -0.35, 0.0)
        + np.where(contract == "two-year", -0.85, 0.0)
        + np.where(internet == "fiber", 0.28, 0.0)
        + np.where(internet == "none", -0.3, 0.0)
        + np.where(paperless == "yes", 0.18, 0.0)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_tickets": tickets,
        "contract_type": contract,
        "internet_service": internet,
        "paperless_billing": paperless,
        "churn": churn,
    })
    # Add a few missing feature values to demonstrate pipeline imputation.
    for col in ["monthly_charges", "internet_service"]:
        idx = rng.choice(n, size=max(1, int(n * 0.015)), replace=False)
        df.loc[idx, col] = np.nan

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} rows: {DATA_PATH}")
    print(f"Churn rate: {df['churn'].mean():.1%}")

if __name__ == "__main__":
    main()
