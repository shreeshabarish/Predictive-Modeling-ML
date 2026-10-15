"""Interactive prediction using the saved best model."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

def ask_choice(prompt, choices):
    choices_text = "/".join(choices)
    while True:
        value = input(f"{prompt} ({choices_text}): ").strip().lower()
        if value in choices:
            return value
        print(f"Please enter one of: {choices_text}")

def main():
    if not MODEL_PATH.exists():
        print("Model not found. First run: python src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"Customer Churn Predictor | Model: {metadata['best_model']}")
    print("This is a demonstration using synthetic training data.\n")

    while True:
        try:
            tenure = float(input("Tenure in months (1–72): "))
            monthly = float(input("Monthly charges (18–125): "))
            tickets = int(input("Number of support tickets (0–9): "))
            if not 1 <= tenure <= 72 or not 18 <= monthly <= 125 or not 0 <= tickets <= 9:
                print("Please enter values within the suggested ranges.\n")
                continue
            break
        except ValueError:
            print("Enter valid numbers and try again.\n")

    contract = ask_choice("Contract type", ["month-to-month", "one-year", "two-year"])
    internet = ask_choice("Internet service", ["fiber", "dsl", "none"])
    paperless = ask_choice("Paperless billing", ["yes", "no"])

    sample = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_tickets": tickets,
        "contract_type": contract,
        "internet_service": internet,
        "paperless_billing": paperless
    }])
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])
    print("\nPrediction:", "Likely to churn" if prediction == 1 else "Likely to stay")
    print(f"Estimated churn probability: {probability:.1%}")
    print("Reminder: this is a learning demo, not a real business decision tool.")

if __name__ == "__main__":
    main()
