# Predictive Modeling Using Machine Learning

A beginner-friendly supervised machine learning project that predicts whether a customer will leave a service (binary classification). It compares:
- Logistic Regression
- Decision Tree
- Random Forest

It includes data preparation, train/test split, model evaluation, confusion matrices, ROC curves, feature importance, saved model, and a simple command-line prediction tool.

## Why classification?
The task mentions confusion matrices and ROC curves, which are standard classification evaluation tools. This project therefore uses a binary classification dataset. The included synthetic dataset is generated for learning and demonstration; it is **not real customer data** and results should not be presented as real-world business findings.

## Requirements
- Python 3.10+
- Internet is not required after installing packages; all project data is included.

## Setup in VS Code (Windows)
1. Extract the ZIP file.
2. Open the `Predictive_Modeling_ML_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install libraries:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run the full project
From the project root:
```powershell
python src/train_model.py
```

This creates:
- `outputs/reports/model_metrics.csv`
- `outputs/reports/classification_reports.txt`
- `outputs/figures/confusion_matrix_*.png`
- `outputs/figures/roc_curves.png`
- `models/best_model.joblib`
- `models/model_metadata.json`

## Try a prediction
After training:
```powershell
python src/predict.py
```
Follow the prompts. Enter:
- tenure_months: e.g. `18`
- monthly_charges: e.g. `75`
- support_tickets: e.g. `3`
- contract_type: `month-to-month`, `one-year`, or `two-year`
- internet_service: `fiber`, `dsl`, or `none`
- paperless_billing: `yes` or `no`

## Project structure
```text
Predictive_Modeling_ML_Project/
├── data/
│   └── customer_churn.csv
├── models/                 # trained model saved here
├── outputs/
│   ├── figures/            # charts created by training
│   └── reports/            # metrics and classification report
├── src/
│   ├── generate_data.py    # optional data regeneration
│   ├── train_model.py     # train, compare and evaluate models
│   └── predict.py         # interactive prediction
├── requirements.txt
├── .gitignore
└── README.md
```

## What to explain in your internship submission
1. **Problem:** Predict customer churn (whether a customer leaves).
2. **Type:** Supervised binary classification.
3. **Features:** tenure, monthly charges, support tickets, contract type, internet service, paperless billing.
4. **Target:** `churn` (0 = stays, 1 = leaves).
5. **Preprocessing:** Missing values are handled in a pipeline; categorical features are one-hot encoded and numerical features are scaled for Logistic Regression.
6. **Models:** Logistic Regression, Decision Tree, Random Forest.
7. **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves.
8. **Reproducibility:** A fixed random seed is used for data generation and train/test splitting.

## Important note
The included dataset is synthetic and intended to demonstrate a complete workflow. For a real-world project, replace it with a legitimate dataset and document its source, permissions, and limitations.
