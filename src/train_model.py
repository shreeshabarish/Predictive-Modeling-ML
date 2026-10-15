"""Train and evaluate multiple supervised classification models."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service", "paperless_billing"]
TARGET = "churn"

def build_preprocessor(scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

def main():
    for folder in (MODEL_DIR, FIG_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\nRun `python src/generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target column 'churn' must contain both 0 and 1.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1500, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=8, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=250, max_depth=10, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
    }

    rows = []
    fitted = {}
    reports = []
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        rows.append(metrics)
        reports.append(
            f"\n{'=' * 72}\n{name}\n{'=' * 72}\n"
            + classification_report(
                y_test, predictions, target_names=["Stays (0)", "Churns (1)"],
                zero_division=0
            )
        )

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted stays", "Predicted churns"],
            yticklabels=["Actual stays", "Actual churns"], ax=ax
        )
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        fig.savefig(FIG_DIR / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close(fig)

        RocCurveDisplay.from_predictions(
            y_test, probabilities, name=f"{name} (AUC={metrics['roc_auc']:.3f})",
            ax=roc_ax
        )

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    (REPORT_DIR / "classification_reports.txt").write_text(
        "\n".join(reports), encoding="utf-8"
    )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    roc_ax.set_title("ROC Curve Comparison")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend(loc="lower right")
    roc_ax.grid(alpha=0.25)
    roc_fig.tight_layout()
    roc_fig.savefig(FIG_DIR / "roc_curves.png", dpi=160)
    plt.close(roc_fig)

    # Feature importance from the best Random Forest pipeline, if available.
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted[best_name]
    if best_name == "Random Forest":
        preprocessor = best_model.named_steps["preprocessor"]
        classifier = best_model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importances
        }).sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
        ax.set_title("Top Feature Importances — Random Forest")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "feature_importance.png", dpi=160)
        plt.close(fig)
        importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    # Save the highest ROC-AUC model as the reusable final model.
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "positive_class": "churn = 1",
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "data_note": "Synthetic educational dataset; not real customer data.",
        "test_size": 0.20,
        "random_state": 42
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nMODEL COMPARISON (sorted by ROC-AUC)")
    print(metrics_df.to_string(index=False, formatters={
        key: (lambda x: f"{x:.3f}") for key in
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    }))
    print(f"\nBest model: {best_name}")
    print(f"Reports saved to: {REPORT_DIR}")
    print(f"Charts saved to: {FIG_DIR}")
    print(f"Reusable model saved to: {MODEL_DIR / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
