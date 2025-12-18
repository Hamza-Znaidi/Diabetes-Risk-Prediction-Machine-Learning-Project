"""
Diabetes Risk Prediction - Auto-Target Version
This version automatically detects the correct target column.
"""

import os
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.figsize"] = (8, 6)


# -----------------------------
# NEW: Auto-detect target column
# -----------------------------
def detect_target_column(df, user_target=None):
    # If user manually provided a target column
    if user_target is not None:
        if user_target in df.columns:
            return user_target
        else:
            raise ValueError(f"Target column '{user_target}' not found in dataset.")

    # Common names in diabetes datasets
    common_names = ["diabetes", "Outcome", "outcome", "target"]
    for c in common_names:
        if c in df.columns:
            return c

    # Fallback: try to find a binary column
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if len(binary_cols) == 1:
        return binary_cols[0]

    raise ValueError(
        f"Unable to detect target column. "
        f"Dataset columns: {list(df.columns)} \n"
        f"Binary columns found: {binary_cols}"
    )


def load_data(path: str, target_col=None):
    df = pd.read_csv(path)

    # Auto detect target
    detected = detect_target_column(df, target_col)
    print(f"Detected target column: {detected}")

    return df, detected


# -----------------------------
# Main pipeline elements
# -----------------------------
def basic_cleaning(df):
    zero_as_nan = ["glucose_conc", "diastolic_bp", "thickness", "insulin", "bmi"]
    for col in zero_as_nan:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols)
    ])

    return preprocessor, numeric_cols


def train_models(X_train, y_train, preprocessor, seed=42):
    models = {}

    # Logistic Regression
    log_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed))
    ])

    log_grid = {
        "clf__C": [0.1, 1, 10]
    }

    log_search = GridSearchCV(log_pipe, log_grid, cv=5, scoring="recall")
    log_search.fit(X_train, y_train)
    models["logistic"] = log_search

    # Random Forest
    rf_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=seed))
    ])

    rf_grid = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [None, 10, 20],
    }

    rf_search = RandomizedSearchCV(rf_pipe, rf_grid, n_iter=5, cv=5, scoring="recall")
    rf_search.fit(X_train, y_train)
    models["random_forest"] = rf_search

    return models


def evaluate(y_test, y_pred, y_proba, outdir):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Metrics ===")
    print(metrics)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Diabetes Prediction")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Diabetes Prediction")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png")
    plt.close()

    return metrics


def main(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)

    df, target_col = load_data(args.data_path, args.target_col)
    df = basic_cleaning(df)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sizes = [len(X_train), len(X_test)]
    labels = ["Training Set (80%)", "Test Set (20%)"]

    plt.bar(labels, sizes)
    plt.title("Train-Test Split Distribution")
    plt.ylabel("Number of Samples")
    plt.savefig(outdir / "train_test_split.png")
    plt.close()

    preprocessor, numeric_cols = build_preprocessor(df.drop(columns=[target_col]))

    print("Training models…")
    models = train_models(X_train, y_train, preprocessor)

    # Select best model by recall
    best_name, best_model = max(models.items(), key=lambda x: x[1].best_score_)
    print(f"Best model: {best_name} | Recall={best_model.best_score_:.4f}")

    final_model = best_model.best_estimator_

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    evaluate(y_test, y_pred, y_proba, outdir)

    # Save
    joblib.dump(final_model, outdir / "diabetes_model.pkl")
    print(f"Model saved to: {outdir/'diabetes_model.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--target_col", type=str, default=None)  # now optional
    args = parser.parse_args()
    main(args)
