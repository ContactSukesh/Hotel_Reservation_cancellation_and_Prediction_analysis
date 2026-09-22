"""
model_trainer.py
Train Logistic Regression, Random Forest, and XGBoost on the hotel reservations
dataset.  The best model (by ROC-AUC) is saved to backend/models/.

Run from the workspace root:
    python hotel-cancellation-project/backend/model_trainer.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed — skipping XGBClassifier.")

# Resolve paths relative to this file's location so the script works from any cwd
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)                    # hotel-cancellation-project/
ROOT_DIR = os.path.dirname(PROJECT_DIR)                    # workspace root
CSV_PATH = os.path.join(ROOT_DIR, "Hotel Reservations.csv")
MODELS_DIR = os.path.join(THIS_DIR, "models")

# Add backend dir to path so we can import preprocessor
sys.path.insert(0, THIS_DIR)
from preprocessor import build_preprocessor, load_data, get_feature_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def extract_importances(pipeline: Pipeline, feature_names: list) -> dict:
    """
    Extract feature importances from the classifier step of the pipeline.
    Works for LR (coef_), RF / XGB (feature_importances_).
    """
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = np.abs(clf.coef_[0])
    else:
        return {}

    # Sort descending
    paired = sorted(
        zip(feature_names, scores.tolist()),
        key=lambda x: x[1], reverse=True
    )
    return {name: score for name, score in paired}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("Hotel Booking Cancellation — Model Trainer")
    print(f"{'='*60}")
    print(f"Loading data from: {CSV_PATH}")

    X, y, _ = load_data(CSV_PATH)
    print(f"Dataset shape: {X.shape}, Cancellation rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Build preprocessor once (shared across all models)
    preprocessor = build_preprocessor()

    # Fit preprocessor on training data to get feature names
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )

    results = {}
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"Training {name} ...")
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        trained_pipelines[name] = pipeline
        print(f"  ROC-AUC: {metrics['roc_auc']}  |  F1: {metrics['f1']}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
    print(f"{'-'*60}")
    for name, m in results.items():
        print(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>7.4f} {m['roc_auc']:>9.4f}"
        )
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Select best model by ROC-AUC
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = trained_pipelines[best_name]
    print(f"Best model: {best_name}  (ROC-AUC = {results[best_name]['roc_auc']})")

    # ------------------------------------------------------------------
    # Save best model pipeline
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved model pipeline -> {model_path}")

    # Save standalone preprocessor (for the Flask API's preprocess_input path)
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(best_pipeline.named_steps["preprocessor"], prep_path)
    print(f"Saved preprocessor   -> {prep_path}")

    # ------------------------------------------------------------------
    # Save feature importances
    # ------------------------------------------------------------------
    importances = extract_importances(
        best_pipeline,
        get_feature_names(best_pipeline.named_steps["preprocessor"])
    )
    fi_path = os.path.join(MODELS_DIR, "feature_importance.json")
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump({"model": best_name, "importances": importances}, f, indent=2)
    print(f"Saved feature importances -> {fi_path}")

    # Save model comparison results
    comparison_path = os.path.join(MODELS_DIR, "model_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Saved model comparison -> {comparison_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
