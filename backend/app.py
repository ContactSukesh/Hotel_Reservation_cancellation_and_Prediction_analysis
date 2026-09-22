"""
app.py  —  Flask REST API
Exposes the trained model and dataset statistics as a REST API.

Start with:
    python hotel-cancellation-project/backend/app.py
Listens on port 5000.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)                    # hotel-cancellation-project/
ROOT_DIR = os.path.dirname(PROJECT_DIR)                    # workspace root
CSV_PATH = os.path.join(ROOT_DIR, "Hotel Reservations.csv")
MODELS_DIR = os.path.join(THIS_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURE_IMP_PATH = os.path.join(MODELS_DIR, "feature_importance.json")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

sys.path.insert(0, THIS_DIR)
from preprocessor import preprocess_input, FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Load model at startup
model_pipeline = None
preprocessor = None

def load_model():
    global model_pipeline, preprocessor
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Run model_trainer.py first to generate the model files.")
        return False
    model_pipeline = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return True


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_pipeline is not None})


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Run model_trainer.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400

    # Validate required fields
    missing = [col for col in FEATURE_COLS if col not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # Use the preprocessor from the pipeline directly for consistency
        pipe_preprocessor = model_pipeline.named_steps["preprocessor"]
        X = preprocess_input(data, pipe_preprocessor)
        clf = model_pipeline.named_steps["classifier"]
        pred_int = int(clf.predict(X)[0])
        prob = float(clf.predict_proba(X)[0][1])
        prediction_label = "Canceled" if pred_int == 1 else "Not_Canceled"
        return jsonify({
            "prediction": prediction_label,
            "probability": round(prob, 4),
            "cancellation_risk": _risk_label(prob),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Dataset statistics endpoint
# ---------------------------------------------------------------------------
@app.route("/stats", methods=["GET"])
def stats():
    try:
        df = pd.read_csv(CSV_PATH)
        df["is_canceled"] = (df["booking_status"] == "Canceled").astype(int)

        total = int(len(df))
        canceled = int(df["is_canceled"].sum())
        not_canceled = total - canceled
        cancellation_rate = round(df["is_canceled"].mean() * 100, 2)
        avg_price = round(df["avg_price_per_room"].mean(), 2)
        avg_lead_time = round(df["lead_time"].mean(), 2)

        # Monthly cancellation trend
        monthly = (
            df.groupby("arrival_month")["is_canceled"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "canceled", "count": "total", "arrival_month": "month"})
        )
        monthly["rate"] = (monthly["canceled"] / monthly["total"] * 100).round(2)
        monthly_trend = monthly.to_dict(orient="records")

        # By market segment
        segment = (
            df.groupby("market_segment_type")["is_canceled"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "canceled", "count": "total", "market_segment_type": "segment"})
        )
        segment["rate"] = (segment["canceled"] / segment["total"] * 100).round(2)
        segment_dist = segment.to_dict(orient="records")

        # By room type
        room = (
            df.groupby("room_type_reserved")["is_canceled"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "canceled", "count": "total", "room_type_reserved": "room_type"})
        )
        room["rate"] = (room["canceled"] / room["total"] * 100).round(2)
        room_dist = room.to_dict(orient="records")

        # Lead time buckets
        bins = [0, 7, 30, 90, 180, 365, 9999]
        labels = ["0–7", "8–30", "31–90", "91–180", "181–365", "365+"]
        df["lead_bucket"] = pd.cut(df["lead_time"], bins=bins, labels=labels, right=True)
        lead_group = (
            df.groupby("lead_bucket", observed=True)["is_canceled"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "canceled", "count": "total", "lead_bucket": "bucket"})
        )
        lead_group["rate"] = (lead_group["canceled"] / lead_group["total"] * 100).round(2)
        lead_buckets = lead_group.to_dict(orient="records")

        # Meal plan distribution
        meal = (
            df.groupby("type_of_meal_plan")["is_canceled"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "canceled", "count": "total", "type_of_meal_plan": "meal_plan"})
        )
        meal["rate"] = (meal["canceled"] / meal["total"] * 100).round(2)
        meal_dist = meal.to_dict(orient="records")

        # Correlation of numeric features with target
        numeric_cols = [
            "no_of_adults", "no_of_children", "no_of_weekend_nights",
            "no_of_week_nights", "lead_time", "avg_price_per_room",
            "no_of_special_requests", "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled", "repeated_guest",
            "required_car_parking_space", "arrival_month",
        ]
        corr = df[numeric_cols + ["is_canceled"]].corr()["is_canceled"].drop("is_canceled")
        correlation = {k: round(float(v), 4) for k, v in corr.items()}

        return jsonify({
            "summary": {
                "total_bookings": total,
                "canceled": canceled,
                "not_canceled": not_canceled,
                "cancellation_rate_pct": cancellation_rate,
                "avg_price_per_room": avg_price,
                "avg_lead_time_days": avg_lead_time,
            },
            "monthly_trend": monthly_trend,
            "segment_distribution": segment_dist,
            "room_type_distribution": room_dist,
            "lead_time_buckets": lead_buckets,
            "meal_plan_distribution": meal_dist,
            "correlation_with_cancellation": correlation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Feature importance endpoint
# ---------------------------------------------------------------------------
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    if not os.path.exists(FEATURE_IMP_PATH):
        return jsonify({"error": "feature_importance.json not found. Run model_trainer.py first."}), 404
    with open(FEATURE_IMP_PATH) as f:
        data = json.load(f)
    # Return top 20
    importances = data.get("importances", {})
    top20 = dict(list(importances.items())[:20])
    return jsonify({
        "model": data.get("model", "unknown"),
        "top_features": top20,
    })


# ---------------------------------------------------------------------------
# Model comparison endpoint
# ---------------------------------------------------------------------------
@app.route("/model-comparison", methods=["GET"])
def model_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return jsonify({"error": "model_comparison.json not found. Run model_trainer.py first."}), 404
    with open(COMPARISON_PATH) as f:
        data = json.load(f)
    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
