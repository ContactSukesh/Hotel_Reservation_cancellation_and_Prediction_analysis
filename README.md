# Hotel Booking Cancellation Analysis & Prediction

A fully Python-based web application for analysing hotel reservation data and predicting booking cancellations using Machine Learning.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Plotly |
| Backend | Flask REST API |
| ML | scikit-learn (LR, Random Forest) + XGBoost |
| Data | pandas, numpy |

---

## Project Structure

```
hotel-cancellation-project/
├── Hotel Reservations.csv          ← source dataset (at workspace root)
├── backend/
│   ├── app.py                      ← Flask REST API (port 5000)
│   ├── model_trainer.py            ← training pipeline
│   ├── preprocessor.py             ← shared preprocessing helpers
│   └── models/
│       ├── best_model.pkl          ← saved best model (generated)
│       ├── preprocessor.pkl        ← saved column transformer (generated)
│       └── feature_importance.json ← feature scores (generated)
├── frontend/
│   └── app.py                      ← Streamlit dashboard (port 8501)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd hotel-cancellation-project
pip install -r requirements.txt
```

### 2. Train the model

Run from the **project root** (the workspace root where `Hotel Reservations.csv` lives):

```bash
python hotel-cancellation-project/backend/model_trainer.py
```

This will:
- Train Logistic Regression, Random Forest, and XGBoost classifiers
- Print a comparison table of accuracy, precision, recall, F1, and ROC-AUC
- Save `backend/models/best_model.pkl`, `backend/models/preprocessor.pkl`, and `backend/models/feature_importance.json`

### 3. Start the Flask API

```bash
python hotel-cancellation-project/backend/app.py
```

API runs at `http://localhost:5000`

### 4. Start the Streamlit dashboard

In a **separate terminal**:

```bash
streamlit run hotel-cancellation-project/frontend/app.py
```

Dashboard opens at `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/stats` | Dataset-level statistics |
| GET | `/feature-importance` | Top feature importances |
| POST | `/predict` | Predict cancellation for a booking |

### POST /predict — example request

```json
{
  "no_of_adults": 2,
  "no_of_children": 0,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 2,
  "type_of_meal_plan": "Meal Plan 1",
  "required_car_parking_space": 0,
  "room_type_reserved": "Room_Type 1",
  "lead_time": 50,
  "arrival_year": 2018,
  "arrival_month": 6,
  "arrival_date": 15,
  "market_segment_type": "Online",
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 0,
  "avg_price_per_room": 100.0,
  "no_of_special_requests": 1
}
```

### POST /predict — example response

```json
{
  "prediction": "Canceled",
  "probability": 0.73
}
```

---

## Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | KPI cards — total bookings, cancellation rate, avg price, avg lead time |
| **EDA** | Cancellation by month, market segment, lead time, room type, correlation heatmap |
| **Model Insights** | Feature importance chart, model comparison table |
| **Predict** | Live prediction form with probability gauge |

---

## Troubleshooting

**`FileNotFoundError: Hotel Reservations.csv`**
- Ensure you run the trainer from the workspace root directory, not from inside `hotel-cancellation-project/`
- The CSV must be at `c:\Users\Sukesh\Desktop\Final Project\Hotel Reservations.csv`

**`best_model.pkl not found`**
- Run `model_trainer.py` before starting the Flask API

**Port already in use**
- Flask: change port in `backend/app.py` → `app.run(port=5001)`
- Streamlit: `streamlit run frontend/app.py --server.port 8502`

**Streamlit shows "API unreachable" banner**
- Make sure the Flask API is running first (`python backend/app.py`)
