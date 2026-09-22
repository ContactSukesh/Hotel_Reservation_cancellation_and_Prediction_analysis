"""
frontend/app.py  —  Streamlit Dashboard
Hotel Booking Cancellation Analysis & Prediction

Start with:
    streamlit run hotel-cancellation-project/frontend/app.py
Dashboard runs at http://localhost:8501
Flask API must be running at http://localhost:5000
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:5000"

st.set_page_config(
    page_title="Hotel Cancellation Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .kpi-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px 16px;
        text-align: center;
    }
    .kpi-title { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1f2328; }
    .kpi-delta { font-size: 12px; color: #57606a; }
    .risk-high   { background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:20px; font-weight:600; }
    .risk-medium { background:#fef9c3; color:#854d0e; padding:6px 14px; border-radius:20px; font-weight:600; }
    .risk-low    { background:#dcfce7; color:#166534; padding:6px 14px; border-radius:20px; font-weight:600; }
    .section-title { font-size:18px; font-weight:700; color:#1f2328; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def fetch_feature_importance():
    try:
        r = requests.get(f"{API_BASE}/feature-importance", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def fetch_model_comparison():
    try:
        r = requests.get(f"{API_BASE}/model-comparison", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def post_predict(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Flask API. Make sure it is running at http://localhost:5000"
    except Exception as e:
        return None, str(e)


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/hotel.png", width=64)
st.sidebar.title("🏨 Hotel Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Model Insights", "Predict"],
    label_visibility="collapsed",
)

api_ok = check_api_health()
if not api_ok:
    st.warning(
        "⚠️ **Flask API is not reachable** at `http://localhost:5000`. "
        "Start it with `python hotel-cancellation-project/backend/app.py`. "
        "Some pages may show errors until the API is running.",
        icon="⚠️",
    )

st.sidebar.markdown("---")
st.sidebar.caption("Hotel Cancellation Analysis · Python + Streamlit")


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("🏨 Hotel Booking Cancellation Analysis")
    st.markdown("Real-time insights from the Hotel Reservations dataset.")

    data, err = fetch_stats()
    if err or data is None:
        st.error(f"Could not load statistics: {err}")
        return

    s = data["summary"]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Bookings</div>
            <div class="kpi-value">{s['total_bookings']:,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cancellation Rate</div>
            <div class="kpi-value">{s['cancellation_rate_pct']}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Price / Room</div>
            <div class="kpi-value">${s['avg_price_per_room']}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Lead Time</div>
            <div class="kpi-value">{s['avg_lead_time_days']} days</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cancellation vs Not-Canceled donut
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Pie(
            labels=["Not Canceled", "Canceled"],
            values=[s["not_canceled"], s["canceled"]],
            hole=0.55,
            marker_colors=["#3b82d4", "#ef4444"],
        ))
        fig.update_layout(
            title="Booking Status Distribution",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Monthly cancellation trend
        monthly = pd.DataFrame(data["monthly_trend"])
        MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
        fig2 = px.bar(
            monthly, x="month_name", y="rate",
            color="rate",
            color_continuous_scale="RdYlGn_r",
            labels={"month_name": "Month", "rate": "Cancellation Rate (%)"},
            title="Monthly Cancellation Rate (%)",
        )
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(margin=dict(t=50, b=40))
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — EDA
# ---------------------------------------------------------------------------
def page_eda():
    st.title("📊 Exploratory Data Analysis")

    data, err = fetch_stats()
    if err or data is None:
        st.error(f"Could not load statistics: {err}")
        return

    # --- Row 1: Segment & Room Type ---
    col1, col2 = st.columns(2)

    with col1:
        seg = pd.DataFrame(data["segment_distribution"])
        fig = px.bar(
            seg, x="segment", y="rate",
            color="rate", color_continuous_scale="Blues",
            labels={"segment": "Market Segment", "rate": "Cancellation Rate (%)"},
            title="Cancellation Rate by Market Segment",
            text="rate",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        room = pd.DataFrame(data["room_type_distribution"])
        fig2 = px.bar(
            room, x="room_type", y="rate",
            color="rate", color_continuous_scale="Purples",
            labels={"room_type": "Room Type", "rate": "Cancellation Rate (%)"},
            title="Cancellation Rate by Room Type",
            text="rate",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Row 2: Lead Time & Meal Plan ---
    col3, col4 = st.columns(2)

    with col3:
        lead = pd.DataFrame(data["lead_time_buckets"])
        fig3 = px.bar(
            lead, x="bucket", y="rate",
            color="rate", color_continuous_scale="Oranges",
            labels={"bucket": "Lead Time (days)", "rate": "Cancellation Rate (%)"},
            title="Cancellation Rate by Lead Time Bucket",
            text="rate",
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_coloraxes(showscale=False)
        fig3.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        meal = pd.DataFrame(data["meal_plan_distribution"])
        fig4 = px.pie(
            meal, names="meal_plan", values="total",
            title="Bookings by Meal Plan",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig4.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig4, use_container_width=True)

    # --- Row 3: Correlation with cancellation ---
    st.markdown("### Correlation of Features with Cancellation")
    corr = data["correlation_with_cancellation"]
    corr_df = pd.DataFrame(
        sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True),
        columns=["Feature", "Correlation"]
    )
    colors = ["#ef4444" if v > 0 else "#3b82d4" for v in corr_df["Correlation"]]
    fig5 = go.Figure(go.Bar(
        x=corr_df["Correlation"],
        y=corr_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in corr_df["Correlation"]],
        textposition="outside",
    ))
    fig5.update_layout(
        title="Feature Correlation with Cancellation (positive = more likely to cancel)",
        xaxis_title="Pearson Correlation",
        yaxis_title="Feature",
        height=420,
        margin=dict(t=60, l=200, b=40),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # --- Stacked bar: canceled vs not per segment ---
    st.markdown("### Booking Volume: Canceled vs Not Canceled by Segment")
    seg2 = pd.DataFrame(data["segment_distribution"])
    seg2["not_canceled"] = seg2["total"] - seg2["canceled"]
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name="Not Canceled", x=seg2["segment"], y=seg2["not_canceled"],
                          marker_color="#3b82d4"))
    fig6.add_trace(go.Bar(name="Canceled", x=seg2["segment"], y=seg2["canceled"],
                          marker_color="#ef4444"))
    fig6.update_layout(
        barmode="stack",
        xaxis_title="Market Segment",
        yaxis_title="Number of Bookings",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig6, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3 — Model Insights
# ---------------------------------------------------------------------------
def page_model_insights():
    st.title("🤖 Model Insights")

    # Feature importance
    fi_data, fi_err = fetch_feature_importance()
    cmp_data, cmp_err = fetch_model_comparison()

    if fi_err or fi_data is None:
        st.error(f"Could not load feature importance: {fi_err}")
    else:
        st.markdown(f"### Top-20 Feature Importances  ·  Model: **{fi_data['model']}**")
        features = list(fi_data["top_features"].keys())
        scores = list(fi_data["top_features"].values())

        # normalise to percentage
        total = sum(scores) if sum(scores) > 0 else 1
        pcts = [round(s / total * 100, 2) for s in scores]

        fig = go.Figure(go.Bar(
            x=pcts,
            y=features,
            orientation="h",
            marker=dict(
                color=pcts,
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{p:.2f}%" for p in pcts],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Relative Importance (%)",
            yaxis=dict(autorange="reversed"),
            height=560,
            margin=dict(t=30, l=220, b=40, r=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison table
    if cmp_err or cmp_data is None:
        st.error(f"Could not load model comparison: {cmp_err}")
    else:
        st.markdown("### Model Comparison (Test Set)")
        best = cmp_data.get("best_model", "")
        results = cmp_data.get("results", {})
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": f"⭐ {name}" if name == best else name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1"],
                "ROC-AUC": m["roc_auc"],
            })
        df_cmp = pd.DataFrame(rows)

        # Highlight the best row
        def highlight_best(row):
            if row["Model"].startswith("⭐"):
                return ["background-color: #f0f9ff; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_cmp.style.apply(highlight_best, axis=1).format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1 Score": "{:.4f}", "ROC-AUC": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ROC-AUC bar chart
        if results:
            fig2 = px.bar(
                df_cmp,
                x="Model", y="ROC-AUC",
                color="ROC-AUC", color_continuous_scale="Blues",
                title="Model ROC-AUC Comparison",
                text="ROC-AUC",
                range_y=[0, 1],
            )
            fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(margin=dict(t=60, b=40))
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Predict
# ---------------------------------------------------------------------------
def page_predict():
    st.title("🔮 Live Cancellation Prediction")
    st.markdown("Fill in the booking details below to get an instant cancellation risk prediction.")

    with st.form("prediction_form"):
        st.markdown("#### Guest & Room Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            no_of_adults = st.number_input("No. of Adults", min_value=0, max_value=10, value=2)
            no_of_children = st.number_input("No. of Children", min_value=0, max_value=10, value=0)
        with c2:
            no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=14, value=1)
            no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=14, value=2)
        with c3:
            room_type_reserved = st.selectbox(
                "Room Type",
                ["Room_Type 1","Room_Type 2","Room_Type 3",
                 "Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"]
            )
            required_car_parking_space = st.selectbox("Car Parking Required?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.markdown("#### Booking Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"])
            market_segment_type = st.selectbox("Market Segment", ["Online","Offline","Corporate","Aviation","Complementary"])
        with c5:
            lead_time = st.slider("Lead Time (days)", 0, 500, 50)
            avg_price_per_room = st.number_input("Avg Price / Room ($)", min_value=0.0, max_value=600.0, value=100.0, step=0.5)
        with c6:
            arrival_year = st.selectbox("Arrival Year", [2017, 2018])
            arrival_month = st.slider("Arrival Month", 1, 12, 6)
            arrival_date = st.slider("Arrival Date", 1, 31, 15)

        st.markdown("#### Guest History & Preferences")
        c7, c8, c9 = st.columns(3)
        with c7:
            repeated_guest = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c8:
            no_of_previous_cancellations = st.number_input("Prev. Cancellations", min_value=0, max_value=20, value=0)
            no_of_previous_bookings_not_canceled = st.number_input("Prev. Confirmed Bookings", min_value=0, max_value=60, value=0)
        with c9:
            no_of_special_requests = st.slider("Special Requests", 0, 5, 0)

        submitted = st.form_submit_button("🔍 Predict Cancellation Risk", use_container_width=True)

    if submitted:
        payload = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
        }

        with st.spinner("Predicting..."):
            result, err = post_predict(payload)

        if err:
            st.error(f"Prediction failed: {err}")
        else:
            prob = result["probability"]
            pred = result["prediction"]
            risk = result["cancellation_risk"]

            st.markdown("---")
            st.markdown("### Prediction Result")
            r1, r2, r3 = st.columns([1, 2, 1])

            with r1:
                if pred == "Canceled":
                    st.error(f"### ❌ {pred}")
                else:
                    st.success(f"### ✅ {pred}")

                risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}[risk]
                st.markdown(f'<span class="{risk_class}">Risk: {risk}</span>', unsafe_allow_html=True)

            with r2:
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%"},
                    title={"text": "Cancellation Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if prob >= 0.5 else "#3b82d4"},
                        "steps": [
                            {"range": [0, 40], "color": "#dcfce7"},
                            {"range": [40, 70], "color": "#fef9c3"},
                            {"range": [70, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#1f2328", "width": 3},
                            "thickness": 0.75,
                            "value": prob * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r3:
                st.markdown("**Confidence Breakdown**")
                st.metric("Cancel probability", f"{prob*100:.1f}%")
                st.metric("Stay probability", f"{(1-prob)*100:.1f}%")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Overview":
    page_overview()
elif page == "EDA":
    page_eda()
elif page == "Model Insights":
    page_model_insights()
elif page == "Predict":
    page_predict()
