import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Ensure log folder exists
os.makedirs("logs", exist_ok=True)

# Logging setup
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

st.set_page_config(page_title="💰 Tide Dynamic Pricing", layout="wide")
st.title("💰 Tide Dynamic Pricing")


# ---------- Accuracy Trend Section (Always Visible at Top) ----------
def generate_dummy_accuracy_log():
    today = datetime.today()
    dates = [today - timedelta(days=i * 5) for i in range(6)][::-1]
    accuracies = np.round(
        np.linspace(0.81, 0.88, num=6) + np.random.normal(0, 0.005, size=6), 4
    )
    df = pd.DataFrame({"date": dates, "accuracy": accuracies})
    return df


API_URL1 = "http://localhost:8000/metrics/accuracy"  # Replace with deployed URL if needed


def display_accuracy_trend():
    try:
        # Fetch from FastAPI
        response = requests.get(API_URL1, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Parse records
        records = data.get("accuracy", [])
        if not records:
            st.warning("No accuracy metrics available.")
            return

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = df["value"].astype(float)

        latest = df["value"].iloc[0]  # Reversed order: latest is first
        previous = df["value"].iloc[1] if len(df) > 1 else latest
        delta = latest - previous

        with st.container():
            chart_col, metric_col = st.columns([4, 1])

            with chart_col:
                st.markdown("### 📊 Directional Accuracy Trend")

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["value"],
                        mode="lines+markers",
                        line=dict(color="#1f77b4", width=2),
                        marker=dict(size=6),
                        hovertemplate="Date: %{x}<br>Accuracy: %{y:.2f}%",
                        name="Accuracy",
                    )
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Directional Accuracy (%)",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=300,
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

            with metric_col:
                st.markdown("###")
                st.metric(
                    label="Latest Accuracy",
                    value=f"{latest:.2f}%",
                    delta=f"{delta:+.2f}%",
                )

    except requests.exceptions.RequestException as req_err:
        st.error(f"Error fetching accuracy metrics: {req_err}")
    except Exception as err:
        st.error(f"Unexpected error: {err}")


# Show accuracy chart at top
display_accuracy_trend()
st.markdown("---")

API_URL = "http://localhost:8000/predict"  # Update with actual FastAPI URL

st.markdown("### 🧾 Product & Competitor Info")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        MRP = st.number_input("Competitor MRP (₹)", min_value=0.0, value=500.0)
        DiscountRate = st.slider(
            "Competitor Discount Rate (%)", 0.0, 100.0, 10.0
        )
        Brand_encoded = st.number_input(
            "Brand (encoded)", min_value=0, max_value=100, value=23
        )
        Is_Premium_Brand = st.selectbox("Is Premium Brand?", [0, 1])

    with col2:
        Is_Peak_Season = st.selectbox("Is Peak Season?", [0, 1])
        Product_Age_Days = st.number_input(
            "Product Age (days)", min_value=0, value=120
        )
        IsMetroMarket = st.selectbox("Metro Market?", [0, 1])
        UnitsSold = st.number_input(
            "Expected Units Sold", min_value=0, value=50
        )

    st.markdown("### 🛒 Customer Behavior Insights")
    col3, col4 = st.columns(2)

    with col3:
        CTR = st.slider("Click-Through Rate (%)", 0.0, 100.0, 5.0)
        BounceRate = st.slider("Bounce Rate (%)", 0.0, 100.0, 30.0)
        ReturningVisitorRatio = st.slider(
            "Returning Visitor Ratio", 0.0, 1.0, 0.3
        )

    with col4:
        FunnelDrop_ViewToCart = st.slider(
            "View → Cart Drop (%)", 0.0, 100.0, 25.0
        )
        FunnelDrop_CartToCheckout = st.slider(
            "Cart → Checkout Drop (%)", 0.0, 100.0, 15.0
        )
        AvgSessionDuration = st.slider(
            "Avg Session Duration (sec)", 0, 600, 180
        )

    submit = st.form_submit_button("🔮 Predict")

if submit:
    try:
        payload = {
            "MRP": MRP,
            "DiscountRate": DiscountRate,
            "Brand_encoded": Brand_encoded,
            "Is_Premium_Brand": Is_Premium_Brand,
            "Is_Peak_Season": Is_Peak_Season,
            "Product_Age_Days": Product_Age_Days,
            "IsMetroMarket": IsMetroMarket,
            "UnitsSold": UnitsSold,
            "CTR": CTR,
            "BounceRate": BounceRate,
            "ReturningVisitorRatio": ReturningVisitorRatio,
            "FunnelDrop_ViewToCart": FunnelDrop_ViewToCart,
            "FunnelDrop_CartToCheckout": FunnelDrop_CartToCheckout,
            "AvgSessionDuration": AvgSessionDuration,
        }

        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        st.markdown("---")
        st.markdown("### 📈 Prediction Results")
        result_col1, result_col2 = st.columns(2)
        result_col1.metric("🎯 Predicted Price", f"₹{result['price']:,.2f}")
        result_col2.metric("📦 Revenue", f"₹{result['revenue']:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
