# streamlit_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet.plot import plot_plotly

st.set_page_config(layout="wide", page_title="Daily Power Forecaster")

CSV_PATH = "processed_data/daily_power_consumption.csv"
MODELS_DIR = "models"
MAX_HORIZON = 20

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["Datetime"]).sort_values("Datetime")
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("D")
    df = df.set_index("Datetime").asfreq("D")
    return df

df = load_data()
st.title("Household Power Consumption Forecaster (Daily)")

# model discovery
available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl")]
available_models_display = sorted(available_models)
st.sidebar.header("Forecast settings")
model_choice = st.sidebar.selectbox("Choose model (pick one saved in models/)", available_models_display)
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=MAX_HORIZON, value=20)

# show dataset summary
st.sidebar.markdown(f"**Data:** {df.shape[0]} days from {df.index.min().date()} to {df.index.max().date()}")
target_candidates = [c for c in df.columns if "active" in c.lower() or "energy" in c.lower()]
target_col = target_candidates[0] if target_candidates else df.columns[0]
st.sidebar.markdown(f"**Target column:** {target_col}")

st.header("Recent data")
st.dataframe(df.tail(10))

# Forecast action
if st.sidebar.button("Generate Forecast"):
    model_path = os.path.join(MODELS_DIR, model_choice)
    st.info(f"Loading model: {model_choice}")
    model = joblib.load(model_path)

    # produce forecast depending on model type
    last_date = df.index.max()
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    preds = None
    if "Prophet" in model_choice:
        # Prophet object
        # build history as df with columns ds, y
        hist = df[target_col].reset_index().rename(columns={"Datetime":"ds", target_col:"y"})
        m = model
        future = m.make_future_dataframe(periods=horizon, freq='D')
        fc = m.predict(future)
        preds = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon).set_index('ds')
    elif "SARIMAX" in model_choice or "SARIMAX_model" in model_choice:
        try:
            fc = model.forecast(steps=horizon)
            preds = pd.DataFrame({'yhat': np.asarray(fc)}, index=forecast_index)
        except Exception as e:
            st.error(f"SARIMAX forecasting failed: {e}")
    elif "ARIMA" in model_choice:
        try:
            fc = model.forecast(steps=horizon)
            preds = pd.DataFrame({'yhat': np.asarray(fc)}, index=forecast_index)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
    elif "HoltWinters" in model_choice:
        try:
            fc = model.forecast(steps=horizon)
            preds = pd.DataFrame({'yhat': np.asarray(fc)}, index=forecast_index)
        except Exception as e:
            st.error(f"Holt-Winters forecasting failed: {e}")
    elif "XGBoost_recursive" in model_choice:
        # need to construct lag features from last ML_LAG days (model trained with ML_LAG=60 by default)
        ML_LAG = 60
        last_vals = list(df[target_col].dropna().values[-ML_LAG:])
        preds_list = []
        for i in range(horizon):
            x_input = np.array(last_vals[-ML_LAG:]).reshape(1, -1)
            p = model.predict(x_input)[0]
            preds_list.append(p)
            last_vals.append(p)
        preds = pd.DataFrame({'yhat': np.array(preds_list)}, index=forecast_index)
    elif model_choice.startswith("XGBoost_h"):
        # direct xgboost model for a specific horizon (e.g. XGBoost_h7_model.pkl)
        # if user asks horizon different than model horizon, we will inform
        try:
            # infer model horizon from filename
            import re
            m = re.search(r"h(\d+)", model_choice)
            if m:
                model_h = int(m.group(1))
                #   if model_h != horizon:
                    #st.warning(f"This XGBoost model was trained for horizon={model_h}. If you want horizon {horizon}, use XGBoost_recursive or train a direct model for that horizon.")
                # build input vector from last ML_LAG days
                ML_LAG = 60
                last_vals = list(df[target_col].dropna().values[-ML_LAG:])
                x_input = np.array(last_vals[-ML_LAG:]).reshape(1,-1)
                p = model.predict(x_input)[0]
                # place the single-point prediction at date last_date + model_h
                pred_index = last_date + pd.Timedelta(days=model_h)
                preds = pd.DataFrame({'yhat': [p]}, index=[pred_index])
            else:
                st.error("Could not parse horizon from model filename.")
        except Exception as e:
            st.error(f"XGBoost direct model forecasting failed: {e}")
    else:
        st.error("Unknown model type or unsupported model. Make sure model file created by modeling_daily.py")

    if preds is not None:
        st.subheader("Forecast")
        st.dataframe(preds.reset_index().rename(columns={'index':'Datetime'}).head(20))

        # Plot actual last 180 days + forecast
        plot_span = 180
        recent = df[target_col].dropna().tail(plot_span)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=preds.index, y=preds['yhat'], mode="lines+markers", name="Forecast"))
        fig.update_layout(title="Actual (recent) and Forecast", xaxis_title="Date", yaxis_title=target_col)
        st.plotly_chart(fig, use_container_width=True)

        # Show metrics for model if available
        metrics_path = os.path.join(MODELS_DIR, "metrics_daily.pkl")
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            st.subheader("Model metrics (on 20-day test)")
            if model_choice.replace("_model.pkl","") in metrics:
                st.json(metrics[model_choice.replace("_model.pkl","")])
            else:
                st.json(metrics)

# --- ALERT LOGIC: show alert if last predicted value changes beyond threshold ---
try:
    ALERT_PCT = 17.0  # threshold %

    # Last predicted value
    last_pred = float(preds['yhat'].dropna().iloc[-1])

    # Last actual value
    last_actual_series = df[target_col].dropna()
    if len(last_actual_series) == 0:
        st.info("No historical actual values available to compare for alerts.")
    else:
        last_actual = float(last_actual_series.iloc[-1])

        # Avoid divide-by-zero
        if last_actual == 0:
            if last_pred > 0:
                st.error(
                    f"⚠️ ALERT: Prediction for {preds.index[-1].date()} is {last_pred:.2f} while last actual is 0."
                )
        else:
            # Percentage change
            pct_change = (last_pred - last_actual) / last_actual * 100.0

            # BAD CASE → Red
            if pct_change >= ALERT_PCT:
                st.error(
                    f"🔴 **ALERT:** Tomorrow’s predicted consumption `{last_pred:.2f}` "
                    f"is **{pct_change:.1f}% HIGHER** than today's value `{last_actual:.2f}` "
                    f"(Threshold: {ALERT_PCT}%)."
                )

            # GOOD CASE → Green
            elif pct_change <= -ALERT_PCT:
                st.success(
                    f"🟢 **GOOD NEWS:** Tomorrow’s predicted consumption `{last_pred:.2f}` "
                    f"is **{abs(pct_change):.1f}% LOWER** than today `{last_actual:.2f}` "
                    f"(Threshold: {ALERT_PCT}%)."
                )

            # Normal / Neutral → Blue
            else:
                st.info(
                    f"ℹ️ Predicted change is `{pct_change:.1f}%` ― "
                    f"within safe range (threshold: ±{ALERT_PCT}%)."
                )

except Exception:
    st.warning("Alert check skipped due to a calculation error.")




