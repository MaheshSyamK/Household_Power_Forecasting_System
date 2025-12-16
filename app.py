# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

app = FastAPI(title="Time Series Forecasting API (Daily)")

CSV_PATH = "processed_data/daily_power_consumption.csv"
MODELS_DIR = "models"

# load data once
df = pd.read_csv(CSV_PATH, parse_dates=["Datetime"]).sort_values("Datetime")
df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("D")
df = df.set_index("Datetime").asfreq("D")
target_candidates = [c for c in df.columns if "active" in c.lower() or "energy" in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else df.columns[0]

class ForecastRequest(BaseModel):
    model_name: str  # model filename in models/ (e.g. Prophet_model.pkl)
    horizon: int = 30

@app.post("/predict")
def predict(req: ForecastRequest):
    model_file = os.path.join(MODELS_DIR, req.model_name)
    if not os.path.exists(model_file):
        return {"error": f"Model {req.model_name} not found in models/"}
    model = joblib.load(model_file)

    last_date = df.index.max()
    horizon = int(req.horizon)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    preds = []
    # Prophet
    if "Prophet" in req.model_name:
        hist = df[[TARGET_COL]].reset_index().rename(columns={"Datetime":"ds", TARGET_COL:"y"})
        m = model
        future = m.make_future_dataframe(periods=horizon, freq='D')
        fc = m.predict(future)
        preds = fc[['ds','yhat']].tail(horizon).rename(columns={'ds':'date','yhat':'pred'}).to_dict('records')
    elif "ARIMA" in req.model_name:
        fc = model.forecast(steps=horizon)
        preds = [{"date": str(d.date()), "pred": float(p)} for d,p in zip(forecast_index, np.asarray(fc))]
    elif "SARIMAX" in req.model_name:
        fc = model.forecast(steps=horizon)
        preds = [{"date": str(d.date()), "pred": float(p)} for d,p in zip(forecast_index, np.asarray(fc))]
    elif "HoltWinters" in req.model_name:
        fc = model.forecast(steps=horizon)
        preds = [{"date": str(d.date()), "pred": float(p)} for d,p in zip(forecast_index, np.asarray(fc))]
    elif "XGBoost_recursive" in req.model_name:
        ML_LAG = 60
        last_vals = list(df[TARGET_COL].dropna().values[-ML_LAG:])
        preds_list = []
        for _ in range(horizon):
            x_input = np.array(last_vals[-ML_LAG:]).reshape(1, -1)
            p = model.predict(x_input)[0]
            preds_list.append(p)
            last_vals.append(p)
        preds = [{"date": str(d.date()), "pred": float(p)} for d,p in zip(forecast_index, preds_list)]
    elif req.model_name.startswith("XGBoost_h"):
        # direct single-horizon xgboost
        import re
        m = re.search(r"h(\d+)", req.model_name)
        if m:
            model_h = int(m.group(1))
            if model_h != horizon:
                return {"warning": f"Model expects horizon {model_h}. Request horizon {horizon} may not match."}
            ML_LAG = 60
            last_vals = list(df[TARGET_COL].dropna().values[-ML_LAG:])
            x_input = np.array(last_vals[-ML_LAG:]).reshape(1, -1)
            p = model.predict(x_input)[0]
            date = (last_date + pd.Timedelta(days=model_h)).date()
            preds = [{"date": str(date), "pred": float(p)}]
        else:
            return {"error":"Could not parse horizon from model filename."}
    else:
        return {"error":"Unsupported model type."}

    # load metrics if available
    metrics_path = os.path.join(MODELS_DIR, "metrics_daily.pkl")
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}

    return {"model": req.model_name, "predictions": preds, "metrics": metrics}
