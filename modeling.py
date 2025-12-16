"""
Train multiple forecasting models on daily aggregated data and save them.
Saves: models/{model_name}_model.pkl and metrics_daily.pkl
- Statistical models: ARIMA, SARIMAX, HoltWinters
- Prophet
- XGBoost:
    * recursive one-step (uses last 60 days lags), saved as XGBoost_recursive
    * direct-horizon models for horizons [1,7,30,90] saved as XGBoost_h{h} if xgboost installed
Notes:
- Uses 60-day lookback for ML features (balanced for ~1400 rows)
- Forecast horizon: 90 days for evaluation
"""
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# optional xgboost
try:
    import xgboost as xgb
except Exception:
    xgb = None

# CONFIG
CSV_PATH = "processed_data/daily_power_consumption.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZON = 90
TEST_SIZE = HORIZON
ML_LAG = 60  # lookback window for ML models (recommended for ~1400 rows)
DIRECT_HORIZONS = [1, 7, 30, 90]  # horizons we'll train direct xgboost models for if xgboost available

TARGET_CANDIDATES = [
    "Global_active_power_kwh",
    "Global_active_power_sum",
    "energy",
    "Global_active_power"
]

def choose_target_column(df):
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            return t
    for c in df.columns:
        if "active" in c.lower():
            return c
    raise RuntimeError("No suitable target column found. Columns: " + ", ".join(df.columns))

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}

# Load data
df = pd.read_csv(CSV_PATH, parse_dates=["Datetime"]).sort_values("Datetime")
df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("D")
df = df.set_index("Datetime").asfreq("D")

print("Loaded daily rows:", len(df), df.index.min().date(), "->", df.index.max().date())

target_col = choose_target_column(df)
print("Target column:", target_col)

series = df[target_col].astype(float).interpolate(limit=7, limit_direction="both")
series = series.dropna()
if len(series) < (TEST_SIZE + 100):
    print("Warning: series has relatively few rows after interpolation:", len(series))

# split
train = series.iloc[:-TEST_SIZE].copy()
test = series.iloc[-TEST_SIZE:].copy()
print("Train rows:", len(train), "Test rows:", len(test))

results = {}
models = {}

# Baselines
last_val = train.iloc[-1]
naive = np.repeat(last_val, HORIZON)
results['NaiveLast'] = evaluate(test, naive)
seasonal7 = np.tile(train.iloc[-7:].values, int(np.ceil(HORIZON/7)))[:HORIZON]
results['SeasonalNaive7'] = evaluate(test, seasonal7)

# ARIMA
try:
    print("Training ARIMA(1,1,1)...")
    arima = ARIMA(train, order=(1,1,1))
    arima_fit = arima.fit()
    arima_pred = arima_fit.forecast(steps=HORIZON)
    results['ARIMA'] = evaluate(test, arima_pred)
    models['ARIMA'] = arima_fit
    joblib.dump(arima_fit, os.path.join(MODELS_DIR, "ARIMA_model.pkl"))
    print("ARIMA done.")
except Exception as e:
    print("ARIMA failed:", e)

# SARIMAX (weekly)
try:
    print("Training SARIMAX seasonal weekly...")
    sarimax = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarimax_fit = sarimax.fit(disp=False)
    sarimax_pred = sarimax_fit.forecast(steps=HORIZON)
    results['SARIMAX'] = evaluate(test, sarimax_pred)
    models['SARIMAX'] = sarimax_fit
    joblib.dump(sarimax_fit, os.path.join(MODELS_DIR, "SARIMAX_model.pkl"))
    print("SARIMAX done.")
except Exception as e:
    print("SARIMAX failed:", e)

# Holt-Winters
try:
    print("Training Holt-Winters seasonal_periods=7...")
    hw = ExponentialSmoothing(train, seasonal="add", seasonal_periods=7).fit()
    hw_pred = hw.forecast(steps=HORIZON)
    results['HoltWinters'] = evaluate(test, hw_pred)
    models['HoltWinters'] = hw
    joblib.dump(hw, os.path.join(MODELS_DIR, "HoltWinters_model.pkl"))
    print("Holt-Winters done.")
except Exception as e:
    print("Holt-Winters failed:", e)

# Prophet
try:
    print("Training Prophet...")
    prophet_df = train.reset_index().rename(columns={"Datetime":"ds", target_col:"y"})
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=HORIZON, freq='D')
    forecast = m.predict(future)
    prophet_pred = forecast['yhat'].tail(HORIZON).values
    results['Prophet'] = evaluate(test, prophet_pred)
    models['Prophet'] = m
    joblib.dump(m, os.path.join(MODELS_DIR, "Prophet_model.pkl"))
    print("Prophet done.")
except Exception as e:
    print("Prophet failed:", e)

# XGBoost models (if available)
if xgb is not None:
    try:
        print("Training XGBoost recursive one-step (lags={} days)...".format(ML_LAG))
        # Prepare lag features using full train (for one-step)
        df_lags = pd.DataFrame(index=train.index)
        for l in range(1, ML_LAG+1):
            df_lags[f"lag_{l}"] = train.shift(l)
        df_lags["y"] = train.values
        df_lags = df_lags.dropna()
        X_train_xgb = df_lags[[c for c in df_lags.columns if c.startswith("lag_")]]
        y_train_xgb = df_lags["y"]
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=4)
        xgb_model.fit(X_train_xgb, y_train_xgb)
        # Recursive forecast
        last_vals = list(train.values[-ML_LAG:])
        preds = []
        for _ in range(HORIZON):
            x_input = np.array(last_vals[-ML_LAG:]).reshape(1, -1)
            p = xgb_model.predict(x_input)[0]
            preds.append(p)
            last_vals.append(p)
        results['XGBoost_recursive'] = evaluate(test, preds)
        models['XGBoost_recursive'] = xgb_model
        joblib.dump(xgb_model, os.path.join(MODELS_DIR, "XGBoost_recursive_model.pkl"))
        print("XGBoost recursive done.")
    except Exception as e:
        print("XGBoost recursive failed:", e)

    # Direct horizon models
    for h in DIRECT_HORIZONS:
        try:
            print(f"Training XGBoost direct model for horizon={h} ...")
            # build supervised dataset where target is value at t+h
            df_sup = pd.DataFrame(index=series.index)
            for l in range(1, ML_LAG+1):
                df_sup[f"lag_{l}"] = series.shift(l)
            df_sup[f"y_h{h}"] = series.shift(-h)
            df_sup = df_sup.dropna()
            X = df_sup[[c for c in df_sup.columns if c.startswith("lag_")]]
            y = df_sup[f"y_h{h}"]
            # train on data up to end of train index
            train_idx = X.index <= train.index[-1]
            X_tr = X.loc[train_idx]
            y_tr = y.loc[train_idx]
            if len(X_tr) < 50:
                print(f"Not enough rows to train direct XGBoost h={h} (rows={len(X_tr)}). Skipping.")
                continue
            model_h = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=4)
            model_h.fit(X_tr, y_tr)
            joblib.dump(model_h, os.path.join(MODELS_DIR, f"XGBoost_h{h}_model.pkl"))
            print(f"Saved XGBoost_h{h}.")
        except Exception as e:
            print(f"XGBoost direct h={h} failed:", e)
else:
    print("xgboost not installed — skipping XGBoost models. (pip install xgboost to enable)")

# Save results & metrics
metrics_path = os.path.join(MODELS_DIR, "metrics_daily.pkl")
joblib.dump(results, metrics_path)
print("Saved metrics to", metrics_path)
print("Modeling complete. Metrics:")
for k,v in results.items():
    print(k, v)
