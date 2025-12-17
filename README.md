# Household Electricity Consumption Forecasting System ⚡

An end-to-end **time-series forecasting pipeline** to predict **hourly and daily Global Active Power consumption** using the **UCI Individual Household Electric Power Consumption** dataset.

**Hackathon-ready • Fully reproducible • Tested • Deployable (Streamlit + FastAPI)**

---

## 🚀 Project Summary

This project delivers a **production-grade forecasting system** that:

- Ingests and cleans ~4 years of minute-level household electricity data  
- Resamples data to **hourly and daily** granularity  
- Engineers rich **lag, rolling, calendar, and seasonal features**  
- Trains and compares **classical**, **statistical**, and **ML-based** models  
- Performs **rolling-window backtesting** for realistic evaluation  
- Provides **model explainability** using SHAP  
- Includes **unit tests and robustness (mutation-style) experiments**  
- Deploys predictions via **Streamlit dashboard** and **FastAPI endpoint**

Designed end-to-end to demonstrate **engineering, modeling, evaluation, and deployment skills** in a hackathon setting.

---

## 🎯 Problem Statement & Business Impact

Accurate short-term household electricity forecasting enables:

- Smart energy usage optimization  
- Peak-load reduction and demand-response strategies  
- Better integration of solar panels and battery storage  
- Reduced electricity bills and grid stress  

Even a **5–10% improvement in forecast accuracy** can translate into significant **cost savings and lower CO₂ emissions**.

---

## ✅ Objective & Success Criteria

**Goal:** Forecast **Global Active Power (kW)** for:

- **Next 1–7 days** (daily forecasting)  
- **Next 24–168 hours** (hourly forecasting)

**Success Metrics:**

- Daily **MAPE < 15%** (baseline naive ≈ 22–25%)  
- All unit tests passing  
- Functional Streamlit dashboard and FastAPI endpoint  
- Clear documentation of experiments and robustness checks  

---

## 📊 Dataset

**Name:** Individual Household Electric Power Consumption  
**Source:** UCI Machine Learning Repository  

- Dataset page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption  
- Direct download: https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip  

**Dataset Details:**

- Time span: December 2006 – November 2010  
- Frequency: 1-minute  
- Total rows: ~2.07 million  
- Target variable: `Global_active_power` (kW)  
- Additional features: voltage, intensity, reactive power, sub-metering 1–3  

Data is resampled to **hourly and daily frequencies** for forecasting.

---

## 🏗️ High-Level Architecture

```
Raw ZIP Data
   ↓
Data Ingestion & Validation
   ↓
Cleaning & Missing Value Handling
   ↓
Resampling (Hourly / Daily)
   ↓
Feature Engineering
   ↓
Time-Based Train / Validation Split
   ↓
Model Training
   ↓
Rolling-Window Backtesting
   ↓
Explainability & Residual Analysis
   ↓
Model Persistence
   ↓
Streamlit Dashboard + FastAPI API
```

---

## 🧠 Feature Engineering

- Lag features: 1h, 24h, 48h, 168h  
- Rolling statistics: mean & std (24h, 168h windows)  
- Calendar features: hour, day-of-week, month, weekend flag  
- Cyclic encodings (sin/cos)  
- Fourier terms for daily and weekly seasonality  
- Holiday effects (France)

---

## 🤖 Models & Experiments

| Category | Models | Notes |
|--------|--------|------|
| Baseline | Seasonal Naive, Moving Average | Reference benchmarks |
| Classical | SARIMA / SARIMAX | Auto ARIMA order selection |
| Additive | Prophet | Strong seasonality handling |
| Machine Learning | XGBoost, LightGBM, Random Forest | Tuned using Optuna |

The **best-performing model** is selected using validation **MAPE**.

---

## 📐 Evaluation Metrics

- MAE  
- RMSE  
- MAPE / sMAPE  
- 95% Prediction Interval Coverage  
- Directional Accuracy  

All metrics are computed using **rolling-origin** and **expanding-window backtesting**.

---

## 🔍 Explainability & Robustness

**Explainability:**
- SHAP summary and force plots (tree-based models)  
- Prophet trend and seasonality decomposition  

**Robustness Experiments (`notebooks/robustness.ipynb`):**
1. Target noise injection  
2. Removal of top lag features  
3. Gaussian noise added to inputs  

---

## 🧪 Unit Tests & Mutation Testing

The `tests/` directory contains **pytest-based unit tests** covering:

- Data ingestion and preprocessing  
- Feature engineering  
- Model training and serialization  

Run tests using:

```bash
pytest -q
```

Robustness experiments serve as **ML-specific mutation testing**.

---

## 🌐 Deployment & Live Demo

- **Streamlit Dashboard** for visualization and forecasting  
- **FastAPI** endpoint for programmatic predictions  

> Replace the links below after deployment:

- Live Demo: https://your-power-forecast.streamlit.app  
- API Docs: https://your-power-api.onrender.com/docs  

A Dockerfile is included for containerized deployment.

---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/YOUR-TEAM/household-power-forecasting.git
cd household-power-forecasting

python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

# Place dataset in data/raw/ or run the download script

# Train the model
python src/train.py

# Launch Streamlit app
streamlit run app/streamlit_app.py

# Launch FastAPI server
uvicorn app.api:app --reload
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── src/            # Core pipeline
├── app/            # Streamlit & FastAPI
├── artifacts/      # Saved models
├── notebooks/      # EDA & experiments
├── tests/          # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 👥 Team

**Team Name:** YOUR TEAM NAME  
**Event:** Hackathon 2025  

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Univariate forecasting (no weather data)  
- No minute-level forecasting  
- No online retraining loop  

**Future Enhancements:**
- Integrate weather and temperature data  
- Deep learning models (LSTM / Transformer)  
- Automated retraining and monitoring  

---

🎉 **Good luck in the hackathon!**  
This README is clean, professional, and judge-friendly — just add your **team name and deployment links**, push to GitHub, and you’re ready to submit.

