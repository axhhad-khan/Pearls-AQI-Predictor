# 🌬️ AQI Predictor — End-to-End Air Quality Forecasting

Predicts the **Air Quality Index (AQI)** for any city for the **next 3 days** using a fully serverless ML stack.

## Architecture

```
Open-Meteo API (free) → Feature Pipeline → MongoDB (Feature Store)
                                                    ↓
                                           Training Pipeline
                                    (Random Forest | XGBoost | LightGBM)
                                                    ↓
                                         MongoDB (Model Registry)
                                                    ↓
                                        Streamlit Dashboard
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data source | Open-Meteo Air Quality API (free, no key) |
| Feature store & model registry | MongoDB Atlas (free tier) |
| ML models | Random Forest · XGBoost · LightGBM |
| Dashboard | Streamlit + Plotly |
| CI/CD automation | GitHub Actions |
| Explainability | SHAP |

---

## Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd aqi_predictor
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — only MONGO_URI needs changing.
# Open-Meteo requires NO API key.
```

`.env` example:
```
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB_NAME=aqi_predictor
CITY_NAME=Karachi
LATITUDE=24.8607
LONGITUDE=67.0011
```

### 3. Backfill historical data (30 days)
```bash
python backfill.py --days 30
```

### 4. Train models
```bash
python run_training_pipeline.py
```

### 5. Launch dashboard
```bash
streamlit run app.py
```

---

## Project Structure

```
aqi_predictor/
├── app.py                          # Streamlit dashboard
├── backfill.py                     # Historical data backfill
├── run_feature_pipeline.py         # CI/CD hourly runner
├── run_training_pipeline.py        # CI/CD daily runner
├── requirements.txt
├── .env.example
│
├── pipelines/
│   ├── fetch_data.py               # Open-Meteo API calls
│   ├── feature_pipeline.py         # Feature engineering + AQI computation
│   └── training_pipeline.py        # RF / XGBoost / LightGBM training + SHAP
│
├── utils/
│   └── db.py                       # MongoDB feature store + model registry
│
└── .github/
    └── workflows/
        └── pipelines.yml           # GitHub Actions (hourly + daily)
```

---

## Features Engineered

| Category | Features |
|---|---|
| Time | hour, day_of_week, month, is_weekend, day_of_year |
| Cyclical | hour_sin/cos, month_sin/cos |
| Pollutants | PM2.5, PM10, CO, NO₂, SO₂, O₃ |
| Weather | temperature, humidity, wind speed/dir, pressure, precipitation |
| Lag | AQI lag 1h / 3h / 6h / 24h |
| Derived | AQI change rate (1h/3h), rolling mean (6h/24h), PM ratio, wind U/V |

---

## ML Models

All three models predict **AQI 24 hours ahead** (regression).
Metrics reported: **RMSE**, **MAE**, **R²** on a held-out temporal test split.

- **Random Forest** — `n_estimators=200`, `max_depth=12`
- **XGBoost** — `n_estimators=300`, `learning_rate=0.05`, `subsample=0.8`
- **LightGBM** — `n_estimators=300`, `num_leaves=63`, `learning_rate=0.05`

The best-scoring model (by RMSE) is also saved as `best_model` in the registry.

---

## CI/CD (GitHub Actions)

Add these **repository secrets** in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `MONGO_URI` | Your MongoDB connection string |
| `MONGO_DB_NAME` | `aqi_predictor` |
| `CITY_NAME` | `Karachi` |
| `LATITUDE` | `24.8607` |
| `LONGITUDE` | `67.0011` |

Schedules:
- Feature pipeline → every hour (`0 * * * *`)
- Training pipeline → daily at 02:00 UTC (`0 2 * * *`)

---

## AQI Scale (US EPA)

| AQI | Category | Color |
|---|---|---|
| 0–50 | Good | 🟢 |
| 51–100 | Moderate | 🟡 |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 |
| 151–200 | Unhealthy | 🔴 |
| 201–300 | Very Unhealthy | 🟣 |
| 301–500 | Hazardous | 🔴⚫ |

Alerts are shown automatically on the dashboard for hazardous conditions.
