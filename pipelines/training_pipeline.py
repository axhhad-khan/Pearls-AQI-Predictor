"""training_pipeline.py - RF, XGBoost, LightGBM with TimeSeriesSplit"""
import logging, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from utils.db import load_features, save_model

logger = logging.getLogger(__name__)

TARGET = "target_aqi_6h"

FEATURE_COLS = [
    "aqi",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_1h", "aqi_change_3h",
    "aqi_rolling_6h_mean", "aqi_rolling_24h_mean",
    "pm10", "pm2_5", "nitrogen_dioxide", "ozone",
    "carbon_monoxide", "sulphur_dioxide",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "surface_pressure",
    "precipitation", "cloud_cover",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",
    "pm_ratio", "wind_u", "wind_v",
]

def evaluate(y_true, y_pred):
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 3),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }

def get_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, min_samples_split=20,
            max_features=0.7, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200, num_leaves=20, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbose=-1),
    }

def train(city: str = "default") -> dict:
    logger.info(f"Loading features for '{city}'")
    df = load_features(city, limit=10_000)
    if df.empty:
        raise ValueError("No data. Run feature pipeline first.")

    # Add target if missing
    if TARGET not in df.columns:
        df[TARGET] = df["aqi"].shift(-6)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Features: {len(feat_cols)}")

    df_c = df[feat_cols + [TARGET]].copy()
    df_c[feat_cols] = df_c[feat_cols].ffill()
    df_c = df_c.dropna()
    logger.info(f"Clean rows: {len(df_c)} | Target std: {df_c[TARGET].std():.2f}")

    if len(df_c) < 100:
        raise ValueError(f"Need 100+ rows, got {len(df_c)}")

    X = df_c[feat_cols].values
    y = df_c[TARGET].values

    tscv = TimeSeriesSplit(n_splits=5, test_size=max(24, len(X)//8))

    results = {}
    best_name, best_r2, best_model = None, float("-inf"), None

    for name, model in get_models().items():
        logger.info(f"Training {name}...")
        fold_m = []
        for i, (tr, te) in enumerate(tscv.split(X)):
            model.fit(X[tr], y[tr])
            m = evaluate(y[te], model.predict(X[te]))
            fold_m.append(m)
            logger.info(f"  Fold {i+1}: RMSE={m['RMSE']} R2={m['R2']} std={y[te].std():.1f}")

        avg = {k: round(float(np.mean([m[k] for m in fold_m])), 4) for k in ["RMSE","MAE","R2"]}
        results[name] = avg
        logger.info(f"  {name} AVG: {avg}")

        model.fit(X, y)  # retrain on full data
        save_model(model, name, avg, feat_cols)

        if avg["R2"] > best_r2:
            best_r2, best_name, best_model = avg["R2"], name, model

    if best_name:
        save_model(best_model, "best_model", results[best_name], feat_cols)
        logger.info(f"Best: {best_name} R2={best_r2}")

    return results

def get_shap_values(model_name: str, city: str = "default", n_samples: int = 200):
    try:
        import shap
        from utils.db import load_model
        model, feat_cols, _ = load_model(model_name)
        if model is None: return None, None
        df        = load_features(city, limit=2000)
        feat_cols = [c for c in feat_cols if c in df.columns]
        X         = df[feat_cols].ffill().dropna().tail(n_samples).values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        imp = pd.DataFrame({
            "feature":    feat_cols,
            "importance": np.abs(shap_vals).mean(axis=0)
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp, shap_vals
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return None, None
