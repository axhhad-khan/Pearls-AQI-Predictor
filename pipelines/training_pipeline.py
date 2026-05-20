"""
training_pipeline.py
Trains Random Forest, XGBoost, and LightGBM models on AQI data.
Evaluates with RMSE, MAE, R² and saves best model to MongoDB.
"""

import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from utils.db import load_features, save_model
from pipelines.feature_pipeline import get_valid_feature_cols

logger = logging.getLogger(__name__)

TARGET = "target_aqi_24h"   # predict AQI 24 hours ahead


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}


def get_models() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
    }


def train(city: str = "default") -> dict:
    """
    Full training run. Returns dict of {model_name: metrics}.
    """
    logger.info(f"Loading features for city='{city}'")
    df = load_features(city, limit=10_000)

    if df.empty:
        raise ValueError("No feature data found in MongoDB. Run the feature pipeline first.")

    feat_cols = get_valid_feature_cols(df)
    df_clean  = df[feat_cols + [TARGET]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Not enough clean rows for training ({len(df_clean)}). Need ≥100.")

    X = df_clean[feat_cols].values
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False   # temporal split
    )

    results    = {}
    best_name  = None
    best_rmse  = float("inf")
    best_model = None

    for name, model in get_models().items():
        logger.info(f"Training {name}…")
        try:
            model.fit(X_train, y_train)
            y_pred   = model.predict(X_test)
            metrics  = evaluate(y_test, y_pred)
            results[name] = metrics
            logger.info(f"  {name}: {metrics}")

            save_model(model, name, metrics, feat_cols)

            if metrics["RMSE"] < best_rmse:
                best_rmse  = metrics["RMSE"]
                best_name  = name
                best_model = model
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    if best_name:
        save_model(best_model, "best_model", results[best_name], feat_cols)
        logger.info(f"Best model: {best_name} (RMSE={best_rmse})")

    return results


def get_shap_values(model_name: str, city: str = "default", n_samples: int = 200):
    """Compute SHAP feature importances for a saved model."""
    try:
        import shap
        from utils.db import load_model
        model, feat_cols, _ = load_model(model_name)
        if model is None:
            return None, None

        df        = load_features(city, limit=2000)
        feat_cols = [c for c in feat_cols if c in df.columns]
        X         = df[feat_cols].dropna().tail(n_samples).values

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        mean_shap  = np.abs(shap_vals).mean(axis=0)
        importance = pd.DataFrame({
            "feature":    feat_cols,
            "importance": mean_shap
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance, shap_vals
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return None, None
