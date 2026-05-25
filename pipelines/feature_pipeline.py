"""feature_pipeline.py - Feature engineering"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

AQI_BREAKPOINTS = [
    (0,   50,  "Good",                          "#00e400"),
    (51,  100, "Moderate",                       "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy",                      "#ff0000"),
    (201, 300, "Very Unhealthy",                 "#8f3f97"),
    (301, 500, "Hazardous",                      "#7e0023"),
]

def aqi_category(val):
    if pd.isna(val):
        return "Unknown", "#888888"
    for lo, hi, label, color in AQI_BREAKPOINTS:
        if lo <= val <= hi:
            return label, color
    return "Hazardous", "#7e0023"

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # AQI
    if "us_aqi" in df.columns:
        df["aqi"] = pd.to_numeric(df["us_aqi"], errors="coerce")
    elif "european_aqi" in df.columns:
        df["aqi"] = pd.to_numeric(df["european_aqi"], errors="coerce")
    else:
        raise ValueError("No AQI column found.")

    # Lag + rolling features
    df["aqi_lag_1h"]           = df["aqi"].shift(1)
    df["aqi_lag_3h"]           = df["aqi"].shift(3)
    df["aqi_lag_6h"]           = df["aqi"].shift(6)
    df["aqi_lag_24h"]          = df["aqi"].shift(24)
    df["aqi_change_1h"]        = df["aqi"].diff(1)
    df["aqi_change_3h"]        = df["aqi"].diff(3)
    df["aqi_rolling_6h_mean"]  = df["aqi"].rolling(6,  min_periods=1).mean()
    df["aqi_rolling_24h_mean"] = df["aqi"].rolling(24, min_periods=1).mean()

    # Derived features
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        wd = np.deg2rad(df["wind_direction_10m"])
        df["wind_u"] = df["wind_speed_10m"] * np.cos(wd)
        df["wind_v"] = df["wind_speed_10m"] * np.sin(wd)

    # Category
    df["aqi_category"] = df["aqi"].apply(lambda x: aqi_category(x)[0])
    df["aqi_color"]    = df["aqi"].apply(lambda x: aqi_category(x)[1])

    # Targets — 6h, 12h, 24h, 48h, 72h
    df["target_aqi_6h"]  = df["aqi"].shift(-6)
    df["target_aqi_12h"] = df["aqi"].shift(-12)
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    df["fetched_at"] = datetime.utcnow().isoformat()
    return df

FEATURE_COLS = [
    "aqi",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend", "day_of_year",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "surface_pressure", "precipitation", "cloud_cover",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_1h", "aqi_change_3h",
    "aqi_rolling_6h_mean", "aqi_rolling_24h_mean",
    "pm_ratio", "wind_u", "wind_v",
]

def get_valid_feature_cols(df):
    return [c for c in FEATURE_COLS if c in df.columns]
