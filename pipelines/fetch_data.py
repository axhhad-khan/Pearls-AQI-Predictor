"""
fetch_data.py
Fetches weather + air quality data from Open-Meteo (completely free, no API key).
Docs: https://open-meteo.com/en/docs/air-quality-api
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open-Meteo endpoints
AQ_URL   = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL   = "https://api.open-meteo.com/v1/forecast"


def fetch_air_quality(lat: float, lon: float,
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch hourly air-quality data.
    If no dates supplied, fetches the current day + 3-day forecast.
    """
    if start_date is None:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if end_date is None:
        end_date   = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")

    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "aerosol_optical_depth",
            "dust", "uv_index", "european_aqi", "us_aqi"
        ],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "auto",
    }

    logger.info(f"Fetching AQ data {start_date} → {end_date}")
    resp = requests.get(AQ_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latitude"]  = lat
    df["longitude"] = lon
    return df


def fetch_weather(lat: float, lon: float,
                  start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch hourly weather data (temperature, humidity, wind, pressure)."""
    if start_date is None:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if end_date is None:
        end_date   = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")

    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "surface_pressure", "precipitation",
            "cloud_cover", "visibility"
        ],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "auto",
    }

    logger.info(f"Fetching weather data {start_date} → {end_date}")
    resp = requests.get(WX_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fetch_combined(lat: float, lon: float,
                   start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Merge air-quality + weather into one DataFrame."""
    aq  = fetch_air_quality(lat, lon, start_date, end_date)
    wx  = fetch_weather(lat, lon, start_date, end_date)
    df  = pd.merge(aq, wx, on="timestamp", how="inner")
    logger.info(f"Combined dataset shape: {df.shape}")
    return df
