"""
db.py  -  MongoDB helpers (feature store + model registry).
"""

import os
import logging
import joblib
import io
import pickle
from datetime import datetime


import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError
from bson import Binary
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Connection ───────────────────────────────────────────────────────────────
_client = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
        )
    return _client


def get_db():
    db_name = os.getenv("MONGO_DB_NAME", "aqi_predictor")
    return get_client()[db_name]


# ── Feature Store ────────────────────────────────────────────────────────────
def upsert_features(df, city="default"):
    db   = get_db()
    col  = db["features"]
    col.create_index([("city", ASCENDING), ("timestamp", ASCENDING)], unique=True)

    records = df.copy()
    records["city"]      = city
    records["timestamp"] = records["timestamp"].astype(str)
    records = records.where(df.notna(), other=None)
    docs = records.to_dict(orient="records")

    inserted = 0
    for doc in docs:
        try:
            col.update_one(
                {"city": doc["city"], "timestamp": doc["timestamp"]},
                {"$set": doc},
                upsert=True
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"Upsert error: {e}")

    logger.info(f"Upserted {inserted} feature rows for city='{city}'")
    return inserted


def load_features(city="default", limit=5000):
    db  = get_db()
    col = db["features"]
    cursor = col.find({"city": city}, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
    docs = list(cursor)
    if not docs:
        return __import__("pandas").DataFrame()
    df = __import__("pandas").DataFrame(docs)
    df["timestamp"] = __import__("pandas").to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_forecast_features(city="default"):
    df  = load_features(city, limit=200)
    now = __import__("pandas").Timestamp.utcnow().tz_localize(None)
    if df.empty:
        return df
    return df[df["timestamp"] >= now].copy()


def save_model(model, model_name, metrics, feature_cols):
    db  = get_db()
    col = db["models"]
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    doc = {
        "name":         model_name,
        "trained_at":   datetime.utcnow().isoformat(),
        "metrics":      metrics,
        "feature_cols": feature_cols,
        "binary":       Binary(buf.read()),
    }
    col.update_one({"name": model_name}, {"$set": doc}, upsert=True)
    logger.info(f"Saved model '{model_name}' to registry. metrics={metrics}")


def load_model(model_name):
    db  = get_db()
    col = db["models"]
    doc = col.find_one({"name": model_name})
    if doc is None:
        return None, None, None
    buf   = io.BytesIO(doc["binary"])
    model = joblib.load(buf)
    return model, doc.get("feature_cols", []), doc.get("metrics", {})


def list_models():
    db  = get_db()
    col = db["models"]
    return list(col.find({}, {"_id": 0, "binary": 0}).sort("trained_at", DESCENDING))