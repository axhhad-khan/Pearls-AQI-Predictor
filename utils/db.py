"""db.py - MongoDB Feature Store + Model Registry"""
import os, io, logging
from datetime import datetime

import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

import certifi
import joblib
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import Binary
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = None

def get_client():
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
        )
    return _client

def get_db():
    return get_client()[os.getenv("MONGO_DB_NAME", "aqi_predictor")]

# ── Feature Store ─────────────────────────────────────────────────────────────
def upsert_features(df: pd.DataFrame, city: str = "default") -> int:
    db  = get_db()
    col = db["features"]
    col.create_index([("city", ASCENDING), ("timestamp", ASCENDING)], unique=True)

    records = df.copy()
    records["city"]      = city
    # Normalize timestamp to string
    if pd.api.types.is_datetime64_any_dtype(records["timestamp"]):
        records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        records["timestamp"] = pd.to_datetime(records["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    records = records.where(pd.notna(records), other=None)
    docs    = records.to_dict(orient="records")

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

    logger.info(f"Upserted {inserted} rows for city='{city}'")
    return inserted

def load_features(city: str = "default", limit: int = 5000) -> pd.DataFrame:
    db  = get_db()
    col = db["features"]
    docs = list(col.find({"city": city}, {"_id": 0})
                   .sort("timestamp", DESCENDING)
                   .limit(limit))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_forecast_features(city: str = "default") -> pd.DataFrame:
    df  = load_features(city, limit=200)
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df[df["timestamp"] >= now].copy() if not df.empty else df

# ── Model Registry ────────────────────────────────────────────────────────────
def save_model(model, name: str, metrics: dict, feature_cols: list):
    db  = get_db()
    col = db["models"]
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    doc = {
        "name":         name,
        "trained_at":   datetime.utcnow().isoformat(),
        "metrics":      metrics,
        "feature_cols": feature_cols,
        "binary":       Binary(buf.read()),
    }
    col.update_one({"name": name}, {"$set": doc}, upsert=True)
    logger.info(f"Saved model '{name}': {metrics}")

def load_model(name: str):
    db  = get_db()
    col = db["models"]
    doc = col.find_one({"name": name})
    if not doc:
        return None, None, None
    model = joblib.load(io.BytesIO(doc["binary"]))
    return model, doc.get("feature_cols", []), doc.get("metrics", {})

def list_models():
    db  = get_db()
    col = db["models"]
    return list(col.find({}, {"_id": 0, "binary": 0}).sort("trained_at", DESCENDING))
