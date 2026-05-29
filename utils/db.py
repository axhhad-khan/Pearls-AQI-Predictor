"""db.py - MongoDB Feature Store + Model Registry"""
import os, io, logging, sys
from datetime import datetime

import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

import certifi, joblib, pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import Binary

# CI/CD safe: read from os.environ first, then .env file
if not os.environ.get("MONGO_URI"):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except:
        pass

logger  = logging.getLogger(__name__)
_client = None

def get_client():
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI", "").strip()
        if not uri:
            print("ERROR: MONGO_URI is empty!")
            sys.exit(1)
        logger.info(f"MongoDB connecting: {uri[:35]}...")
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
    return get_client()[os.environ.get("MONGO_DB_NAME","aqi_predictor")]

def upsert_features(df, city="default"):
    db  = get_db()
    col = db["features"]
    col.create_index([("city",ASCENDING),("timestamp",ASCENDING)], unique=True)
    records = df.copy()
    records["city"] = city
    if pd.api.types.is_datetime64_any_dtype(records["timestamp"]):
        records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        records["timestamp"] = pd.to_datetime(records["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    records = records.where(pd.notna(records), other=None)
    inserted = 0
    for doc in records.to_dict(orient="records"):
        try:
            col.update_one({"city":doc["city"],"timestamp":doc["timestamp"]},
                           {"$set":doc}, upsert=True)
            inserted += 1
        except Exception as e:
            logger.warning(f"Upsert: {e}")
    logger.info(f"Upserted {inserted} rows for '{city}'")
    return inserted

def load_features(city="default", limit=5000):
    db   = get_db()
    col  = db["features"]
    docs = list(col.find({"city":city},{"_id":0}).sort("timestamp",DESCENDING).limit(limit))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def load_forecast_features(city="default"):
    df  = load_features(city, 200)
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df[df["timestamp"] >= now].copy() if not df.empty else df

def save_model(model, name, metrics, feature_cols):
    db  = get_db()
    col = db["models"]
    buf = io.BytesIO()
    joblib.dump(model, buf); buf.seek(0)
    col.update_one({"name":name},
        {"$set":{"name":name,"trained_at":datetime.utcnow().isoformat(),
                 "metrics":metrics,"feature_cols":feature_cols,
                 "binary":Binary(buf.read())}}, upsert=True)
    logger.info(f"Saved '{name}': {metrics}")

def load_model(name):
    db  = get_db()
    doc = db["models"].find_one({"name":name})
    if not doc: return None,None,None
    return joblib.load(io.BytesIO(doc["binary"])), doc.get("feature_cols",[]), doc.get("metrics",{})

def list_models():
    db = get_db()
    return list(db["models"].find({},{"_id":0,"binary":0}).sort("trained_at",DESCENDING))