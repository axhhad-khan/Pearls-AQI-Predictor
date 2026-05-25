import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']
from dotenv import load_dotenv
load_dotenv()
from utils.db import get_db
from pipelines.fetch_data import fetch_combined
from pipelines.feature_pipeline import compute_features
import pandas as pd

# Fresh data fetch karo
print("Fetching May 15 - May 21 data...")
raw  = fetch_combined(24.8607, 67.0011, 
                      start_date='2026-05-15', 
                      end_date='2026-05-21')
feat = compute_features(raw)

print("Fetched rows:", len(feat))
print("Sample timestamps:", feat['timestamp'].head(3).tolist())
print("Timestamp type:", type(feat['timestamp'].iloc[0]))

# Manual insert with string conversion
db  = get_db()
col = db['features']

feat['city']      = 'Karachi'
feat['timestamp'] = feat['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Replace NaN with None
feat = feat.where(feat.notna(), other=None)
docs = feat.to_dict(orient='records')

inserted = 0
for doc in docs:
    try:
        col.update_one(
            {'city': doc['city'], 'timestamp': doc['timestamp']},
            {'$set': doc},
            upsert=True
        )
        inserted += 1
    except Exception as e:
        print(f"Error: {e}")

print(f"Inserted: {inserted} rows")
print("Total Karachi docs now:", col.count_documents({'city': 'Karachi'}))
latest = col.find_one({'city': 'Karachi'}, sort=[('timestamp', -1)])
print("Latest:", latest['timestamp'])   