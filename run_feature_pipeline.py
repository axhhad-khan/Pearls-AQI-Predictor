# """run_feature_pipeline.py — hourly CI/CD job"""
# import os, logging, dns.resolver

# dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
# dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

# from dotenv import load_dotenv
# load_dotenv()
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')

# from pipelines.fetch_data import fetch_combined
# from pipelines.feature_pipeline import compute_features
# from utils.db import upsert_features

# # Safe float conversion with fallback
# def safe_float(val, default):
#     try:
#         return float(val) if val and str(val).strip() else default
#     except:
#         return default

# lat  = safe_float(os.getenv('LATITUDE'),  24.8607)
# lon  = safe_float(os.getenv('LONGITUDE'), 67.0011)
# city = os.getenv('CITY_NAME', 'Karachi') or 'Karachi'

# print(f'City={city} | Lat={lat} | Lon={lon}')
# raw  = fetch_combined(lat, lon)
# feat = compute_features(raw)
# n    = upsert_features(feat, city=city)
# print(f'Done — {n} rows upserted for {city}.')

"""run_feature_pipeline.py — hourly CI/CD job"""

import os, logging, dns.resolver

# DNS Configuration
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ["8.8.8.8", "8.8.4.4"]

from dotenv import load_dotenv

load_dotenv()

# Logging setup for CI/CD tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pipelines.fetch_data import fetch_combined
from pipelines.feature_pipeline import compute_features
from utils.db import upsert_features


def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() else default
    except:
        return default


def main():
    lat = safe_float(os.getenv("LATITUDE"), 24.8607)
    lon = safe_float(os.getenv("LONGITUDE"), 67.0011)
    city = os.getenv("CITY_NAME", "Karachi") or "Karachi"

    logging.info(f"Starting pipeline for: {city} (Lat: {lat}, Lon: {lon})")

    try:
        raw = fetch_combined(lat, lon)
        feat = compute_features(raw)  # Yahan naye features compute honge
        n = upsert_features(feat, city=city)
        logging.info(f"Successfully upserted {n} rows for {city}.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise e  # GitHub Action ko batane ke liye ke step fail ho gaya


if __name__ == "__main__":
    main()
