"""run_feature_pipeline.py — hourly CI/CD job"""
import os, logging, dns.resolver

# Fix DNS for GitHub Actions / cloud environments
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

from pipelines.fetch_data import fetch_combined
from pipelines.feature_pipeline import compute_features
from utils.db import upsert_features

lat  = float(os.getenv('LATITUDE',  '24.8607'))
lon  = float(os.getenv('LONGITUDE', '67.0011'))
city = os.getenv('CITY_NAME', 'Karachi')

print(f'Fetching data for {city} ({lat}, {lon})...')
raw  = fetch_combined(lat, lon)
feat = compute_features(raw)
n    = upsert_features(feat, city=city)
print(f'Feature pipeline done — {n} rows upserted for {city}.')
