"""backfill.py — historical data backfill"""
import argparse, logging, os, dns.resolver

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() else default
    except:
        return default

def backfill(lat, lon, city, days=30):
    from pipelines.fetch_data import fetch_combined
    from pipelines.feature_pipeline import compute_features
    from utils.db import upsert_features

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    chunk    = 14
    cursor   = start_dt
    total    = 0

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk), end_dt)
        s = cursor.strftime('%Y-%m-%d')
        e = chunk_end.strftime('%Y-%m-%d')
        try:
            raw  = fetch_combined(lat, lon, start_date=s, end_date=e)
            if raw.empty:
                logger.warning(f'No data for {s} → {e}')
            else:
                feat  = compute_features(raw)
                n     = upsert_features(feat, city=city)
                total += n
                logger.info(f'✓ {s} → {e}: {n} rows')
        except Exception as ex:
            logger.error(f'✗ {s} → {e}: {ex}')
        cursor = chunk_end

    logger.info(f'Backfill complete. Total: {total} rows')
    return total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int,   default=30)
    parser.add_argument('--city', type=str,   default=None)
    parser.add_argument('--lat',  type=float, default=None)
    parser.add_argument('--lon',  type=float, default=None)
    args = parser.parse_args()

    city = args.city or os.getenv('CITY_NAME','Karachi') or 'Karachi'
    lat  = args.lat  or safe_float(os.getenv('LATITUDE'),  24.8607)
    lon  = args.lon  or safe_float(os.getenv('LONGITUDE'), 67.0011)

    backfill(lat=lat, lon=lon, city=city, days=args.days)