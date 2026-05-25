"""
fix_city2.py - nan docs delete karo (Karachi duplicates already exist)
"""
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']

from dotenv import load_dotenv
load_dotenv()
from utils.db import get_db

db  = get_db()
col = db['features']

print("Before fix:")
print("  Total docs     :", col.count_documents({}))
print("  Karachi docs   :", col.count_documents({"city": "Karachi"}))
print("  nan city docs  :", col.count_documents({"city": None}))

# nan wale docs seedha delete karo — Karachi data already hai
result = col.delete_many({"city": None})
print(f"\nDeleted {result.deleted_count} nan-city docs")

print("\nAfter fix:")
print("  Total docs     :", col.count_documents({}))
print("  Karachi docs   :", col.count_documents({"city": "Karachi"}))
print("  Distinct cities:", col.distinct('city'))

# Latest/oldest verify
latest = col.find_one({"city": "Karachi"}, sort=[('timestamp', -1)])
oldest = col.find_one({"city": "Karachi"}, sort=[('timestamp',  1)])
print(f"\n  Latest : {latest['timestamp'] if latest else 'None'}")
print(f"  Oldest : {oldest['timestamp'] if oldest else 'None'}")