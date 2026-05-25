"""
fix_city.py - nan city ko Karachi mein fix karo
"""
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']

from dotenv import load_dotenv
load_dotenv()
from utils.db import get_db

db  = get_db()
col = db['features']

# nan city wale docs ko Karachi mein update karo
result = col.update_many(
    {"city": {"$in": [None, float('nan'), "nan"]}},
    {"$set": {"city": "Karachi"}}
)
print(f"Fixed {result.modified_count} docs — nan → Karachi")

# Verify
print("Distinct cities now:", col.distinct('city'))
print("Total Karachi docs:", col.count_documents({"city": "Karachi"}))