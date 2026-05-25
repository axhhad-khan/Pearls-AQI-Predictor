import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']
from dotenv import load_dotenv
load_dotenv()
from utils.db import get_db

db  = get_db()
col = db['features']

print('Total Karachi docs:', col.count_documents({'city': 'Karachi'}))

latest = col.find_one({'city': 'Karachi'}, sort=[('timestamp', -1)])
oldest = col.find_one({'city': 'Karachi'}, sort=[('timestamp',  1)])
print('Latest:', latest['timestamp'])
print('Oldest:', oldest['timestamp'])

# May 15 ke baad
after = col.count_documents({
    'city': 'Karachi',
    'timestamp': {'$gt': '2026-05-15'}
})
print('Docs after May 15:', after)

# Sample timestamps
docs = list(col.find({'city': 'Karachi'}, {'timestamp': 1}).sort('timestamp', -1).limit(10))
print('\nLatest 10 timestamps:')
for d in docs:
    print(' ', d['timestamp'])