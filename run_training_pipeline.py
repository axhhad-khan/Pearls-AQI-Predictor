"""run_training_pipeline.py — daily CI/CD job"""
import os, logging, dns.resolver

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

from pipelines.training_pipeline import train

city = os.getenv('CITY_NAME', 'Karachi') or 'Karachi'
print(f'Training for city={city}...')
results = train(city=city)

print('\n=== Training Results ===')
for name, metrics in results.items():
    print(f'  {name:15s}: {metrics}')