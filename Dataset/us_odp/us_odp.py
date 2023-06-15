import requests
import pandas as pd
import os
import string
import logging


def clean_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_filename = ''.join(c for c in filename if c in valid_chars)
    cleaned_filename = cleaned_filename.replace(' ', '_')
    return cleaned_filename


# Set up logging
logging.basicConfig(filename='log_us_odp.txt', level=logging.INFO, format='%(message)s')

# Make request to data.gov
response = requests.get("https://catalog.data.gov/api/3/action/package_search")
data = response.json()

path = "I:\\Datasets\\Feature_Discovery\\us_odp"
os.makedirs(path, exist_ok=True)

# Downloading
count = 0
for dataset in data['result']['results']:
    for resource in dataset['resources']:
        if resource['format'].lower() == 'csv':
            url = resource['url']
            filename = clean_filename(dataset['title']) + '.csv'
            logging.info(f'Downloading {url} to {filename}')

            try:
                data = pd.read_csv(url)
                data.to_csv(f'{path}/{filename}', index=False)
                count += 1
            except Exception as e:
                logging.info(f'Could not download {url} because {str(e)}')

            break
    if count >= 10:  # number limitation of datasets for downloading
        break

logging.info('Finished downloading datasets')
