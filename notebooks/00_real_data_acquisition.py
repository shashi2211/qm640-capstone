"""
REAL CMS/Census data acquisition script.

It uses CMS's documented public Data
API (https://data.cms.gov/data-api/v1/dataset/{distribution-id}/data),
which is the same REST pattern CMS uses for every dataset on data.cms.gov.

HOW TO GET THE DISTRIBUTION ID FOR EACH DATASET (one-time manual step):
  1. Open the dataset's page on data.cms.gov (links below).
  2. Click the "API Access" tab.
  3. Copy the "Distribution ID" shown there into DATASET_IDS below.
These IDs occasionally change when CMS refreshes a dataset's vintage, so
re-check them if a request below returns a 404.
"""
import requests
import pandas as pd
import time

BASE = "https://data.cms.gov/data-api/v1/dataset/{dist_id}/data"

# Fill in each distribution ID from the dataset's "API Access" tab.
DATASET_IDS = {
    "dme_supplier": "<PASTE_DISTRIBUTION_ID_HERE>",
    # https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-supplier
    "dme_referring_provider": "<PASTE_DISTRIBUTION_ID_HERE>",
    # https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider-and-service
    "geographic_variation": "<PASTE_DISTRIBUTION_ID_HERE>",
    # https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county
}


def fetch_dataset(dist_id, page_size=5000, max_pages=None):
    """Paginates through a CMS Data API dataset and returns a concatenated DataFrame."""
    records, offset, page = [], 0, 0
    while True:
        url = BASE.format(dist_id=dist_id)
        resp = requests.get(url, params={"size": page_size, "offset": offset}, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        offset += page_size
        page += 1
        print(f"  fetched page {page}, running total {len(records)} rows")
        if max_pages and page >= max_pages:
            break
        time.sleep(0.25)  # be polite to the API
    return pd.DataFrame(records)


def fetch_acs_county(api_key, year=2023):
    """American Community Survey 5-Year Estimates, county-level income/poverty/rurality."""
    # Variables: B19013_001E = median household income, B17001_002E = count below poverty
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B19013_001E,B17001_002E,B01003_001E",
        "for": "county:*",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data[1:], columns=data[0])


if __name__ == "__main__":
    import os

    os.makedirs("../data/raw", exist_ok=True)

    print("Fetching DME Supplier PUF...")
    supplier_df = fetch_dataset(DATASET_IDS["dme_supplier"])
    supplier_df.to_csv("../data/raw/real_dme_supplier.csv", index=False)

    print("Fetching DME Referring Provider PUF...")
    provider_df = fetch_dataset(DATASET_IDS["dme_referring_provider"])
    provider_df.to_csv("../data/raw/real_dme_referring_provider.csv", index=False)

    print("Fetching Geographic Variation PUF...")
    geo_df = fetch_dataset(DATASET_IDS["geographic_variation"])
    geo_df.to_csv("../data/raw/real_geographic_variation.csv", index=False)

    # ACS requires a free API key from https://api.census.gov/data/key_signup.html
    census_key = os.environ.get("CENSUS_API_KEY", "")
    if census_key:
        print("Fetching ACS county covariates...")
        acs_df = fetch_acs_county(census_key)
        acs_df.to_csv("../data/raw/real_acs_county.csv", index=False)
    else:
        print("Skipping ACS pull: set the CENSUS_API_KEY environment variable to enable it.")

    print("Done. Replace the pilot CSVs referenced in 02_analysis_pipeline.py with these real files.")
