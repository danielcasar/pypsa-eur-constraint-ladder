"""
Refetch ENTSO-E installed capacity for multi-zone countries (NO, SE, DK, IT)
trying all known bidding-zone codes per country.

Earlier fetch_entsoe_installed_capacity.py used one zone per country and got
empty data for IT and SE. This script tries every zone code per country and
accumulates whatever ENTSO-E returns.

Output:
  Analysis/data/ENTSOE_installed_capacity_multizone.csv
    columns: country, zone, year, <tech1, tech2, ...>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "ENTSOE_installed_capacity_multizone.csv"

PAPER1_DOTENV = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/.env"
)

# Zones to try per country. The first that returns non-empty data per year
# is kept; multiple may succeed.
ZONES_TO_TRY = {
    "IT": ["IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD", "IT_SARD", "IT_SICI",
           "IT_BRNN", "IT_CALA", "IT_PRGP", "IT"],
    "SE": ["SE_1", "SE_2", "SE_3", "SE_4", "SE"],
    "NO": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5", "NO"],
    "DK": ["DK_1", "DK_2", "DK"],
}

YEARS = list(range(2015, 2026))

MAX_RETRIES = 2
INITIAL_BACKOFF = 5

load_dotenv(PAPER1_DOTENV)
api_key = os.environ.get("ENTSOE_API_TOKEN")
if not api_key:
    print("ERROR: ENTSOE_API_TOKEN not found"); sys.exit(1)
client = EntsoePandasClient(api_key=api_key)


def query_with_retry(area, year):
    backoff = INITIAL_BACKOFF
    s = pd.Timestamp(f"{year}0101", tz="UTC")
    e = pd.Timestamp(f"{year+1}0101", tz="UTC")
    last_exc = None
    for _ in range(MAX_RETRIES):
        try:
            df = client.query_installed_generation_capacity(
                country_code=area, start=s, end=e)
            return df, None
        except requests.exceptions.HTTPError as ex:
            sc = getattr(ex.response, "status_code", None)
            last_exc = f"HTTP {sc}"
            if sc in (503, 504):
                time.sleep(backoff); backoff *= 2; continue
            return None, last_exc
        except Exception as ex:
            last_exc = f"exc {str(ex)[:80]}"
            time.sleep(backoff); backoff *= 2; continue
    return None, last_exc


rows = []
for country, zones in ZONES_TO_TRY.items():
    print(f"\n=== {country} ===")
    for zone in zones:
        zone_has_data = False
        for year in YEARS:
            df, err = query_with_retry(zone, year)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            if isinstance(df, pd.DataFrame):
                row = df.iloc[0].to_dict() if len(df) else {}
            else:
                row = df.to_dict()
            row = {k: float(v) for k, v in row.items() if pd.notna(v)}
            if not row:
                continue
            rows.append({"country": country, "zone": zone, "year": year, **row})
            zone_has_data = True
            print(f"  {zone}  {year}: {len(row)} carriers, total = "
                  f"{sum(row.values())/1000:.1f} GW")
            time.sleep(0.3)
        if not zone_has_data:
            print(f"  {zone}: NO DATA across all years")

df_out = pd.DataFrame(rows)
if df_out.empty:
    print("\nNo data fetched.")
    sys.exit(0)

df_out = df_out.fillna(0.0)
fixed = ["country", "zone", "year"]
tech_cols = sorted([c for c in df_out.columns if c not in fixed])
df_out = df_out[fixed + tech_cols]
df_out.to_csv(OUT, index=False)

print(f"\nWrote {OUT}")
print(f"  rows: {len(df_out)}; countries: {df_out['country'].nunique()}; "
      f"zones: {df_out['zone'].nunique()}")
print("\nCoverage per zone (years):")
print(df_out.groupby(["country", "zone"])["year"].count().to_string())
