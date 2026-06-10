"""
Fetch annual installed generation capacity from ENTSO-E Transparency Platform
for 2015-2025, all 28 European countries.

Reuses Paper 1's API key. Annual resolution -> small fetch.

Output:
  Analysis/data/ENTSOE_installed_capacity_2015_2025.csv
    columns: country, year, tech_<n>...   (one row per country-year)
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
PAPER1_DOTENV = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/.env"
)

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "ENTSOE_installed_capacity_2015_2025.csv"

YEARS = list(range(2015, 2026))

COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "NL",
    "NO", "PL", "PT", "RO", "SE", "SI", "SK", "DK", "UK",
]
COUNTRY_MAP = {
    "AT": "AT", "BE": "BE", "BG": "BG", "CH": "CH", "CZ": "CZ",
    "DE": "DE_LU",          # post-2018 zone; for 2015-2017 fallback to DE_AT_LU
    "DK": "DK_1", "EE": "EE", "ES": "ES", "FI": "FI", "FR": "FR",
    "GR": "GR", "HR": "HR", "HU": "HU", "IE": "IE_SEM",
    "IT": "IT_NORD", "LT": "LT", "LU": "LU", "LV": "LV", "NL": "NL",
    "NO": "NO_2", "PL": "PL", "PT": "PT", "RO": "RO", "SE": "SE_3",
    "SI": "SI", "SK": "SK", "UK": "GB",
}
DE_PRE_2018 = "DE_AT_LU"

MAX_RETRIES = 2
INITIAL_BACKOFF = 5

load_dotenv(PAPER1_DOTENV)
api_key = os.environ.get("ENTSOE_API_TOKEN")
if not api_key:
    print("ERROR: ENTSOE_API_TOKEN not found"); sys.exit(1)
client = EntsoePandasClient(api_key=api_key)


def query_with_retry(cc: str, year: int):
    backoff = INITIAL_BACKOFF
    s = pd.Timestamp(f"{year}0101", tz="UTC")
    e = pd.Timestamp(f"{year+1}0101", tz="UTC")
    last_exc = None
    for _ in range(MAX_RETRIES):
        try:
            df = client.query_installed_generation_capacity(
                country_code=cc, start=s, end=e
            )
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


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
print(f"Fetching installed capacity {YEARS[0]}-{YEARS[-1]} for {len(COUNTRIES)} countries\n")
rows = []
errors = []
for label in COUNTRIES:
    cc = COUNTRY_MAP.get(label)
    print(f"--- {label} ({cc}) ---")
    for year in YEARS:
        # DE pre-2018 used the merged DE_AT_LU bidding zone
        cc_y = DE_PRE_2018 if (label == "DE" and year < 2018) else cc
        df, err = query_with_retry(cc_y, year)
        if df is None or df.empty:
            print(f"  {year}: empty/error ({err})")
            errors.append((label, year, err))
            continue
        # df is typically a 1-row DataFrame indexed by year-start, columns = tech
        if isinstance(df, pd.DataFrame):
            row = df.iloc[0].to_dict() if len(df) else {}
        else:
            row = df.to_dict()
        row = {k: float(v) for k, v in row.items() if pd.notna(v)}
        rows.append({"country": label, "year": year, **row})
        print(f"  {year}: {len(row)} carriers, total = "
              f"{sum(row.values())/1000:.1f} GW")
        time.sleep(0.3)

# ---------------------------------------------------------------------------
# Save (long->wide is more useful per-tech later, save the raw wide-tech form)
# ---------------------------------------------------------------------------
df_out = pd.DataFrame(rows).fillna(0.0)
# Reorder columns: country, year, then sorted tech cols
fixed = ["country", "year"]
tech_cols = [c for c in df_out.columns if c not in fixed]
df_out = df_out[fixed + sorted(tech_cols)]
df_out.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  ({len(df_out)} country-years, {len(tech_cols)} tech cols)")

if errors:
    print("\nErrors / no-data:")
    for c, y, e in errors:
        print(f"  {c} {y}: {e}")
