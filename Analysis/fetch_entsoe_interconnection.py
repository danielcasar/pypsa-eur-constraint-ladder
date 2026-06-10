"""
Fetch annual Net Transfer Capacity (NTC) between European bidding zones for
2015-2025. Per country, aggregate the sum across all incident borders as a
proxy for total interconnection capacity.

Strategy:
  - For each (border_a, border_b, year), try query_net_transfer_capacity_yearahead
    first (one annual number per direction). Fall back to dayahead aggregated
    (annual median of daily NTC) if yearahead is empty.
  - Per country, sum NTC inbound + outbound across all incident borders, then
    divide by 2 to avoid double-counting (each border counted once per side).
  - Save raw per-border NTCs plus per-country aggregates.

Output:
  Analysis/data/ENTSOE_interconnection_capacity_2015_2025.csv
    columns: country, year, total_ntc_mw, n_borders, source

  Analysis/data/ENTSOE_interconnection_per_border_2015_2025.csv (debug)
    columns: from, to, year, ntc_mw, source
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

OUT_AGG = DATA_DIR / "ENTSOE_interconnection_capacity_2015_2025.csv"
OUT_RAW = DATA_DIR / "ENTSOE_interconnection_per_border_2015_2025.csv"

YEARS = list(range(2015, 2026))

# Each pair listed ONCE — we'll fetch both directions in code.
# Areas chosen to match what entsoe-py / Transparency Platform accept.
# For multi-zone countries (DK, NO, SE), we use the dominant bidding zone
# bordering the listed neighbour. Coverage is best-effort.
BORDERS: list[tuple[str, str]] = [
    ("AT", "CZ"), ("AT", "DE_LU"), ("AT", "HU"), ("AT", "IT_NORD"),
    ("AT", "SI"), ("AT", "CH"),
    ("BE", "DE_LU"), ("BE", "FR"), ("BE", "NL"), ("BE", "GB"), ("BE", "LU"),
    ("BG", "GR"), ("BG", "RO"),
    ("CH", "DE_LU"), ("CH", "FR"), ("CH", "IT_NORD"),
    ("CZ", "DE_LU"), ("CZ", "PL"), ("CZ", "SK"),
    ("DE_LU", "DK_1"), ("DE_LU", "FR"), ("DE_LU", "NL"), ("DE_LU", "PL"),
    ("DE_LU", "SE_4"),
    ("DK_1", "NO_2"), ("DK_1", "SE_3"),
    ("DK_2", "DE_LU"), ("DK_2", "SE_4"),
    ("EE", "FI"), ("EE", "LV"),
    ("ES", "FR"), ("ES", "PT"),
    ("FI", "NO_4"), ("FI", "SE_1"), ("FI", "SE_3"),
    ("FR", "GB"), ("FR", "IT_NORD"),
    ("GR", "IT_BRNN"),  # GR-IT via Brindisi
    ("HR", "HU"), ("HR", "SI"),
    ("HU", "RO"), ("HU", "SK"), ("HU", "RS"),
    ("IE_SEM", "GB"),
    ("LT", "LV"), ("LT", "PL"), ("LT", "SE_4"),
    ("NL", "GB"), ("NL", "NO_2"),
    ("PL", "SE_4"), ("PL", "SK"),
    ("RO", "RS"),
    ("SI", "IT_NORD"),
]

MAX_RETRIES = 2
INITIAL_BACKOFF = 5

load_dotenv(PAPER1_DOTENV)
api_key = os.environ.get("ENTSOE_API_TOKEN")
if not api_key:
    print("ERROR: ENTSOE_API_TOKEN not found"); sys.exit(1)
client = EntsoePandasClient(api_key=api_key)


def annual_ntc(area_from: str, area_to: str, year: int):
    """Return (ntc_mw, source) — best-effort annual NTC, both directions averaged.

    source ∈ {"yearahead", "dayahead", None}
    """
    s = pd.Timestamp(f"{year}0101", tz="UTC")
    e = pd.Timestamp(f"{year+1}0101", tz="UTC")
    ntc_vals = []
    src_used = None
    for direction in [(area_from, area_to), (area_to, area_from)]:
        a, b = direction
        # Try yearahead first
        v = None
        for attempt in range(MAX_RETRIES):
            try:
                ser = client.query_net_transfer_capacity_yearahead(a, b, start=s, end=e)
                if ser is not None and len(ser) > 0 and not pd.Series(ser).isna().all():
                    v = float(pd.Series(ser).median())
                    src_used = "yearahead"
                    break
                else:
                    break
            except requests.exceptions.HTTPError as ex:
                sc = getattr(ex.response, "status_code", None)
                if sc in (503, 504):
                    time.sleep(INITIAL_BACKOFF); continue
                break
            except Exception:
                break
        # Fall back to dayahead-aggregated
        if v is None:
            for attempt in range(MAX_RETRIES):
                try:
                    ser = client.query_net_transfer_capacity_dayahead(a, b, start=s, end=e)
                    if ser is not None and len(ser) > 0 and not pd.Series(ser).isna().all():
                        v = float(pd.Series(ser).median())
                        src_used = "dayahead"
                        break
                    else:
                        break
                except requests.exceptions.HTTPError as ex:
                    sc = getattr(ex.response, "status_code", None)
                    if sc in (503, 504):
                        time.sleep(INITIAL_BACKOFF); continue
                    break
                except Exception:
                    break
        if v is not None:
            ntc_vals.append(v)
    if not ntc_vals:
        return None, None
    # Average both directions
    return float(sum(ntc_vals) / len(ntc_vals)), src_used


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
print(f"Fetching NTC 2015-2025 across {len(BORDERS)} border pairs (both directions)\n")
raw_rows = []
for i, (a, b) in enumerate(BORDERS):
    print(f"--- [{i+1}/{len(BORDERS)}] {a} <-> {b} ---")
    for y in YEARS:
        v, src = annual_ntc(a, b, y)
        if v is None:
            print(f"  {y}: no data")
        else:
            print(f"  {y}: {v:>7.0f} MW  ({src})")
            raw_rows.append({"from": a, "to": b, "year": y,
                             "ntc_mw": v, "source": src})
        time.sleep(0.3)

raw = pd.DataFrame(raw_rows)
raw.to_csv(OUT_RAW, index=False)


# ---------------------------------------------------------------------------
# Aggregate per country
# ---------------------------------------------------------------------------
# Each ENTSO-E area maps to one of our 22 country codes:
AREA_TO_COUNTRY = {
    "AT": "AT", "BE": "BE", "BG": "BG", "CH": "CH", "CZ": "CZ",
    "DE_LU": "DE", "DK_1": "DK", "DK_2": "DK", "EE": "EE", "ES": "ES",
    "FI": "FI", "FR": "FR", "GB": "UK", "GR": "GR", "HR": "HR",
    "HU": "HU", "IE_SEM": "IE", "IT_NORD": "IT", "IT_BRNN": "IT",
    "LT": "LT", "LU": "LU", "LV": "LV", "NL": "NL", "NO_2": "NO",
    "NO_4": "NO", "PL": "PL", "PT": "PT", "RO": "RO", "RS": "RS",
    "SE_1": "SE", "SE_3": "SE", "SE_4": "SE", "SI": "SI", "SK": "SK",
}

agg_rows = []
for y in YEARS:
    sub = raw[raw["year"] == y]
    if sub.empty:
        continue
    per_country = {}
    border_count = {}
    for _, r in sub.iterrows():
        cf = AREA_TO_COUNTRY.get(r["from"])
        ct = AREA_TO_COUNTRY.get(r["to"])
        if cf is None or ct is None or cf == ct:
            continue
        per_country.setdefault(cf, 0.0)
        per_country.setdefault(ct, 0.0)
        per_country[cf] += r["ntc_mw"]
        per_country[ct] += r["ntc_mw"]
        border_count[cf] = border_count.get(cf, 0) + 1
        border_count[ct] = border_count.get(ct, 0) + 1
    for c, total in per_country.items():
        agg_rows.append({
            "country": c, "year": int(y),
            "total_ntc_mw": float(total),
            "n_borders": int(border_count.get(c, 0)),
        })

agg = pd.DataFrame(agg_rows).sort_values(["country", "year"]).reset_index(drop=True)
agg.to_csv(OUT_AGG, index=False)
print(f"\nWrote {OUT_AGG}  ({len(agg)} country-years, "
      f"{agg['country'].nunique()} countries)")
print(f"\n{agg.head(15)}")
