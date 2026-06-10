"""
Fetch ENTSO-E hourly system load for 2015-2025, 22 countries.

Output:
  Analysis/data/ENTSOE_load_2015_2025.csv
    timestamp (UTC, hourly), one column per country (MW).

Pattern follows fetch_entsoe_generation_2015_2017.py:
  - Progressive save after each country
  - Resume capability: skip countries already in partial CSV
  - Monthly chunks (yearly is often too big for the API)
  - Early bailout: 3 consecutive empty months -> skip country
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
OUT = DATA_DIR / "ENTSOE_load_2015_2025.csv"

START_TS = pd.Timestamp("20150101", tz="UTC")
END_TS   = pd.Timestamp("20260101", tz="UTC")

COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
    "SI", "SK",
]

# For most countries the country code works. For DE we use the joint zone
# pre-2018 and DE_LU after; entsoe-py's "DE" alias handles this in some
# versions but to be safe we'll branch.
COUNTRY_MAP = {c: c for c in COUNTRIES}
COUNTRY_MAP["DE"] = "DE_LU"      # post 2018-10
DE_PRE_2018 = "DE_AT_LU"
COUNTRY_MAP["IE"] = "IE_SEM"
COUNTRY_MAP["DK"] = "DK_1"

MAX_RETRIES = 2
INITIAL_BACKOFF = 5
EARLY_BAIL_AFTER = 3

load_dotenv(PAPER1_DOTENV)
api_key = os.environ.get("ENTSOE_API_TOKEN")
if not api_key:
    print("ERROR: ENTSOE_API_TOKEN not found"); sys.exit(1)
client = EntsoePandasClient(api_key=api_key)


def month_range(start, end):
    cur = start
    while cur < end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt > end:
            nxt = end
        yield cur, nxt
        cur = nxt


def query_with_retry(cc, start, end):
    backoff = INITIAL_BACKOFF
    last = None
    for _ in range(MAX_RETRIES):
        try:
            df = client.query_load(country_code=cc, start=start, end=end)
            return df, None
        except requests.exceptions.HTTPError as ex:
            sc = getattr(ex.response, "status_code", None)
            last = f"HTTP {sc}"
            if sc in (503, 504):
                time.sleep(backoff); backoff *= 2; continue
            return None, last
        except Exception as ex:
            last = f"exc {str(ex)[:80]}"
            time.sleep(backoff); backoff *= 2; continue
    return None, last


print(f"Window: {START_TS.date()} -> {END_TS.date()}  ({len(COUNTRIES)} countries)\n")

all_series: dict[str, pd.Series] = {}
done_countries: set[str] = set()
if OUT.exists():
    print(f"Resume: loading partial {OUT}")
    existing = pd.read_csv(OUT, index_col="timestamp", parse_dates=True)
    existing.index = pd.to_datetime(existing.index, utc=True)
    for c in existing.columns:
        all_series[c] = existing[c]
        done_countries.add(c)
    print(f"  Already covered: {sorted(done_countries)}\n")


def save_partial():
    if not all_series:
        return
    df = pd.DataFrame(all_series).sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = "timestamp"
    df.to_csv(OUT)


errors = []
for label in COUNTRIES:
    if label in done_countries:
        print(f"--- {label}: SKIP (in partial)"); continue

    print(f"--- {label} ---")
    pieces = []
    consec_fail = 0
    bailed = False
    for s, e in month_range(START_TS, END_TS):
        # DE uses DE_AT_LU pre Oct 2018, DE_LU after
        if label == "DE" and s < pd.Timestamp("2018-10-01", tz="UTC"):
            cc_use = DE_PRE_2018
        else:
            cc_use = COUNTRY_MAP[label]
        df, err = query_with_retry(cc_use, s, e)
        if df is not None and not (df.empty if hasattr(df, 'empty') else len(df) == 0):
            if isinstance(df, pd.DataFrame):
                # entsoe-py query_load returns a DF with one column ("Actual Load")
                ser = df.iloc[:, 0]
            else:
                ser = df
            ser = ser.tz_convert("UTC") if ser.index.tz is not None else ser.tz_localize("UTC")
            pieces.append(ser)
            print(f"  {s.date()}->{e.date()}: {len(ser):>5} rows  (zone {cc_use})")
            consec_fail = 0
        else:
            print(f"  {s.date()}->{e.date()}: empty/error ({err})")
            consec_fail += 1
            if consec_fail >= EARLY_BAIL_AFTER:
                print(f"  -> BAILING on {label} after {EARLY_BAIL_AFTER} fails")
                bailed = True
                break
        time.sleep(0.4)

    if bailed or not pieces:
        errors.append((label, "no data / bailed"))
        save_partial()
        continue

    ser_all = pd.concat(pieces).sort_index()
    ser_all = ser_all[~ser_all.index.duplicated(keep="first")]
    # If entsoe returned 15-min resolution for some periods, resample to hourly
    ser_all = ser_all.resample("h").mean()
    all_series[label] = ser_all
    save_partial()
    print(f"  -> {label}: {len(ser_all)} hourly rows; saved partial "
          f"({len(all_series)} cols total)")


save_partial()
print(f"\n=== Final: {OUT} ({len(all_series)} cols) ===")
if errors:
    print("\nErrors / no-data:")
    for c, e in errors:
        print(f"  {c}: {e}")
