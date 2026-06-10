"""
Fetch ENTSO-E hourly generation by carrier for 2015-2017 (the gap not covered
by Paper 1's existing ENTSOE_generation_mix.csv which starts at 2018-01-01).

Improvements over v1:
  - Saves PROGRESSIVELY after each country (resume-safe)
  - Skips countries already in the partial CSV (resume capability)
  - Fast bailout: if the first 3 months of a country all return empty/error,
    skip the country entirely (Croatia-style problem)

Reuses Paper 1's ENTSOE_API_TOKEN. Saves results to:
  Analysis/data/ENTSOE_generation_mix_2015_2017.csv  (per-country, accumulating)
And after final pass, merges with the existing 2018-2025 file into:
  Analysis/data/ENTSOE_generation_mix_2015_2025.csv
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PAPER1_DOTENV = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/.env"
)
PAPER1_GEN = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/data/ENTSOE_generation_mix.csv"
)

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARTIAL = DATA_DIR / "ENTSOE_generation_mix_2015_2017.csv"
OUT_MERGED  = DATA_DIR / "ENTSOE_generation_mix_2015_2025.csv"

START_TS = pd.Timestamp("20150101", tz="UTC")
END_TS   = pd.Timestamp("20180101", tz="UTC")

COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "NL",
    "NO", "PL", "PT", "RO", "SE", "SI", "SK", "DK", "UK",
]
COUNTRY_MAP = {
    "AT": "AT", "BE": "BE", "BG": "BG", "CH": "CH", "CZ": "CZ",
    "DE": "DE_AT_LU",
    "DK": "DK_1", "EE": "EE", "ES": "ES", "FI": "FI", "FR": "FR",
    "GR": "GR", "HR": "HR", "HU": "HU", "IE": "IE_SEM",
    "IT": "IT_NORD", "LT": "LT", "LU": "LU", "LV": "LV", "NL": "NL",
    "NO": "NO_2", "PL": "PL", "PT": "PT", "RO": "RO", "SE": "SE_3",
    "SI": "SI", "SK": "SK", "UK": "GB",
}

MAX_RETRIES = 2          # was 4; reduce wasted time on persistently bad months
INITIAL_BACKOFF = 5      # seconds; was 10
EARLY_BAIL_AFTER = 3     # consecutive empty/failed months -> skip country


def month_range(start, end):
    cur = start
    while cur < end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt > end:
            nxt = end
        yield cur, nxt
        cur = nxt


def query_with_retry(client, cc, start, end):
    backoff = INITIAL_BACKOFF
    last_exc: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = client.query_generation(country_code=cc, start=start, end=end, psr_type=None)
            return df, None
        except requests.exceptions.HTTPError as e:
            sc = getattr(e.response, "status_code", None)
            last_exc = f"HTTP {sc}: {str(e)[:60]}"
            if sc in (503, 504):
                time.sleep(backoff); backoff *= 2; continue
            return None, last_exc                          # fast-fail on real HTTP errors
        except Exception as e:
            last_exc = f"exc: {str(e)[:80]}"
            time.sleep(backoff); backoff *= 2; continue
    return None, f"failed after {MAX_RETRIES}: {last_exc}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv(PAPER1_DOTENV)
api_key = os.environ.get("ENTSOE_API_TOKEN")
if not api_key:
    print("ERROR: ENTSOE_API_TOKEN not found"); sys.exit(1)
client = EntsoePandasClient(api_key=api_key)
print(f"Window: {START_TS.date()} to {END_TS.date()}  ({len(COUNTRIES)} countries)")
print(f"MAX_RETRIES={MAX_RETRIES}, EARLY_BAIL_AFTER={EARLY_BAIL_AFTER}\n")

# Resume: load partial if present, see which countries are already covered
all_data: dict[str, pd.Series] = {}
done_countries: set[str] = set()
if OUT_PARTIAL.exists():
    print(f"Found existing partial: {OUT_PARTIAL}")
    existing = pd.read_csv(OUT_PARTIAL, index_col="timestamp", parse_dates=True)
    existing.index = pd.to_datetime(existing.index, utc=True)
    for col in existing.columns:
        all_data[col] = existing[col]
        done_countries.add(col.split("_", 1)[0])
    print(f"  Already covered countries: {sorted(done_countries)}\n")


def save_partial():
    if not all_data:
        return
    df = pd.DataFrame(all_data).sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = "timestamp"
    df.to_csv(OUT_PARTIAL)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
errors: list[tuple[str, str]] = []
for label in COUNTRIES:
    if label in done_countries:
        print(f"--- {label}: SKIP (already in partial)")
        continue

    cc = COUNTRY_MAP.get(label)
    if cc is None:
        errors.append((label, "no zone mapping")); continue

    print(f"--- {label} ({cc}) ---")
    months: list[pd.DataFrame] = []
    consecutive_failures = 0
    bailed = False
    for s, e in month_range(START_TS, END_TS):
        df, err = query_with_retry(client, cc, s, e)
        if df is not None and not df.empty:
            months.append(df)
            print(f"  {s.date()}->{e.date()}: {len(df):>5} rows")
            consecutive_failures = 0
        else:
            print(f"  {s.date()}->{e.date()}: empty/error ({err or 'no data'})")
            consecutive_failures += 1
            if consecutive_failures >= EARLY_BAIL_AFTER:
                print(f"  -> {EARLY_BAIL_AFTER} consecutive failures, BAILING on {label}")
                bailed = True
                break
        time.sleep(0.5)

    if bailed or not months:
        errors.append((label, f"bailed after {consecutive_failures} consecutive failures"))
        # Save partial now so we don't lose other countries on the next interruption
        save_partial()
        continue

    df_c = pd.concat(months).sort_index()
    df_c = df_c[~df_c.index.duplicated(keep="first")]
    n_added = 0
    if isinstance(df_c.columns, pd.MultiIndex):
        for col in df_c.columns:
            if "Actual Aggregated" in str(col):
                tech = (col[0] if isinstance(col, tuple) else str(col)) \
                       .replace(" ", "_").replace("-", "_")
                all_data[f"{label}_{tech}"] = df_c[col].resample("h").mean()
                n_added += 1
    else:
        for col in df_c.columns:
            tech = str(col).replace(" ", "_").replace("-", "_")
            all_data[f"{label}_{tech}"] = df_c[col].resample("h").mean()
            n_added += 1

    print(f"  -> added {n_added} carriers for {label}")
    save_partial()                                         # save after each country
    print(f"  partial saved: {len(all_data)} cols total")

# ---------------------------------------------------------------------------
# Final partial save + merge
# ---------------------------------------------------------------------------
save_partial()
print(f"\n=== Final partial: {OUT_PARTIAL} ({len(all_data)} cols) ===")

if not all_data:
    print("No data downloaded.")
    sys.exit(1)

print(f"\nMerging with {PAPER1_GEN} ...")
df_partial = pd.DataFrame(all_data).sort_index()
df_partial.index = pd.to_datetime(df_partial.index, utc=True)
df_partial.index.name = "timestamp"

existing = pd.read_csv(PAPER1_GEN, index_col="timestamp", parse_dates=True)
existing.index = pd.to_datetime(existing.index, utc=True)

all_cols = sorted(set(df_partial.columns) | set(existing.columns))
df_partial = df_partial.reindex(columns=all_cols)
existing  = existing.reindex(columns=all_cols)
combined = pd.concat([df_partial, existing])
combined = combined[~combined.index.duplicated(keep="last")].sort_index()
combined.to_csv(OUT_MERGED)
print(f"\nMerged 2015-2025: {OUT_MERGED}")
print(f"  rows: {len(combined)}  cols: {len(combined.columns)}")
print(f"  span: {combined.index.min()} -> {combined.index.max()}")

if errors:
    print("\nSkipped / no-data countries:")
    for c, err in errors:
        print(f"  {c}: {err}")
