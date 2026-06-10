"""
Build the unified fuel_prices.csv from the user-provided source files in
Analysis/data/:

  - natural-gas-prices.csv       (OWID, USD/MWh, TTF, annual 2018-2024)
  - coal-prices.csv              (OWID, USD/tonne, Australia, annual 2018-2024)
  - emission-spot-primary-market-auction-report-{YYYY}-data.{xls,xlsx}
                                 (EEX primary EUA auction prices, daily, 2018-2025)

Output:
  Analysis/data/fuel_prices.csv with schema
      year, ttf_eur_per_mwh, eua_eur_per_t, api2_eur_per_t

Notes:
  - 'api2_eur_per_t' here uses the OWID coal series, which is the Australian
    thermal benchmark. API2 (Northwest European ARA) tracks Australian thermal
    closely (correlation ~0.9 over 2010s-2020s); the absolute level differs by
    a few EUR/t. For the cross-year-shift regression where the response is a
    correlation, the slight level offset is irrelevant — only the *trajectory*
    matters and they are highly co-trending.
  - USD->EUR uses ECB annual average reference rates, hard-coded inline.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ECB reference rate, USD per 1 EUR, annual averages
# Source: ECB Statistical Data Warehouse, EXR.D.USD.EUR.SP00.A annualised
# ---------------------------------------------------------------------------
USD_PER_EUR = {
    2018: 1.1810,
    2019: 1.1195,
    2020: 1.1422,
    2021: 1.1827,
    2022: 1.0537,
    2023: 1.0813,
    2024: 1.0824,
    2025: 1.0852,
}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "fuel_prices.csv"

# ---------------------------------------------------------------------------
# 1. Gas — TTF, USD/MWh -> EUR/MWh
# ---------------------------------------------------------------------------
gas = pd.read_csv(DATA / "natural-gas-prices.csv")
gas = gas[gas["Entity"] == "Netherlands TTF"].copy()
gas["Year"] = gas["Year"].astype(int)
gas["ttf_eur_per_mwh"] = gas.apply(
    lambda r: r["Gas price"] / USD_PER_EUR.get(int(r["Year"]), float("nan")),
    axis=1,
)
gas = gas[["Year", "ttf_eur_per_mwh"]].rename(columns={"Year": "year"})

# ---------------------------------------------------------------------------
# 2. Coal — Australian thermal, USD/t -> EUR/t (proxy for API2)
# ---------------------------------------------------------------------------
coal = pd.read_csv(DATA / "coal-prices.csv")
coal = coal[coal["Entity"] == "Australia"].copy()
coal["Year"] = coal["Year"].astype(int)
coal["api2_eur_per_t"] = coal.apply(
    lambda r: r["Coal"] / USD_PER_EUR.get(int(r["Year"]), float("nan")),
    axis=1,
)
coal = coal[["Year", "api2_eur_per_t"]].rename(columns={"Year": "year"})

# ---------------------------------------------------------------------------
# 3. EUA — EEX primary auction reports (daily) -> annual mean
# ---------------------------------------------------------------------------
EUA_DIR = DATA / "emission-spot-primary-market-auction-report-2012-2025-data"
eua_records = []

# Possible price column names across EEX report variants
PRICE_COL_PATTERNS = [
    re.compile(r"auction\s*price", re.I),
    re.compile(r"settlement\s*price", re.I),
    re.compile(r"price.*€|price.*eur", re.I),
    re.compile(r"price$", re.I),
]


def find_price_column(df: pd.DataFrame) -> str | None:
    for pat in PRICE_COL_PATTERNS:
        for c in df.columns:
            if pat.search(str(c)):
                return c
    return None


def find_date_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if re.search(r"date|tag", str(c), re.I):
            return c
    return None


for fp in sorted(EUA_DIR.glob("*.xls*")):
    year = int(re.search(r"(\d{4})", fp.name).group(1))
    if year < 2018 or year > 2025:
        continue
    # Try multiple sheet positions / header rows; EEX changed format over years
    found = False
    for sheet in [0, 1, "Auction Results", "Spot Auction Report", "Primary Market"]:
        for header_row in [0, 1, 2, 3, 4, 5]:
            try:
                df = pd.read_excel(fp, sheet_name=sheet, header=header_row,
                                   engine=None)
            except Exception:
                continue
            pc = find_price_column(df)
            dc = find_date_column(df)
            if pc is None:
                continue
            prices = pd.to_numeric(df[pc], errors="coerce").dropna()
            # filter implausible values (header bleed, etc.)
            prices = prices[(prices > 0) & (prices < 500)]
            if len(prices) >= 30:        # at least ~30 trading-day-like observations
                eua_records.append({"year": year, "mean": float(prices.mean()),
                                    "n": int(len(prices)),
                                    "src": f"{fp.name}::sheet={sheet}::row={header_row}"})
                found = True
                break
        if found:
            break
    if not found:
        print(f"  WARNING: could not parse {fp.name}")

eua = pd.DataFrame(eua_records).rename(columns={"mean": "eua_eur_per_t"})
eua = eua.sort_values("year").drop_duplicates("year", keep="first")[
    ["year", "eua_eur_per_t"]
]

# ---------------------------------------------------------------------------
# Merge & write
# ---------------------------------------------------------------------------
fp = (gas.merge(eua, on="year", how="outer")
         .merge(coal, on="year", how="outer")
         .sort_values("year"))
fp = fp[(fp["year"] >= 2018) & (fp["year"] <= 2025)].reset_index(drop=True)
fp = fp[["year", "ttf_eur_per_mwh", "eua_eur_per_t", "api2_eur_per_t"]]
fp.to_csv(OUT, index=False, float_format="%.3f")

print("=" * 64)
print(f"Wrote {OUT}")
print("=" * 64)
print(fp.to_string(index=False))
print()
missing = fp[fp.isna().any(axis=1)]
if not missing.empty:
    print("Years with missing values (will be NaN-dropped in fuel-level specs):")
    print(missing.to_string(index=False))
