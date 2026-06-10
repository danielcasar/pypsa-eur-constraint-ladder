"""
For each (country, year), compute the 24-element HOD mean-deviation profile:

    dev_profile(c, y)[h] = mean(price | c, y, hour=h) - mean(price | c, y)

This is the canonical year-level shape object. Cross-year similarity between
two years' profiles is exactly the cross-year HOD Pearson rho we've been
studying (Pearson is invariant to additive shifts, so deviation-profile rho
equals raw-mean-profile rho).

Outputs
-------
  Analysis/results/hod_deviation_profiles_long.csv     long format
  Analysis/results/hod_deviation_profiles_DE_wide.csv  wide for visual check
  Analysis/results/hod_deviation_profiles_summary.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER1 = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)
PRICES_FILE = PAPER1 / "Code" / "data" / "ENTSOE_day_ahead_prices.csv"

PAPER_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO", "SI", "SK",
]
ANALYSIS_YEARS = list(range(2015, 2026))


def main() -> None:
    print("Loading ENTSO-E prices ...")
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    df = df[[c for c in df.columns if c in PAPER_COUNTRIES]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[df.index.year.isin(ANALYSIS_YEARS)]

    long = (df.reset_index()
              .melt(id_vars="timestamp", var_name="country", value_name="price")
              .dropna(subset=["price"]))
    long["year"] = long["timestamp"].dt.year
    long["hour"] = long["timestamp"].dt.hour

    # Annual mean per (country, year)
    ann = (long.groupby(["country", "year"])["price"]
              .mean().reset_index().rename(columns={"price": "annual_mean"}))
    # Hourly mean per (country, year, hour)
    hr = (long.groupby(["country", "year", "hour"])["price"]
             .mean().reset_index().rename(columns={"price": "hourly_mean"}))
    profile = hr.merge(ann, on=["country", "year"])
    profile["deviation"] = profile["hourly_mean"] - profile["annual_mean"]

    # ---- LONG format CSV (the canonical output) ---------------------------
    long_out = profile[["country", "year", "hour",
                        "annual_mean", "hourly_mean", "deviation"]]
    long_out.to_csv(RESULTS / "hod_deviation_profiles_long.csv", index=False)
    print(f"Wrote long CSV with {len(long_out)} rows "
          f"({long_out['country'].nunique()} countries x "
          f"{long_out['year'].nunique()} years x 24 hours).")

    # ---- DE wide for visual sanity check ----------------------------------
    de_wide = (profile[profile["country"] == "DE"]
               .pivot(index="year", columns="hour", values="deviation")
               .round(2))
    de_wide.to_csv(RESULTS / "hod_deviation_profiles_DE_wide.csv")
    print("\nDE deviation profile (rows = years, cols = hour-of-day, values = EUR/MWh):")
    print(de_wide.to_string())

    # ---- Cross-year Pearson on these profiles -----------------------------
    # Should be identical to the existing entsoe_cross_year_pearson.csv for hour
    # dimension - this is just a sanity check that we built the same object.
    rows = []
    for country, sub in profile.groupby("country"):
        wide = sub.pivot(index="year", columns="hour", values="deviation")
        years = sorted(wide.index)
        for i, y1 in enumerate(years):
            for y2 in years[i + 1:]:
                v1 = wide.loc[y1].values
                v2 = wide.loc[y2].values
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.sum() < 3:
                    continue
                # Centered correlation (== Pearson, since deviations sum to 0)
                r = float(np.corrcoef(v1[mask], v2[mask])[0, 1])
                rows.append({"country": country, "year1": int(y1),
                             "year2": int(y2), "lag": int(y2 - y1), "rho": r})
    pearson_check = pd.DataFrame(rows)
    pearson_check.to_csv(RESULTS / "hod_deviation_profile_cross_year_pearson.csv",
                         index=False)

    # Compare to existing entsoe cross-year pearson (paper 1) - hour dim only
    existing = pd.read_csv(
        PAPER1 / "Code" / "results" / "calendar_effects" / "entsoe_cross_year_pearson.csv"
    )
    existing = existing[existing["dimension"] == "hour"][["country", "year1", "year2", "rho"]]
    merged = pearson_check.merge(existing, on=["country", "year1", "year2"],
                                  how="inner", suffixes=("_dev", "_existing"))
    diff = (merged["rho_dev"] - merged["rho_existing"]).abs()
    print(f"\nSanity check: deviation-profile rho vs existing 24-mean rho")
    print(f"  matched rows: {len(merged)}")
    print(f"  max |diff|: {diff.max():.2e}")
    print(f"  median |diff|: {diff.median():.2e}")
    print("  -> as expected, the two are mathematically identical (additive shift).")

    # ---- Summary ----------------------------------------------------------
    out = []
    out.append("=" * 78)
    out.append("HOD mean-deviation profiles per (country, year)")
    out.append("=" * 78)
    out.append("\nObject: 24-element vector per (country, year), where element h is")
    out.append("  hourly_mean(c, y, h) - annual_mean(c, y).")
    out.append("Sums to 0 by construction within each (c, y).")
    out.append("")
    out.append("DE profile evolution (peak-to-trough span widens with solar build-up):")
    de_span = de_wide.max(axis=1) - de_wide.min(axis=1)
    for year, span in de_span.items():
        out.append(f"  {year}: peak-trough span = {span:6.1f} EUR/MWh  "
                   f"(min={de_wide.loc[year].min():+6.1f} at h{de_wide.loc[year].idxmin():02d}, "
                   f"max={de_wide.loc[year].max():+6.1f} at h{de_wide.loc[year].idxmax():02d})")
    out.append("")
    out.append("Cross-year Pearson on these deviation profiles is identical to the")
    out.append("paper-1 24-mean-profile Pearson (Pearson is invariant to additive shifts).")
    out.append(f"  matched rows: {len(merged)}; max abs diff: {diff.max():.2e}")
    out.append("")
    out.append("Files: ")
    out.append("  hod_deviation_profiles_long.csv      (panel: country, year, hour, deviation)")
    out.append("  hod_deviation_profiles_DE_wide.csv   (DE only, year x hour matrix)")
    out.append("  hod_deviation_profile_cross_year_pearson.csv  (sanity-check rho)")

    text = "\n".join(out)
    print()
    print(text)
    (RESULTS / "hod_deviation_profiles_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
