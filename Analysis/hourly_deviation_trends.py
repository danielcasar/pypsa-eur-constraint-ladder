"""
Per-hour linear trend in HOD deviation from the annual mean.

For each (country, year, hour-of-day):
  dev(c, y, h) = mean(price | c, y, h) - mean(price | c, y)

For each (country, hour) we fit dev ~ year. Slope_{c,h} = how much hour h's
contribution to the daily shape shifts per year (EUR/MWh per year), at that
country.

Aggregated to per-hour: median slope across the 22 countries. Expectation
under the duck-curve hypothesis: midday slopes negative (belly deepens),
evening slopes positive (peak rises), night slopes ~ 0.

Reads (READ-ONLY from first paper):
  Code/data/ENTSOE_day_ahead_prices.csv

Writes:
  Analysis/results/hourly_deviation_trends_per_country.csv
  Analysis/results/hourly_deviation_trends_per_hour.csv
  Analysis/results/hourly_deviation_trends_summary.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
ANALYSIS_YEARS = list(range(2018, 2026))


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    df = df[[c for c in df.columns if c in PAPER_COUNTRIES]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[df.index.year.isin(ANALYSIS_YEARS)]
    long = df.reset_index().melt(
        id_vars="timestamp", var_name="country", value_name="price"
    )
    long["year"] = long["timestamp"].dt.year
    long["hour"] = long["timestamp"].dt.hour
    return long.dropna(subset=["price"])


def main() -> None:
    print("Loading ENTSO-E prices...")
    prices = load_prices()

    # Per (country, year): annual mean and per-hour mean -> deviation
    annual_mean = (prices.groupby(["country", "year"])["price"]
                   .mean().reset_index().rename(columns={"price": "annual_mean"}))
    hourly_mean = (prices.groupby(["country", "year", "hour"])["price"]
                   .mean().reset_index().rename(columns={"price": "hourly_mean"}))
    panel = hourly_mean.merge(annual_mean, on=["country", "year"])
    panel["deviation"] = panel["hourly_mean"] - panel["annual_mean"]

    # Per-(country, hour) linear fit of deviation ~ year
    rows = []
    for (country, hour), sub in panel.groupby(["country", "hour"]):
        if len(sub) < 3:
            continue
        res = stats.linregress(sub["year"].values, sub["deviation"].values)
        rows.append({
            "country": country,
            "hour": int(hour),
            "n_years": int(len(sub)),
            "slope_eur_per_year": float(res.slope),
            "intercept": float(res.intercept),
            "r2": float(res.rvalue ** 2),
            "slope_p": float(res.pvalue),
        })
    per_ch = pd.DataFrame(rows)
    per_ch.to_csv(RESULTS / "hourly_deviation_trends_per_country.csv", index=False)

    # Per-hour aggregation across countries
    per_h = (per_ch.groupby("hour")
             .agg(n_countries=("country", "count"),
                  slope_median=("slope_eur_per_year", "median"),
                  slope_q25=("slope_eur_per_year", lambda s: s.quantile(0.25)),
                  slope_q75=("slope_eur_per_year", lambda s: s.quantile(0.75)),
                  r2_median=("r2", "median"),
                  share_negative_slope=("slope_eur_per_year", lambda s: (s < 0).mean()),
                  share_signif=("slope_p", lambda s: (s < 0.05).mean()))
             .reset_index().sort_values("hour"))
    per_h.to_csv(RESULTS / "hourly_deviation_trends_per_hour.csv", index=False)

    # ---- Human summary -----------------------------------------------------
    out = []
    out.append("=" * 78)
    out.append("Per-hour linear trend in HOD deviation from annual mean")
    out.append("=" * 78)
    out.append("\nSlope = how much hour h's deviation from the annual mean shifts")
    out.append("per year, in EUR/MWh. Negative -> belly deepening (price falls vs daily mean).")
    out.append("Positive -> peak rising (price rises vs daily mean). Median across 22 countries.\n")
    out.append(f"{'h':>3}  {'slope med':>12}  {'IQR':>22}  {'%neg':>5} "
               f"{'%sig':>5}  {'R^2 med':>9}")
    out.append("-" * 78)
    for _, r in per_h.iterrows():
        sl_iqr = f"[{r['slope_q25']:+.3f}, {r['slope_q75']:+.3f}]"
        out.append(
            f"{int(r['hour']):>3}  {r['slope_median']:>+12.3f}  {sl_iqr:>22}  "
            f"{r['share_negative_slope']*100:>4.0f}% "
            f"{r['share_signif']*100:>4.0f}%  "
            f"{r['r2_median']:>9.3f}"
        )

    out.append("\n3 STEEPEST belly-deepening hours (most negative slope):")
    for _, r in per_h.sort_values("slope_median").head(3).iterrows():
        out.append(f"  hour {int(r['hour']):>2}: {r['slope_median']:+.3f} EUR/MWh/yr")
    out.append("3 STEEPEST peak-rising hours (most positive slope):")
    for _, r in per_h.sort_values("slope_median").tail(3).iloc[::-1].iterrows():
        out.append(f"  hour {int(r['hour']):>2}: {r['slope_median']:+.3f} EUR/MWh/yr")

    text = "\n".join(out)
    print(text)
    (RESULTS / "hourly_deviation_trends_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
