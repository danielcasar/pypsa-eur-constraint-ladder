"""
Per-hour test of the cross-year Pearson rho linear-decay law.

For each country and each hour-of-day h, build a (year x day-of-year) matrix
of prices at hour h, compute pairwise cross-year Pearson rho between each
year-pair, then fit the same lag -> rho linear regression PER HOUR. If the
linear-decay law is solar-driven, midday hours should show the steepest
decay and night hours the weakest.

Reads (READ-ONLY from first paper):
  Code/data/ENTSOE_day_ahead_prices.csv

Writes:
  Analysis/results/linearity_per_hour_pairs.csv     (every c, h, y1, y2 row)
  Analysis/results/linearity_per_hour_per_country.csv (per c, h fit)
  Analysis/results/linearity_per_hour_per_hour.csv    (per h, median R^2 / slope)
  Analysis/results/linearity_per_hour_summary.txt
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
ANALYSIS_YEARS = list(range(2018, 2026))     # 2018..2025
MIN_DOY = 350     # require at least this many days-of-year non-NaN per year-hour


# ---------------------------------------------------------------------------
# Build per-(country, hour, year) day-of-year vectors
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    df = df[[c for c in df.columns if c in PAPER_COUNTRIES]]
    df.index = pd.to_datetime(df.index, utc=True)
    df["hour"] = df.index.hour
    df["doy"]  = df.index.dayofyear
    df["year"] = df.index.year
    return df.reset_index(drop=False).rename(columns={"index": "timestamp"})


def cross_year_pearson_per_hour(prices: pd.DataFrame) -> pd.DataFrame:
    """For each (country, hour), build year x day-of-year matrix and compute
    pairwise cross-year Pearson rho between each year-pair."""
    rows = []
    for country in PAPER_COUNTRIES:
        if country not in prices.columns:
            continue
        sub = prices[["year", "hour", "doy", country]].rename(columns={country: "p"})
        sub = sub[sub["year"].isin(ANALYSIS_YEARS)].copy()
        for h in range(24):
            chunk = sub[sub["hour"] == h]
            # year x doy matrix
            mat = (chunk
                   .groupby(["year", "doy"])["p"].mean()
                   .unstack(level="doy"))
            # require minimum coverage
            mat = mat.dropna(axis=1, thresh=2)              # drop doy with <2 years coverage
            mat = mat.loc[(mat.notna().sum(axis=1) >= MIN_DOY)]   # drop years short on doy
            if len(mat) < 2:
                continue
            years = mat.index.tolist()
            for i, y1 in enumerate(years):
                for y2 in years[i + 1:]:
                    v1 = mat.loc[y1].values
                    v2 = mat.loc[y2].values
                    mask = ~(np.isnan(v1) | np.isnan(v2))
                    if mask.sum() < MIN_DOY:
                        continue
                    r, p = stats.pearsonr(v1[mask], v2[mask])
                    rows.append({
                        "country": country, "hour": h,
                        "year1": int(y1), "year2": int(y2),
                        "lag": int(y2 - y1),
                        "rho": float(r), "p_value": float(p),
                        "n_doy": int(mask.sum()),
                    })
        print(f"  {country}: done")
    return pd.DataFrame(rows)


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.var(x) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "slope_p": np.nan, "n": int(len(x))}
    res = stats.linregress(x, y)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "r2": float(res.rvalue ** 2), "slope_p": float(res.pvalue),
            "n": int(len(x))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading ENTSO-E prices...")
    prices = load_prices()

    print("\nComputing per-hour cross-year Pearson rho ...")
    pairs = cross_year_pearson_per_hour(prices)
    pairs.to_csv(RESULTS / "linearity_per_hour_pairs.csv", index=False)
    print(f"\n  pairs panel: {len(pairs)} (country, hour, y1, y2) rows")

    # Per-(country, hour) line fit on full pairs (no lag-averaging)
    per_ch = []
    for (country, hour), sub in pairs.groupby(["country", "hour"]):
        f = fit_line(sub["lag"].values, sub["rho"].values)
        per_ch.append({
            "country": country, "hour": int(hour),
            "n_obs": f["n"],
            "slope": f["slope"], "intercept": f["intercept"],
            "r2": f["r2"], "slope_p": f["slope_p"],
        })
    per_ch_df = pd.DataFrame(per_ch)
    per_ch_df.to_csv(RESULTS / "linearity_per_hour_per_country.csv", index=False)

    # Per-hour aggregation across countries
    per_h = []
    for hour, sub in per_ch_df.groupby("hour"):
        per_h.append({
            "hour": int(hour),
            "n_countries": int(sub["country"].nunique()),
            "r2_median": float(sub["r2"].median()),
            "r2_q25": float(sub["r2"].quantile(0.25)),
            "r2_q75": float(sub["r2"].quantile(0.75)),
            "slope_median": float(sub["slope"].median()),
            "slope_q25": float(sub["slope"].quantile(0.25)),
            "slope_q75": float(sub["slope"].quantile(0.75)),
            "share_negative_slope": float((sub["slope"] < 0).mean()),
            "share_signif": float((sub["slope_p"] < 0.05).mean()),
        })
    per_h_df = pd.DataFrame(per_h).sort_values("hour")
    per_h_df.to_csv(RESULTS / "linearity_per_hour_per_hour.csv", index=False)

    # ---- Human summary ----------------------------------------------------
    out = []
    out.append("=" * 78)
    out.append("Linear cross-year rho decay — per hour-of-day, full-data per country")
    out.append("=" * 78)
    out.append(f"\nOne fit per (country, hour) on all year-pair observations.")
    out.append(f"Slope = decay rate of cross-year rho per year of separation, at that hour.")
    out.append(f"Median across {per_ch_df['country'].nunique()} countries shown.\n")
    out.append(f"{'h':>3}  {'R^2 med':>9} {'IQR':>16}  {'slope med':>10}  "
               f"{'IQR':>22}  {'%neg':>5} {'%sig':>5}")
    out.append("-" * 78)
    for _, r in per_h_df.iterrows():
        r2_iqr = f"[{r['r2_q25']:.2f}, {r['r2_q75']:.2f}]"
        sl_iqr = f"[{r['slope_q25']:+.4f}, {r['slope_q75']:+.4f}]"
        out.append(
            f"{int(r['hour']):>3}  {r['r2_median']:>9.3f} {r2_iqr:>16}  "
            f"{r['slope_median']:>+10.4f}  {sl_iqr:>22}  "
            f"{r['share_negative_slope']*100:>4.0f}% "
            f"{r['share_signif']*100:>4.0f}%"
        )

    # Identify steepest-decay vs flattest-decay hours
    steep = per_h_df.sort_values("slope_median").head(3)
    flat  = per_h_df.sort_values("slope_median").tail(3)
    out.append("\n3 STEEPEST-decay hours (largest negative slope):")
    for _, r in steep.iterrows():
        out.append(f"  hour {int(r['hour']):>2}: slope = {r['slope_median']:+.4f}/yr, "
                   f"R^2 median = {r['r2_median']:.3f}")
    out.append("3 FLATTEST hours (smallest |slope|):")
    for _, r in flat.iterrows():
        out.append(f"  hour {int(r['hour']):>2}: slope = {r['slope_median']:+.4f}/yr, "
                   f"R^2 median = {r['r2_median']:.3f}")

    text = "\n".join(out)
    print(text)
    (RESULTS / "linearity_per_hour_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
