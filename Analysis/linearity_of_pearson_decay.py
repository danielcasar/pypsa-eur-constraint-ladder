"""
Quantify the linearity of the cross-year Pearson rho decay seen in
'Figure 8' (plot_pearson_stability_decay.py) of the first paper.

For each calendar dimension (hour, weekday, month, season) we fit
    median_rho(lag) = a * lag + b
to the same 7-point line shown in that figure (lags 1..7). We report:
  - R^2 of the fit (linearity strength)
  - slope a (rate of decay per year)
  - intercept b (extrapolated lag-0 rho)
  - slope p-value (against H0: a = 0)

We also fit the same line per country (country-level R^2 + slope),
so we can see whether the linearity is universal across countries or
driven by a few outliers.

Inputs (READ-ONLY from first paper):
  Code/results/calendar_effects/entsoe_cross_year_pearson.csv

Outputs:
  Analysis/results/entsoe_pearson_decay_linearity_summary.csv
  Analysis/results/entsoe_pearson_decay_linearity_per_country.csv
  Analysis/results/entsoe_pearson_decay_linearity_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER1 = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)
INPUT = PAPER1 / "Code" / "results" / "calendar_effects" / "entsoe_cross_year_pearson.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT)
df["lag"] = df["year2"] - df["year1"]
df = df[df["lag"] > 0].copy()                # drop within-year (lag=0) rows
df["dimension"] = df["dimension"].astype(str)

DIMS = ["hour", "weekday", "month", "season"]
DIM_LABEL = {"hour": "Hour-of-Day", "weekday": "Day-of-Week",
             "month": "Month-of-Year", "season": "Season"}


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS fit y = a*x + b. Returns slope, intercept, r2, slope p-value, n."""
    if len(x) < 2 or np.var(x) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "slope_p": np.nan, "n": int(len(x))}
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
        "slope_p": float(res.pvalue),
        "n": int(len(x)),
    }


# ---------------------------------------------------------------------------
# Summary line: median rho across countries per lag, one fit per dimension
# (this is exactly the line drawn in Figure 8)
# ---------------------------------------------------------------------------
summary_rows = []
for dim in DIMS:
    sub = df[df["dimension"] == dim]
    by_lag = (
        sub.groupby("lag")["rho"]
        .agg(["median", "mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "count"])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        .reset_index()
        .sort_values("lag")
    )
    fit_med = fit_line(by_lag["lag"].values, by_lag["median"].values)
    fit_mean = fit_line(by_lag["lag"].values, by_lag["mean"].values)

    # NEW: full-data fit — every individual (country, year-pair) observation,
    # not collapsed to medians. R^2 here = fraction of total rho variance
    # explained by lag alone. Much more demanding test than the median-line R^2.
    fit_all = fit_line(sub["lag"].values, sub["rho"].values)

    summary_rows.append({
        "dimension": dim,
        "label": DIM_LABEL[dim],
        "n_lags": int(len(by_lag)),
        "lags": list(by_lag["lag"].astype(int)),
        # median-line fit (matches Figure 8 line; n = 7)
        "median_slope":     round(fit_med["slope"], 5),
        "median_intercept": round(fit_med["intercept"], 4),
        "median_r2":        round(fit_med["r2"], 4),
        "median_slope_p":   fit_med["slope_p"],
        # mean-line fit (alternative central tendency; n = 7)
        "mean_slope":       round(fit_mean["slope"], 5),
        "mean_r2":          round(fit_mean["r2"], 4),
        # full-data fit (every country × year-pair; n in hundreds)
        "full_n":           fit_all["n"],
        "full_slope":       round(fit_all["slope"], 5),
        "full_intercept":   round(fit_all["intercept"], 4),
        "full_r2":          round(fit_all["r2"], 4),
        "full_slope_p":     fit_all["slope_p"],
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULTS / "entsoe_pearson_decay_linearity_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Per-country: fit one line per (country, dimension); then summarise R^2 / slope
# distribution across countries
# ---------------------------------------------------------------------------
per_country_rows = []
for (country, dim), sub in df.groupby(["country", "dimension"]):
    by_lag = (
        sub.groupby("lag")["rho"].mean().reset_index().sort_values("lag")
    )
    f = fit_line(by_lag["lag"].values, by_lag["rho"].values)
    per_country_rows.append({
        "country": country,
        "dimension": dim,
        "n_lags": f["n"],
        "slope": round(f["slope"], 5) if not np.isnan(f["slope"]) else np.nan,
        "intercept": round(f["intercept"], 4) if not np.isnan(f["intercept"]) else np.nan,
        "r2": round(f["r2"], 4) if not np.isnan(f["r2"]) else np.nan,
        "slope_p": f["slope_p"],
    })
per_country = pd.DataFrame(per_country_rows)
per_country.to_csv(RESULTS / "entsoe_pearson_decay_linearity_per_country.csv", index=False)

# Per-dimension distribution of country-level R^2 / slope
dist = (
    per_country.groupby("dimension")
    .agg(
        n_countries=("country", "count"),
        r2_median=("r2", "median"),
        r2_q25=("r2", lambda s: s.quantile(0.25)),
        r2_q75=("r2", lambda s: s.quantile(0.75)),
        slope_median=("slope", "median"),
        slope_q25=("slope", lambda s: s.quantile(0.25)),
        slope_q75=("slope", lambda s: s.quantile(0.75)),
        share_negative_slope=("slope", lambda s: (s < 0).mean()),
        share_signif_slope=("slope_p", lambda s: (s < 0.05).mean()),
    )
    .reset_index()
)
dist.to_csv(RESULTS / "entsoe_pearson_decay_linearity_per_country_dist.csv", index=False)

# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Linearity of cross-year Pearson rho decay (ENTSO-E, Paper 1 figure 8)")
out.append("=" * 78)
out.append("\nMETHOD: per dimension, fit OLS line through (lag, median_rho_across_countries).")
out.append("Reported R^2 measures how cleanly the median curve decays linearly with lag.")
out.append("")
out.append(f"{'Dimension':<14} {'n_lags':>6}  {'slope':>10}  {'intercept':>10}  {'R^2':>6}  {'slope_p':>10}")
out.append("-" * 78)
for _, r in summary.iterrows():
    out.append(
        f"{r['label']:<14} {r['n_lags']:>6}  "
        f"{r['median_slope']:>+10.4f}  "
        f"{r['median_intercept']:>+10.4f}  "
        f"{r['median_r2']:>6.3f}  "
        f"{r['median_slope_p']:>10.3g}"
    )

out.append("")
out.append("FULL-DATA fit: every individual (country, year-pair) observation, not")
out.append("collapsed to medians. R^2 = fraction of total rho variance explained by")
out.append("lag alone (much more demanding test than the 7-median-points R^2 above).")
out.append("")
out.append(f"{'Dimension':<14} {'n':>5}  {'slope':>10}  {'intercept':>10}  {'R^2':>6}  {'slope_p':>10}")
out.append("-" * 78)
for _, r in summary.iterrows():
    out.append(
        f"{r['label']:<14} {int(r['full_n']):>5}  "
        f"{r['full_slope']:>+10.4f}  "
        f"{r['full_intercept']:>+10.4f}  "
        f"{r['full_r2']:>6.3f}  "
        f"{r['full_slope_p']:>10.3g}"
    )

out.append("")
out.append("Interpretation key:")
out.append("  R^2 close to 1  -> median rho decays cleanly linearly with lag")
out.append("  slope < 0       -> profiles become less correlated as years grow apart")
out.append("  slope ~ 0       -> profile is stable across years (no temporal shift)")
out.append("  slope > 0       -> profiles become MORE correlated with lag (rare)")
out.append("")

out.append("\nPer-country distribution (does the linearity hold across all countries?):")
out.append(f"{'Dimension':<14} {'n':>4}  {'R^2 median':>10}  {'R^2 IQR':>16}  "
           f"{'slope med':>10}  {'% slope<0':>10}  {'% slope sig':>12}")
out.append("-" * 78)
for _, r in dist.iterrows():
    iqr = f"[{r['r2_q25']:.2f}, {r['r2_q75']:.2f}]"
    out.append(
        f"{DIM_LABEL[r['dimension']]:<14} {int(r['n_countries']):>4}  "
        f"{r['r2_median']:>10.3f}  {iqr:>16}  "
        f"{r['slope_median']:>+10.4f}  "
        f"{r['share_negative_slope']*100:>9.0f}%  "
        f"{r['share_signif_slope']*100:>11.0f}%"
    )

text = "\n".join(out)
print(text)
(RESULTS / "entsoe_pearson_decay_linearity_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'entsoe_pearson_decay_linearity_summary.txt'}")
