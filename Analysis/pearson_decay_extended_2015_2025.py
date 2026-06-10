"""
Extend the cross-year Pearson rho-decay analysis from Paper 1 backward to 2015.

Paper 1's ENTSOE_YEARS = [2018, ..., 2025] (8 years -> 28 year-pairs).
ENTSOE_day_ahead_prices.csv actually spans 2015-01-01 -> 2026-01-01, so we
recompute everything on years [2015, ..., 2025] (11 years -> 55 year-pairs)
and compare:
  - median-line slope, intercept, R^2 per dimension
  - full-data (every country x year-pair) slope, R^2 per dimension

We reuse the paper's helpers via sys.path import (no copy-paste port).

Outputs:
  Analysis/results/entsoe_cross_year_pearson_2015_2025.csv
  Analysis/results/entsoe_pearson_decay_linearity_2015_2025_summary.csv
  Analysis/results/entsoe_pearson_decay_linearity_2015_2025_summary.txt
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER1_SCRIPTS = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/paper/scripts"
)
sys.path.insert(0, str(PAPER1_SCRIPTS))
from compute_entsoe_cross_year_pearson import _add_time_dims, TIME_DIMENSIONS  # noqa: E402
from config import COUNTRIES  # noqa: E402

ENTSOE_FILE = (PAPER1_SCRIPTS.parent.parent / "data" / "ENTSOE_day_ahead_prices.csv")
YEARS_EXTENDED = list(range(2015, 2026))
YEARS_PAPER1   = list(range(2018, 2026))

DIMS = TIME_DIMENSIONS
DIM_LABEL = {"hour": "Hour-of-Day", "weekday": "Day-of-Week",
             "month": "Month-of-Year", "season": "Season"}


def load_profiles_for_country(df_raw: pd.DataFrame, country: str,
                              years: list[int]):
    if country not in df_raw.columns:
        return {}
    s = df_raw[[country]].rename(columns={country: "price"}).dropna()
    s = _add_time_dims(s)
    profiles = {}
    for y in years:
        sub = s[s.index.year == y]
        if len(sub) < 100:
            continue
        yp = {}
        for dim in DIMS:
            gm = sub.groupby(dim)["price"].mean()
            if len(gm) >= 2:
                yp[dim] = gm
        if yp:
            profiles[y] = yp
    return profiles


def compute_cross_year_pearson(years: list[int]) -> pd.DataFrame:
    df_raw = pd.read_csv(ENTSOE_FILE, index_col=0, parse_dates=True)
    pairs = list(itertools.combinations(years, 2))
    rows = []
    for country in COUNTRIES:
        prof = load_profiles_for_country(df_raw, country, years)
        if not prof:
            continue
        for y1, y2 in pairs:
            if y1 not in prof or y2 not in prof:
                continue
            for dim in DIMS:
                if dim not in prof[y1] or dim not in prof[y2]:
                    continue
                p1, p2 = prof[y1][dim], prof[y2][dim]
                common = p1.index.intersection(p2.index)
                if len(common) < 3:
                    continue
                rho, pval = pearsonr(p1.loc[common].values, p2.loc[common].values)
                rows.append({
                    "country": country, "dimension": dim,
                    "year1": y1, "year2": y2,
                    "rho": float(rho), "p_value": float(pval),
                    "n_groups": int(len(common)),
                })
    return pd.DataFrame(rows)


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 2 or np.var(x) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "slope_p": np.nan, "n": int(len(x))}
    r = stats.linregress(x, y)
    return {"slope": float(r.slope), "intercept": float(r.intercept),
            "r2": float(r.rvalue ** 2), "slope_p": float(r.pvalue),
            "n": int(len(x))}


def linearity_summary(df_pearson: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df_pearson.copy()
    df["lag"] = df["year2"] - df["year1"]
    df = df[df["lag"] > 0]
    rows = []
    for dim in DIMS:
        sub = df[df["dimension"] == dim]
        med = sub.groupby("lag")["rho"].median().sort_index()
        fm = fit_line(med.index.values.astype(float), med.values)
        fa = fit_line(sub["lag"].values.astype(float), sub["rho"].values)
        rows.append({
            "label": label, "dimension": dim,
            "n_lags": int(len(med)), "max_lag": int(med.index.max()),
            "med_slope": fm["slope"], "med_intercept": fm["intercept"],
            "med_r2": fm["r2"], "med_slope_p": fm["slope_p"],
            "full_n": fa["n"], "full_slope": fa["slope"],
            "full_intercept": fa["intercept"], "full_r2": fa["r2"],
            "full_slope_p": fa["slope_p"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Extended panel: years {YEARS_EXTENDED[0]}-{YEARS_EXTENDED[-1]} "
      f"({len(YEARS_EXTENDED)} years, "
      f"{len(list(itertools.combinations(YEARS_EXTENDED, 2)))} pairs)")
print(f"Paper-1 panel : years {YEARS_PAPER1[0]}-{YEARS_PAPER1[-1]} "
      f"({len(YEARS_PAPER1)} years, "
      f"{len(list(itertools.combinations(YEARS_PAPER1, 2)))} pairs)")
print(f"Countries     : {len(COUNTRIES)}\n")

print("Computing cross-year Pearson on extended panel ...")
df_ext = compute_cross_year_pearson(YEARS_EXTENDED)
df_ext.to_csv(RESULTS / "entsoe_cross_year_pearson_2015_2025.csv", index=False)
print(f"  -> {len(df_ext)} (country, dim, pair) rows")

# Old fit (just filter the extended one to 2018-2025 — same code path)
df_p1 = df_ext[(df_ext["year1"] >= 2018) & (df_ext["year2"] >= 2018)]

sum_ext = linearity_summary(df_ext, "2015-2025 (extended)")
sum_p1  = linearity_summary(df_p1, "2018-2025 (paper 1)")
both = pd.concat([sum_p1, sum_ext], ignore_index=True)
both.to_csv(RESULTS / "entsoe_pearson_decay_linearity_2015_2025_summary.csv",
            index=False)


# ---------------------------------------------------------------------------
# Human-readable side-by-side
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Linearity of cross-year Pearson rho decay")
out.append("Comparison: paper 1 panel (2018-2025) vs extended panel (2015-2025)")
out.append("=" * 78)

out.append("\nMEDIAN-LINE fit  (one line per dimension through median rho per lag):")
out.append(f"{'panel':<22}  {'dim':<10}  {'n_lags':>6}  "
           f"{'slope':>10}  {'intercept':>10}  {'R^2':>6}  {'slope_p':>10}")
out.append("-" * 78)
for _, r in both.iterrows():
    out.append(
        f"{r['label']:<22}  {r['dimension']:<10}  {r['n_lags']:>6}  "
        f"{r['med_slope']:>+10.4f}  {r['med_intercept']:>+10.4f}  "
        f"{r['med_r2']:>6.3f}  {r['med_slope_p']:>10.3g}"
    )

out.append("\nFULL-DATA fit  (every individual country x year-pair observation):")
out.append(f"{'panel':<22}  {'dim':<10}  {'n':>5}  "
           f"{'slope':>10}  {'intercept':>10}  {'R^2':>6}  {'slope_p':>10}")
out.append("-" * 78)
for _, r in both.iterrows():
    out.append(
        f"{r['label']:<22}  {r['dimension']:<10}  {int(r['full_n']):>5}  "
        f"{r['full_slope']:>+10.4f}  {r['full_intercept']:>+10.4f}  "
        f"{r['full_r2']:>6.3f}  {r['full_slope_p']:>10.3g}"
    )

out.append("\nDelta (extended minus paper-1) for each dimension:")
out.append(f"{'dim':<10}  {'d_med_slope':>12}  {'d_med_r2':>10}  "
           f"{'d_full_slope':>13}  {'d_full_r2':>10}")
out.append("-" * 78)
for dim in DIMS:
    a = both[(both["label"] == "2018-2025 (paper 1)") & (both["dimension"] == dim)].iloc[0]
    b = both[(both["label"] == "2015-2025 (extended)") & (both["dimension"] == dim)].iloc[0]
    out.append(
        f"{dim:<10}  {b['med_slope']-a['med_slope']:>+12.4f}  "
        f"{b['med_r2']-a['med_r2']:>+10.3f}  "
        f"{b['full_slope']-a['full_slope']:>+13.4f}  "
        f"{b['full_r2']-a['full_r2']:>+10.3f}"
    )

text = "\n".join(out)
print(text)
(RESULTS / "entsoe_pearson_decay_linearity_2015_2025_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'entsoe_pearson_decay_linearity_2015_2025_summary.txt'}")
