"""
Test alternative fixed-effects specifications to eliminate co-trending artefacts.

Variants (all on capnorm spec):
  M0  : country FE only             (current baseline)
  M1  : country FE + year FE        (two-way FE; standard fix for co-trend)
  M2  : country FE + country-trend  (country-specific linear time trend)
  M3  : country FE + year FE + country-trend (kitchen sink)

For each: report LOO R^2 + offwind morning coefficient (h08).
We use the h08 coefficient because that's where the current model has its
most suspicious positive coefficient on offwind.

Outputs:
  Analysis/results/two_way_fe_summary.txt
  Analysis/results/plots/two_way_fe_offwind_coef.{pdf,png}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"

DEV_FILE = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_capnorm_panel.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPNORM = [f"capnorm_{t}" for t in DIFF_TECHS]
EXPOSED = [f"exposed_capnorm_{t}" for t in DIFF_TECHS]
PREDICTORS = CAPNORM + EXPOSED + ["ntc_norm"]

state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
panel = (dev.merge(state, on=["country","year"], how="inner")
            .query("country in @COUNTRIES")
            .dropna(subset=["deviation"] + PREDICTORS))
panel["year_centered"] = panel["year"] - panel["year"].mean()


# ---------------------------------------------------------------------------
# Design matrix builders for each FE spec
# ---------------------------------------------------------------------------
def build_X(df, predictors, *, country_fe=True, year_fe=False, country_trend=False):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    countries = sorted(df["country"].unique())
    years = sorted(df["year"].unique())
    if country_fe:
        for c in countries[1:]:
            X[f"FE_c_{c}"] = (df["country"].values == c).astype(float)
    if year_fe:
        for y in years[1:]:
            X[f"FE_y_{y}"] = (df["year"].values == int(y)).astype(float)
    if country_trend:
        for c in countries[1:]:
            X[f"trend_{c}"] = ((df["country"].values == c).astype(float)
                                * df["year_centered"].values)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X, countries


def fit_per_hour(train_df, predictors, **fe_kwargs):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=["deviation"] + predictors)
        if len(sub) < 50:
            continue
        X, _ = build_X(sub, predictors, **fe_kwargs)
        y = sub["deviation"].astype(float)
        if X.shape[1] >= len(sub):
            continue   # too many params, skip
        res = sm.OLS(y, X).fit()
        out[h] = {
            "params": res.params, "x_cols": list(X.columns),
            "tvalues": res.tvalues, "rsq": float(res.rsquared),
            "countries": countries, "n": int(res.nobs),
        }
    return out


def predict_for_row(model_h, country, year, row, predictors):
    cols = model_h["x_cols"]; x = {c: 0.0 for c in cols}; x["const"] = 1.0
    if f"FE_c_{country}" in x: x[f"FE_c_{country}"] = 1.0
    if f"FE_y_{int(year)}" in x: x[f"FE_y_{int(year)}"] = 1.0
    if f"trend_{country}" in x:
        x[f"trend_{country}"] = float(row["year_centered"])
    for s in predictors:
        x[s] = float(row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


def loo(panel, predictors, **fe_kwargs):
    rows = []
    for y0 in sorted(panel["year"].unique()):
        tr = panel[panel["year"] != y0]
        te = panel[panel["year"] == y0]
        m_loo = fit_per_hour(tr, predictors, **fe_kwargs)
        for _, r in te.iterrows():
            h = int(r["hour"])
            if h not in m_loo or r["country"] not in m_loo[h]["countries"]:
                continue
            p = predict_for_row(m_loo[h], r["country"], r["year"], r, predictors)
            rows.append({"hour": h, "observed": float(r["deviation"]),
                         "predicted": p,
                         "residual": float(r["deviation"]) - p})
    return pd.DataFrame(rows)


def overall(df_loo):
    if df_loo.empty:
        return {"r2": np.nan, "rmse": np.nan, "n": 0}
    ss_res = (df_loo["residual"] ** 2).sum()
    ss_tot = ((df_loo["observed"] - df_loo["observed"].mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot, "rmse": float(np.sqrt((df_loo["residual"]**2).mean())),
            "n": int(len(df_loo))}


SPECS = {
    "M0_country_FE_only":       dict(country_fe=True, year_fe=False, country_trend=False),
    "M1_country_year_FE":       dict(country_fe=True, year_fe=True,  country_trend=False),
    "M2_country_FE_+_ctrend":   dict(country_fe=True, year_fe=False, country_trend=True),
    "M3_country_year_FE_+_ctrend": dict(country_fe=True, year_fe=True,  country_trend=True),
}

print(f"Panel: {len(panel)} rows, {panel['country'].nunique()} countries, "
      f"years {panel['year'].min()}-{panel['year'].max()}\n")

results = {}
for name, kw in SPECS.items():
    print(f"=== {name} ===")
    m_full = fit_per_hour(panel, PREDICTORS, **kw)
    print(f"  fit hours: {len(m_full)}/24")
    loo_df = loo(panel, PREDICTORS, **kw)
    o = overall(loo_df)
    print(f"  LOO R2 = {o['r2']:.3f}, RMSE = {o['rmse']:.2f}")
    # Offwind coefficient per hour
    rows = []
    for h, m in m_full.items():
        if "capnorm_offwind" not in m["x_cols"]:
            continue
        idx = m["x_cols"].index("capnorm_offwind")
        rows.append({"hour": h, "coef": float(m["params"].values[idx]),
                     "t": float(m["tvalues"].values[idx])})
    off_df = pd.DataFrame(rows).sort_values("hour")
    # Solar coefficient per hour
    sol_rows = []
    for h, m in m_full.items():
        if "capnorm_solar" not in m["x_cols"]:
            continue
        idx = m["x_cols"].index("capnorm_solar")
        sol_rows.append({"hour": h, "coef": float(m["params"].values[idx]),
                         "t": float(m["tvalues"].values[idx])})
    sol_df = pd.DataFrame(sol_rows).sort_values("hour")
    results[name] = {
        "loo": o, "offwind": off_df, "solar": sol_df,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Two-way FE / country-trend comparison on capnorm spec")
out.append("=" * 78)
out.append(f"\nPanel: {len(panel)} rows, {panel['country'].nunique()} countries\n")

out.append(f"{'spec':<32}  {'LOO R2':>8}  {'LOO RMSE':>10}")
out.append("-" * 78)
for name, r in results.items():
    out.append(f"{name:<32}  {r['loo']['r2']:>8.3f}  {r['loo']['rmse']:>10.2f}")

# Offwind h08 coefficient comparison
out.append("\nOffwind coefficient (capnorm_offwind) per hour, across specs:")
out.append(f"{'h':>3}  " + "  ".join(f"{n:>14}" for n in results.keys()))
out.append("-" * 78)
for h in range(24):
    cells = []
    for name in results:
        sub = results[name]["offwind"].query(f"hour == {h}")
        if sub.empty:
            cells.append(f"{'NA':>14}")
        else:
            c = sub["coef"].iloc[0]; t = sub["t"].iloc[0]
            cells.append(f"{c:>+7.1f}(t={t:>+4.1f})")
    out.append(f"{h:>3}  " + "  ".join(cells))

# Solar h10 coefficient comparison
out.append("\nSolar coefficient (capnorm_solar) per hour, across specs:")
out.append(f"{'h':>3}  " + "  ".join(f"{n:>14}" for n in results.keys()))
out.append("-" * 78)
for h in range(24):
    cells = []
    for name in results:
        sub = results[name]["solar"].query(f"hour == {h}")
        if sub.empty:
            cells.append(f"{'NA':>14}")
        else:
            c = sub["coef"].iloc[0]; t = sub["t"].iloc[0]
            cells.append(f"{c:>+7.1f}(t={t:>+4.1f})")
    out.append(f"{h:>3}  " + "  ".join(cells))

text = "\n".join(out)
print(text)
(RESULTS / "two_way_fe_summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plot: offwind h08 coefficient bars across specs + solar h10 across specs
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
specs_ordered = list(results.keys())
colors = ["#999999", "#1f77b4", "#2ca02c", "#d62728"]
for ax, hour, label, tech in [(a1, 8, "offwind h08 (was spurious +147)", "offwind"),
                               (a2, 10, "solar h10 (duck trough)", "solar")]:
    vals = [results[s][tech].query(f"hour == {hour}")["coef"].iloc[0]
            if not results[s][tech].query(f"hour == {hour}").empty else np.nan
            for s in specs_ordered]
    ts   = [results[s][tech].query(f"hour == {hour}")["t"].iloc[0]
            if not results[s][tech].query(f"hour == {hour}").empty else np.nan
            for s in specs_ordered]
    ax.bar(range(len(specs_ordered)), vals, color=colors[:len(specs_ordered)])
    ax.axhline(0, color="black", linewidth=0.5)
    for i, t in enumerate(ts):
        if not np.isnan(t):
            ax.text(i, vals[i] + (1 if vals[i] >= 0 else -1) * 2, f"t={t:+.1f}",
                    ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=7)
    ax.set_xticks(range(len(specs_ordered)))
    ax.set_xticklabels([s.replace("_", "\n", 2) for s in specs_ordered],
                       rotation=0, fontsize=7)
    ax.set_ylabel("Coefficient (EUR/MWh per +1.0)")
    ax.set_title(label, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(PLOTS / "two_way_fe_offwind_coef.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "two_way_fe_offwind_coef.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote: {RESULTS / 'two_way_fe_summary.txt'}")
print(f"       {PLOTS / 'two_way_fe_offwind_coef.pdf'}")
