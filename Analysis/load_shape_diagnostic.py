"""
Diagnose whether load_share_h adds nothing because:
  (A) within-country variation across years is tiny (no signal to fit), OR
  (B) the signal is real but collinear with capshare / neighbour predictors.

Tests:
  1. Variance decomposition of load_share_h: between-country vs within-country.
  2. M_load_only: dev_h ~ alpha_c + load_share_h    (no caps, no neighbour)
     If within-country load variation predicts within-country deviation,
     LOO R^2 here will be substantially > 0.
  3. M_load_plus_caps: dev_h ~ alpha_c + caps + load_share_h
     Compare load coefficient and its t-stat to M_load_only — does it shrink
     when caps are present? That's the test for collinearity-driven absorption.
  4. Has DE load shape *actually* shifted? Plot DE load_share_h trajectory 2015->2025.

Outputs:
  Analysis/results/load_shape_diagnostic_summary.txt
  Analysis/results/plots/load_shape_variance_decomposition.{pdf,png}
  Analysis/results/plots/load_shape_de_trajectory.{pdf,png}
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
    "legend.fontsize": 8, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE   = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_panel_with_neighbors.csv"
LOAD_PANEL = RESULTS / "load_share_panel.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]
EXPOSED   = [f"exposed_{t}"  for t in DIFF_TECHS]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
load_long = pd.read_csv(LOAD_PANEL)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .merge(load_long[["country", "year", "hour", "load_share_h"]],
                   on=["country", "year", "hour"], how="left")
            .query("country in @COUNTRIES"))


# ---------------------------------------------------------------------------
# 1. Variance decomposition of load_share_h per hour
# ---------------------------------------------------------------------------
print("=" * 78)
print("1. Variance decomposition of load_share_h")
print("=" * 78)
print(f"Sample: {len(panel)} rows from {panel['country'].nunique()} countries x 11 years x 24 hours\n")

var_rows = []
for h in range(24):
    sub = panel[panel["hour"] == h].dropna(subset=["load_share_h"])
    if sub.empty:
        continue
    # Total variance
    total_var = sub["load_share_h"].var()
    # Between-country variance: variance of country means
    country_mean = sub.groupby("country")["load_share_h"].mean()
    between_var = country_mean.var()
    # Within-country variance: average of within-country variances
    within_var = sub.groupby("country")["load_share_h"].var().mean()
    var_rows.append({
        "hour": h,
        "total_var": total_var,
        "between_var": between_var,
        "within_var": within_var,
        "within_share": within_var / total_var if total_var > 0 else 0.0,
        "load_share_mean": sub["load_share_h"].mean(),
        "within_std": np.sqrt(within_var),
    })
var_df = pd.DataFrame(var_rows)

print(f"{'h':>3}  {'mean_share':>10}  {'within_std':>10}  {'within%':>8}  "
      f"{'between%':>9}")
print("-" * 60)
for _, r in var_df.iterrows():
    print(f"{int(r['hour']):>3}  {r['load_share_mean']:>10.4f}  "
          f"{r['within_std']:>10.5f}  {r['within_share']*100:>7.1f}%  "
          f"{(1-r['within_share'])*100:>8.1f}%")

print(f"\nMedian within-country share of total variance: "
      f"{var_df['within_share'].median()*100:.1f}%")
print("If within% is small (<10%), load shape barely moves within countries -> "
      "country FE will absorb essentially all of it.")


# ---------------------------------------------------------------------------
# 2. M_load_only: just FE + load_share_h
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. Stand-alone load-share model: dev_h ~ alpha_c + load_share_h")
print("=" * 78)

def make_design_simple(df, train_countries, predictors):
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(train_df, predictors):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = (train_df[train_df["hour"] == h]
               .dropna(subset=["deviation"] + predictors))
        if len(sub) < len(predictors) + len(countries) + 5:
            continue
        X = make_design_simple(sub, countries, predictors)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": list(X.columns),
                  "residuals": (y - res.predict(X)).values,
                  "tvalues": res.tvalues, "rsq": float(res.rsquared),
                  "n": int(res.nobs), "countries": countries}
    return out


panel_use = panel.dropna(subset=["deviation", "load_share_h"] + CAPSHARES + EXPOSED + ["ntc_share"]).copy()
print(f"Sample (no-NaN on all of caps + exposed + ntc + load): {len(panel_use)} rows")

m_load_only   = fit_per_hour(panel_use, ["load_share_h"])
m_caps_only   = fit_per_hour(panel_use, CAPSHARES)
m_caps_load   = fit_per_hour(panel_use, CAPSHARES + ["load_share_h"])
m_full        = fit_per_hour(panel_use, CAPSHARES + EXPOSED + ["ntc_share", "load_share_h"])


def summarize(model_dict, name):
    if not model_dict:
        return name, np.nan, np.nan
    r2s = [m["rsq"] for m in model_dict.values()]
    return name, np.mean(r2s), np.median(r2s)


print("\nIn-sample R^2 (median across hours):")
for name, models in [("M_load_only (FE + load_share)", m_load_only),
                     ("M_caps_only (FE + caps)", m_caps_only),
                     ("M_caps + load (FE + caps + load)", m_caps_load),
                     ("M_full (FE + caps + exposed + ntc + load)", m_full)]:
    _, mean_r2, med_r2 = summarize(models, name)
    print(f"  {name:<45s}  mean={mean_r2:.3f}, median={med_r2:.3f}")

# Pull the load coefficient + t-stat across model specs to see if it shrinks
def extract_load_stats(model_dict):
    rows = []
    for h, m in model_dict.items():
        if "load_share_h" not in m["x_cols"]:
            continue
        idx = m["x_cols"].index("load_share_h")
        rows.append({"hour": h,
                     "coef": float(m["params"].values[idx]),
                     "t": float(m["tvalues"].values[idx])})
    return pd.DataFrame(rows).sort_values("hour")


load_stats_only  = extract_load_stats(m_load_only)
load_stats_caps  = extract_load_stats(m_caps_load)
load_stats_full  = extract_load_stats(m_full)

print("\nLoad-share coefficient & t-stat across specs (per hour):")
print(f"{'h':>3}  {'coef_only':>11}  {'t_only':>7}  {'coef_+caps':>12}  "
      f"{'t_+caps':>9}  {'coef_full':>11}  {'t_full':>7}")
print("-" * 80)
for h in range(24):
    r1 = load_stats_only.query(f"hour == {h}")
    r2 = load_stats_caps.query(f"hour == {h}")
    r3 = load_stats_full.query(f"hour == {h}")
    if r1.empty or r2.empty or r3.empty:
        continue
    print(f"{h:>3}  {r1['coef'].iloc[0]:>+11.0f}  {r1['t'].iloc[0]:>+7.2f}  "
          f"{r2['coef'].iloc[0]:>+12.0f}  {r2['t'].iloc[0]:>+9.2f}  "
          f"{r3['coef'].iloc[0]:>+11.0f}  {r3['t'].iloc[0]:>+7.2f}")


# ---------------------------------------------------------------------------
# 3. DE load shape trajectory 2015-2025: has it actually shifted?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. DE load shape trajectory 2015-2025")
print("=" * 78)

de_load = (load_long[load_long["country"] == "DE"]
           .pivot(index="year", columns="hour", values="load_share_h"))
print("\nDE share at key hours (%) by year:")
keep_h = [0, 3, 6, 9, 12, 15, 18, 21]
print(de_load[keep_h].round(4).mul(100).to_string())
print()
# Trend per hour: slope of share vs year
slopes = []
for h in range(24):
    if h not in de_load.columns: continue
    sub = de_load[h].dropna()
    if len(sub) < 3: continue
    s, _ = np.polyfit(sub.index.values, sub.values, 1)
    slopes.append({"hour": h, "slope_per_year": s,
                   "total_change_2015_2025": s * (sub.index.max() - sub.index.min())})
slopes_df = pd.DataFrame(slopes)
print("DE per-hour slope of load_share over 2015->2025 (pp shift over 10 years):")
print(f"{'h':>3}  {'slope/yr (pp)':>13}  {'10yr change (pp)':>17}")
print("-" * 50)
for _, r in slopes_df.iterrows():
    print(f"{int(r['hour']):>3}  "
          f"{r['slope_per_year']*100:>+13.4f}  "
          f"{r['total_change_2015_2025']*100:>+17.3f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.bar(var_df["hour"], var_df["within_share"] * 100, color="#1f77b4",
       label="within-country %")
ax.bar(var_df["hour"], (1 - var_df["within_share"]) * 100,
       bottom=var_df["within_share"] * 100, color="#d62728",
       label="between-country %")
ax.set_xlabel("Hour of day"); ax.set_ylabel("% of total load_share variance")
ax.set_xticks(range(0, 24, 3))
ax.set_title("load_share_h variance: within vs between country")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "load_shape_variance_decomposition.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "load_shape_variance_decomposition.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# DE trajectory
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5), constrained_layout=True)
ax = axes[0]
for h in [0, 3, 6, 9, 12, 15, 18, 21]:
    if h in de_load.columns:
        ax.plot(de_load.index, de_load[h] * 100, marker="o", markersize=3,
                label=f"h{h:02d}")
ax.set_xlabel("Year"); ax.set_ylabel("DE load share at hour (%)")
ax.set_title("DE within-day load share by year (selected hours)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))

ax2 = axes[1]
ax2.bar(slopes_df["hour"], slopes_df["total_change_2015_2025"] * 100,
        color=["#1f77b4" if v < 0 else "#d62728" for v in slopes_df["total_change_2015_2025"]])
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Hour of day"); ax2.set_ylabel("DE 2015 -> 2025 change in load share (pp)")
ax2.set_title("DE: how much did the load share at each hour move over 10 years?")
ax2.set_xticks(range(0, 24, 3))
ax2.spines[["top", "right"]].set_visible(False)

fig.savefig(PLOTS / "load_shape_de_trajectory.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "load_shape_de_trajectory.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nWrote plots to", PLOTS)
