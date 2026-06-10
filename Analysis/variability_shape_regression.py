"""
Variability-shape regression: parallel structural analysis to the mean-shape
regression, with the within-day price-variability profile as target.

Definition:
  For each (country, year, hour-of-day h):
    var_h(c, y) = std over all days d in year y of  price(c, y, d, h)

  (i.e., how variable the noon price is across all noons of that year)

Model: same pooled-FE specification on capnorm + neighbour + ntc, with
country FE absorbing time-invariant variability differences across markets.

This delivers a *second* structural shape: instead of "at hour h, the
typical price is X €/MWh above the annual mean," we get "at hour h, the
spread of prices across days is Y €/MWh, and Y depends on the system mix."

Outputs:
  Analysis/results/variability_panel.csv
  Analysis/results/variability_shape_regression_summary.txt
  Analysis/results/variability_shape_coefs.csv
  Analysis/results/plots/variability_shape_profile.{pdf,png}
  Analysis/results/plots/variability_shape_tech_slopes.{pdf,png}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

PAPER1_PRICES = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators/Code/data/"
    "ENTSOE_day_ahead_prices.csv")
STATE_FILE = RESULTS / "system_state_capnorm_panel.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPNORM = [f"capnorm_{t}" for t in DIFF_TECHS]
EXPOSED = [f"exposed_capnorm_{t}" for t in DIFF_TECHS]
PREDICTORS = CAPNORM + EXPOSED + ["ntc_norm"]
YEARS = list(range(2015, 2026))


# ---------------------------------------------------------------------------
# 1. Compute within-day price variability per (country, year, hour)
# ---------------------------------------------------------------------------
print("Loading raw hourly prices...")
prices = pd.read_csv(PAPER1_PRICES, index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index, utc=True)
prices = prices[[c for c in prices.columns if c in COUNTRIES]]
prices = prices[prices.index.year.isin(YEARS)]
print(f"  loaded {prices.shape}; countries: {list(prices.columns)}")

print("\nComputing std per (country, year, hour-of-day)...")
prices.index.name = "timestamp"
prices_long = (prices.reset_index()
                  .melt(id_vars="timestamp", var_name="country", value_name="price")
                  .dropna(subset=["price"]))
prices_long["year"] = prices_long["timestamp"].dt.year
prices_long["hour"] = prices_long["timestamp"].dt.hour

# Standard deviation of price across days at each (country, year, hour)
var_panel = (prices_long.groupby(["country", "year", "hour"])["price"]
                       .std()
                       .reset_index()
                       .rename(columns={"price": "variability_std"}))
var_panel.to_csv(RESULTS / "variability_panel.csv", index=False)
print(f"  -> {len(var_panel)} (country, year, hour) cells; "
      f"median std = {var_panel['variability_std'].median():.2f} EUR/MWh; "
      f"range {var_panel['variability_std'].min():.1f}-"
      f"{var_panel['variability_std'].max():.1f}")


# ---------------------------------------------------------------------------
# 2. Merge with state panel + fit per-hour regression
# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
panel = (var_panel.merge(state, on=["country", "year"], how="inner")
                  .query("country in @COUNTRIES")
                  .dropna(subset=["variability_std"] + PREDICTORS))
print(f"\nMerged panel: {len(panel)} rows")


def make_X(df, train_countries, predictors):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(train_df, predictors, target="variability_std"):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=[target] + predictors)
        if len(sub) < 50:
            continue
        X = make_X(sub, countries, predictors)
        y = sub[target].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": list(X.columns),
                  "tvalues": res.tvalues, "rsq": float(res.rsquared),
                  "residuals": (y - res.predict(X)).values,
                  "countries": countries, "n": int(res.nobs)}
    return out


print("\nFitting variability-shape regression per hour ...")
m_full = fit_per_hour(panel, PREDICTORS)
print(f"  Fit hours: {len(m_full)}/24; "
      f"median in-sample R² = {np.median([m['rsq'] for m in m_full.values()]):.3f}")


# LOO across years
print("\nLOO across years ...")
def predict(model_h, country, row, predictors):
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}; x["const"] = 1.0
    if f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for s in predictors:
        x[s] = float(row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


loo_rows = []
for y0 in sorted(panel["year"].unique()):
    tr = panel[panel["year"] != y0]
    te = panel[panel["year"] == y0]
    m_loo = fit_per_hour(tr, PREDICTORS)
    for _, r in te.iterrows():
        h = int(r["hour"])
        if h not in m_loo or r["country"] not in m_loo[h]["countries"]:
            continue
        p = predict(m_loo[h], r["country"], r, PREDICTORS)
        loo_rows.append({"country": r["country"], "year": int(y0), "hour": h,
                         "observed": float(r["variability_std"]),
                         "predicted": p,
                         "residual": float(r["variability_std"]) - p})
loo = pd.DataFrame(loo_rows)
ss_res = (loo["residual"] ** 2).sum()
ss_tot = ((loo["observed"] - loo["observed"].mean()) ** 2).sum()
overall_r2 = 1 - ss_res / ss_tot
overall_rmse = float(np.sqrt((loo["residual"] ** 2).mean()))
print(f"  Overall LOO R² = {overall_r2:.3f}, RMSE = {overall_rmse:.2f}")


# ---------------------------------------------------------------------------
# 3. Extract coefficients
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m_full.items():
    for name, coef, t in zip(m["x_cols"], m["params"].values, m["tvalues"].values):
        coef_rows.append({"hour": h, "predictor": name,
                          "coef": float(coef), "t": float(t)})
coefs = pd.DataFrame(coef_rows)
coefs.to_csv(RESULTS / "variability_shape_coefs.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
# 4a. Cross-country average variability profile (median across countries per year)
fig, ax = plt.subplots(figsize=(6.5, 3.0))
agg = panel.groupby(["year", "hour"])["variability_std"].median().reset_index()
years_sorted = sorted(agg["year"].unique())
cmap = matplotlib.colormaps["viridis"]
for i, y in enumerate(years_sorted):
    sub = agg[agg["year"] == y].sort_values("hour")
    ax.plot(sub["hour"], sub["variability_std"],
            color=cmap(i / len(years_sorted)),
            linewidth=1.2, marker="o", markersize=2.5, label=str(y))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Median std of price across days (€/MWh)")
ax.set_xticks(range(0, 24, 3))
ax.set_title("Within-day price variability profile by year\n"
             "(median across 22 European countries)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.18))
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "variability_shape_profile.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "variability_shape_profile.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 4b. Tech-slope profiles for variability (standardized)
DISPLAY_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
                 "nuclear", "gas", "hardcoal"]
TECH_LABEL = {"solar":"Solar","onwind":"Onshore wind","offwind":"Offshore wind",
              "hydro_disp":"Hydro (reservoir+pumped)","hydro_ror":"Hydro (RoR)",
              "nuclear":"Nuclear","gas":"Gas","hardcoal":"Hard coal"}

stds = {t: panel[f"capnorm_{t}"].std() for t in DIFF_TECHS}

# Heatmap of standardized variability coefficients
M = np.zeros((len(DISPLAY_TECHS), 24))
for i, t in enumerate(DISPLAY_TECHS):
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"].sort_values("hour")
    for _, r in sub.iterrows():
        M[i, int(r["hour"])] = r["coef"] * stds[t]

vmax = float(np.max(np.abs(M)))
fig, ax = plt.subplots(figsize=(7.5, 3.5))
cmap = "RdBu_r"
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
ax.set_yticks(range(len(DISPLAY_TECHS)))
ax.set_yticklabels([TECH_LABEL[t] for t in DISPLAY_TECHS])
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_title(
    "How each technology shifts the within-day price VARIABILITY profile  "
    "(standardized: €/MWh-std per +1 std of cap/peak_load)",
    fontsize=8.5,
)
for i in range(len(DISPLAY_TECHS)):
    for j in range(24):
        v = M[i, j]
        color = "white" if abs(v) > 0.55 * vmax else "black"
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                fontsize=5.5, color=color)
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.01)
cb.set_label("€/MWh-std per +1 std capnorm", fontsize=7)
cb.ax.tick_params(labelsize=6)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "variability_shape_tech_slopes.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "variability_shape_tech_slopes.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Variability-shape regression")
out.append("Target: std(price) across days within (country, year, hour)")
out.append("=" * 78)
out.append(f"\nPanel: {len(panel)} rows; {panel['country'].nunique()} countries; "
           f"years {panel['year'].min()}-{panel['year'].max()}")
out.append(f"\nMedian within-day std: {panel['variability_std'].median():.1f} EUR/MWh")
out.append(f"Range:                  {panel['variability_std'].min():.1f} - "
           f"{panel['variability_std'].max():.1f}")
out.append(f"\nOverall LOO R²:  {overall_r2:.3f}")
out.append(f"Overall LOO RMSE: {overall_rmse:.2f} EUR/MWh")

# Compare to mean-shape regression
out.append("\nCompare to mean-shape regression on same predictors:")
out.append("  Mean-shape    : LOO R² = 0.732, RMSE = 11.01")
out.append(f"  Variability   : LOO R² = {overall_r2:.3f}, RMSE = {overall_rmse:.2f}")

# Per-tech variability coefficient peak-trough range (standardized)
out.append("\nStandardized peak-trough range per tech (€/MWh-std per 1 std capnorm):")
for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"]
    if sub.empty:
        continue
    std_coef = sub["coef"].values * stds[t]
    out.append(f"  {t:<12s}: max-min = {std_coef.max() - std_coef.min():>+7.2f}")

# Key tech-hour cells
out.append("\nSolar variability coefficient by hour (raw, EUR/MWh-std per +1.0 capnorm):")
sol = coefs[coefs["predictor"] == "capnorm_solar"].sort_values("hour")
for _, r in sol.iterrows():
    sig = " ***" if abs(r["t"]) > 2 else ""
    out.append(f"  h{int(r['hour']):02d}: coef = {r['coef']:>+8.2f}, t = {r['t']:>+5.2f}{sig}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "variability_shape_regression_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'variability_shape_regression_summary.txt'}")
print(f"       {PLOTS / 'variability_shape_profile.pdf'}")
print(f"       {PLOTS / 'variability_shape_tech_slopes.pdf'}")
