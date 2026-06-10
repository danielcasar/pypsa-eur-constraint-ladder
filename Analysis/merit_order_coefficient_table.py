"""
Compute per-technology absolute-price coefficients by combining:
  (1) a LEVEL model:    mean_price ~ country FE + capnorm + fuel + CO2 + coal + load
  (2) a SHAPE model:    dev_h ~ country FE + year FE + capnorm + exposed + ntc

  Total absolute-price effect at hour h:
      beta_total^(h)(tech) = beta_level(tech) + beta_dev^(h)(tech)

  Standard errors propagate: SE_total = sqrt(SE_level^2 + SE_dev^2)
  (Approximate — assumes independence between the two regressions, which holds
  because the dev model uses mean-removed targets, orthogonal to the mean.)

Output: per-tech, per-hour absolute-price coefficient with 95% CI.

Outputs:
  Analysis/results/merit_order_coefficient_table.csv
  Analysis/results/merit_order_coefficient_summary.txt
  Analysis/results/plots/merit_order_coefficient_solar.{pdf,png}
  Analysis/results/plots/merit_order_coefficient_all_techs.{pdf,png}
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
DATA = HERE / "data"
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


# ---------------------------------------------------------------------------
# 1. Build annual fuel-price series (year-varying, common to all countries)
# ---------------------------------------------------------------------------
gas = pd.read_csv(DATA / "natural-gas-prices.csv")
gas_ttf = gas[gas["Entity"] == "Netherlands TTF"][["Year", "Gas price"]].rename(
    columns={"Year": "year", "Gas price": "gas_eur_mwh"})

coal = pd.read_csv(DATA / "coal-prices.csv")
coal_nweu = coal[coal["Entity"] == "Northwest Europe"][["Year", "Coal"]].rename(
    columns={"Year": "year", "Coal": "coal_eur_t"})

eua_fuel = pd.read_csv(DATA / "fuel_prices.csv")[["year", "eua_eur_per_t"]].rename(
    columns={"eua_eur_per_t": "eua_eur_t"})
# Add historical EUA for 2015-2017 from public records (annual averages, EUA spot)
hist_eua = pd.DataFrame({"year": [2015, 2016, 2017],
                          "eua_eur_t": [7.69, 5.34, 5.84]})
eua = pd.concat([hist_eua, eua_fuel], ignore_index=True).drop_duplicates("year").sort_values("year")

fuel_panel = (gas_ttf.merge(coal_nweu, on="year", how="outer")
                    .merge(eua, on="year", how="outer")
                    .sort_values("year"))
print("Annual fuel prices used:")
print(fuel_panel[(fuel_panel["year"] >= 2015) & (fuel_panel["year"] <= 2025)].to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Build annual panel (country, year): mean_price, capnorm, load, fuel
# ---------------------------------------------------------------------------
dev = pd.read_csv(DEV_FILE)
annual_price = (dev.groupby(["country", "year"])["annual_mean"].first().reset_index()
                .rename(columns={"annual_mean": "mean_price"}))
state = pd.read_csv(STATE_FILE)

panel_lvl = (annual_price.merge(state, on=["country", "year"], how="inner")
             .merge(fuel_panel, on="year", how="left")
             .query("country in @COUNTRIES")
             .dropna(subset=["mean_price", "gas_eur_mwh", "coal_eur_t",
                             "eua_eur_t"] + CAPNORM + ["mean_load_mw"]))
print(f"\nLevel-model panel: {len(panel_lvl)} rows ({panel_lvl['country'].nunique()} "
      f"countries x {panel_lvl['year'].nunique()} years)")


# ---------------------------------------------------------------------------
# 3. Fit LEVEL model: mean_price ~ country FE + capnorm + fuel
# ---------------------------------------------------------------------------
def build_X(df, predictors, *, country_fe=True, year_fe=False):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    countries = sorted(df["country"].unique())
    years = sorted(df["year"].unique())
    if country_fe:
        for c in countries[1:]:
            X[f"FE_c_{c}"] = (df["country"].values == c).astype(float)
    if year_fe:
        for y in years[1:]:
            X[f"FE_y_{int(y)}"] = (df["year"].values == int(y)).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


level_predictors = CAPNORM + ["gas_eur_mwh", "coal_eur_t", "eua_eur_t", "mean_load_mw"]
X_lvl = build_X(panel_lvl, level_predictors, country_fe=True, year_fe=False)
y_lvl = panel_lvl["mean_price"].astype(float)
lvl_res = sm.OLS(y_lvl, X_lvl).fit()
print(f"\nLevel model R^2 = {lvl_res.rsquared:.3f}, RMSE = "
      f"{np.sqrt(lvl_res.mse_resid):.2f} EUR/MWh, n = {int(lvl_res.nobs)}")

# Extract level coefficients on the structural predictors
level_coef = {}
for t in DIFF_TECHS:
    name = f"capnorm_{t}"
    level_coef[t] = {"coef": float(lvl_res.params[name]),
                     "se":   float(lvl_res.bse[name]),
                     "t":    float(lvl_res.tvalues[name]),
                     "p":    float(lvl_res.pvalues[name])}
print("\nLevel coefficients (€/MWh on annual mean per +1.0 capnorm):")
for t in DIFF_TECHS:
    c = level_coef[t]
    sig = "***" if abs(c["t"]) > 2 else ""
    print(f"  {t:<12s}: {c['coef']:>+8.2f}  (SE={c['se']:>5.2f}, t={c['t']:>+5.2f}) {sig}")
print(f"  gas_eur_mwh : {lvl_res.params['gas_eur_mwh']:>+8.3f} "
      f"(t={lvl_res.tvalues['gas_eur_mwh']:>+5.2f})")
print(f"  eua_eur_t   : {lvl_res.params['eua_eur_t']:>+8.3f} "
      f"(t={lvl_res.tvalues['eua_eur_t']:>+5.2f})")
print(f"  coal_eur_t  : {lvl_res.params['coal_eur_t']:>+8.3f} "
      f"(t={lvl_res.tvalues['coal_eur_t']:>+5.2f})")


# ---------------------------------------------------------------------------
# 4. Fit SHAPE model: dev_h ~ country FE + year FE + capnorm + exposed + ntc
#    Use two-way FE for clean within-country, within-year identification
# ---------------------------------------------------------------------------
panel_shape = (dev.merge(state, on=["country", "year"], how="inner")
                .query("country in @COUNTRIES")
                .dropna(subset=["deviation"] + CAPNORM + EXPOSED + ["ntc_norm"]))
print(f"\nShape-model panel: {len(panel_shape)} rows")

shape_predictors = CAPNORM + EXPOSED + ["ntc_norm"]
shape_coef = {h: {} for h in range(24)}
shape_res_objs = {}

for h in range(24):
    sub = panel_shape[panel_shape["hour"] == h]
    if sub.empty:
        continue
    Xh = build_X(sub, shape_predictors, country_fe=True, year_fe=True)
    yh = sub["deviation"].astype(float)
    res = sm.OLS(yh, Xh).fit()
    shape_res_objs[h] = res
    for t in DIFF_TECHS:
        name = f"capnorm_{t}"
        shape_coef[h][t] = {
            "coef": float(res.params[name]),
            "se":   float(res.bse[name]),
            "t":    float(res.tvalues[name]),
        }


# ---------------------------------------------------------------------------
# 5. Combine level + shape -> absolute price coefficient at each hour
# ---------------------------------------------------------------------------
rows = []
for t in DIFF_TECHS:
    lc = level_coef[t]
    for h in range(24):
        sc = shape_coef[h][t]
        total_coef = lc["coef"] + sc["coef"]
        # SE propagation: sqrt(SE_lvl^2 + SE_dev^2)
        total_se = float(np.sqrt(lc["se"]**2 + sc["se"]**2))
        ci_lo = total_coef - 1.96 * total_se
        ci_hi = total_coef + 1.96 * total_se
        rows.append({
            "tech": t, "hour": h,
            "level_coef":   round(lc["coef"], 3),
            "level_se":     round(lc["se"], 3),
            "level_t":      round(lc["t"], 2),
            "dev_coef":     round(sc["coef"], 3),
            "dev_se":       round(sc["se"], 3),
            "dev_t":        round(sc["t"], 2),
            "total_coef":   round(total_coef, 3),
            "total_se":     round(total_se, 3),
            "total_ci_lo":  round(ci_lo, 3),
            "total_ci_hi":  round(ci_hi, 3),
        })
table = pd.DataFrame(rows)
table.to_csv(RESULTS / "merit_order_coefficient_table.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Per-technology absolute-price coefficient")
out.append("Total effect = LEVEL effect (annual mean) + SHAPE effect (at hour h)")
out.append("=" * 78)
out.append(f"\nLevel model: mean_price ~ country FE + capnorm + gas + coal + CO2 + load")
out.append(f"  n = {int(lvl_res.nobs)}, R² = {lvl_res.rsquared:.3f}, "
           f"RMSE = {np.sqrt(lvl_res.mse_resid):.2f} EUR/MWh")
out.append(f"\nShape model: dev_h ~ country FE + year FE + capnorm + exposed + ntc_norm")

out.append("\nLevel coefficients on capnorm (per +1.0 cap/peak_load, annual-mean impact):")
out.append(f"  {'tech':<12} {'coef':>8} {'SE':>6} {'t':>6}  {'95% CI':>20}")
out.append("  " + "-" * 60)
for t in DIFF_TECHS:
    c = level_coef[t]
    ci = f"[{c['coef']-1.96*c['se']:+.2f}, {c['coef']+1.96*c['se']:+.2f}]"
    sig = " ***" if abs(c["t"]) > 2 else ""
    out.append(f"  {t:<12} {c['coef']:>+8.2f} {c['se']:>6.2f} {c['t']:>+6.2f}  {ci:>20}{sig}")

out.append("\nFuel-price coefficients on annual mean:")
out.append(f"  gas_eur_mwh : {lvl_res.params['gas_eur_mwh']:>+7.3f} EUR/MWh per +1 EUR/MWh gas "
           f"(t={lvl_res.tvalues['gas_eur_mwh']:>+5.2f})")
out.append(f"  eua_eur_t   : {lvl_res.params['eua_eur_t']:>+7.3f} EUR/MWh per +1 EUR/t CO2 "
           f"(t={lvl_res.tvalues['eua_eur_t']:>+5.2f})")
out.append(f"  coal_eur_t  : {lvl_res.params['coal_eur_t']:>+7.3f} EUR/MWh per +1 EUR/t coal "
           f"(t={lvl_res.tvalues['coal_eur_t']:>+5.2f})")

# Solar per-hour absolute-price coefficient
out.append("\nSolar absolute-price coefficient by hour:")
out.append(f"  {'h':>3}  {'level':>8}  {'shape':>8}  {'TOTAL':>8}  {'95% CI':>20}")
out.append("  " + "-" * 60)
for h in range(24):
    r = table[(table["tech"] == "solar") & (table["hour"] == h)].iloc[0]
    ci = f"[{r['total_ci_lo']:+.2f}, {r['total_ci_hi']:+.2f}]"
    out.append(f"  {int(h):>3}  {r['level_coef']:>+8.2f}  {r['dev_coef']:>+8.2f}  "
               f"{r['total_coef']:>+8.2f}  {ci:>20}")

out.append("\n=== KEY STATEMENT EXAMPLE (solar, h10) ===")
r = table[(table["tech"] == "solar") & (table["hour"] == 10)].iloc[0]
out.append(f"  Adding +1.0 cap/peak_load of solar (extreme: from 0 to capacity = peak load)")
out.append(f"  is associated with a {r['total_coef']:+.1f} EUR/MWh change in the h10 absolute price")
out.append(f"  (95% CI: [{r['total_ci_lo']:+.1f}, {r['total_ci_hi']:+.1f}]).")
out.append(f"  Decomposition: level effect {r['level_coef']:+.1f} + shape effect {r['dev_coef']:+.1f}")
out.append("\n  For a realistic +0.10 (10pp) change:")
out.append(f"    -> h10 price change: {r['total_coef']*0.1:+.2f} EUR/MWh "
           f"(95% CI: [{r['total_ci_lo']*0.1:+.2f}, {r['total_ci_hi']*0.1:+.2f}])")

text = "\n".join(out)
print()
print(text)
(RESULTS / "merit_order_coefficient_summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
# 7a. Solar absolute coefficient by hour with CI
fig, ax = plt.subplots(figsize=(6.7, 3.2))
sol = table[table["tech"] == "solar"].sort_values("hour")
ax.fill_between(sol["hour"], sol["total_ci_lo"], sol["total_ci_hi"],
                color="#fdb863", alpha=0.35, linewidth=0, label="95% CI")
ax.plot(sol["hour"], sol["total_coef"], color="#d97b00", linewidth=1.5,
        marker="o", markersize=4, label="Total absolute-price effect")
ax.plot(sol["hour"], sol["level_coef"].iloc[0] * np.ones(len(sol)),
        color="#5f3dc4", linewidth=1.0, linestyle="--",
        label=f"Level component (constant, {sol['level_coef'].iloc[0]:+.1f})")
ax.plot(sol["hour"], sol["dev_coef"], color="#1f5fa8", linewidth=1.0,
        linestyle=":", label="Shape component (hour-varying)")
ax.axhline(0, color="black", linewidth=0.4)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Solar absolute-price effect (EUR/MWh per +1.0 cap/peak_load)")
ax.set_xticks(range(0, 24, 3))
ax.set_title("Solar: total absolute-price effect = level + hour-varying shape")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="best")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "merit_order_coefficient_solar.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "merit_order_coefficient_solar.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. All techs total coefficient by hour
PALETTE = {"solar":"#fdb863","onwind":"#5e3c99","offwind":"#3690c0",
           "hydro_disp":"#0570b0","hydro_ror":"#74a9cf","biomass":"#33a02c",
           "nuclear":"#e41a1c","gas":"#fb9a99","hardcoal":"#525252",
           "lignite":"#8c510a","oil":"#bf812d"}
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for t in DIFF_TECHS:
    sub = table[table["tech"] == t].sort_values("hour")
    ax.plot(sub["hour"], sub["total_coef"], color=PALETTE[t], linewidth=1.2,
            marker="o", markersize=2.8, label=t)
ax.axhline(0, color="black", linewidth=0.4)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Absolute-price effect (EUR/MWh per +1.0 cap/peak_load)")
ax.set_xticks(range(0, 24, 3))
ax.set_title("Per-technology absolute-price coefficient (level + shape combined)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.17))
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "merit_order_coefficient_all_techs.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "merit_order_coefficient_all_techs.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nWrote: {RESULTS / 'merit_order_coefficient_table.csv'}")
print(f"       {RESULTS / 'merit_order_coefficient_summary.txt'}")
print(f"       {PLOTS / 'merit_order_coefficient_solar.pdf'}")
print(f"       {PLOTS / 'merit_order_coefficient_all_techs.pdf'}")
