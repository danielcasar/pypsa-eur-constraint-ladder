"""
Track how the structural-shape coefficients evolve over time.

Method: rolling 3-year windows (2015-17, 2016-18, ..., 2023-25 = 9 windows).
Per window, fit the same pooled-FE per-hour regression we use for the
headline result. Track how key coefficients evolve.

This is the cheap-but-meaningful version of "parameter evolution":
no ARMAX-GARCH-X, just the existing regression refit on sliding windows.

Outputs:
  Analysis/results/rolling_window_coefs.csv
  Analysis/results/rolling_window_evolution_summary.txt
  Analysis/results/plots/rolling_window_evolution_solar.{pdf,png}
  Analysis/results/plots/rolling_window_evolution_grid.{pdf,png}
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

# Rolling 3-year windows centered on each year 2016-2024.
WINDOWS = [(y - 1, y, y + 1) for y in range(2016, 2025)]

DISPLAY_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
                 "nuclear", "gas", "hardcoal"]
TECH_LABEL = {"solar":"Solar","onwind":"Onshore wind","offwind":"Offshore wind",
              "hydro_disp":"Hydro (res+pumped)","hydro_ror":"Hydro (RoR)",
              "nuclear":"Nuclear","gas":"Gas","hardcoal":"Hard coal"}
PALETTE = {"solar":"#fdb863","onwind":"#5e3c99","offwind":"#3690c0",
           "hydro_disp":"#0570b0","hydro_ror":"#74a9cf","biomass":"#33a02c",
           "nuclear":"#e41a1c","gas":"#fb9a99","hardcoal":"#525252",
           "lignite":"#8c510a","oil":"#bf812d"}


# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
panel_all = (dev.merge(state, on=["country", "year"], how="inner")
                .query("country in @COUNTRIES")
                .dropna(subset=["deviation"] + PREDICTORS))


def make_X(df, train_countries, predictors):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(df, predictors):
    out = {}
    countries = sorted(df["country"].unique())
    for h in range(24):
        sub = df[df["hour"] == h].dropna(subset=["deviation"] + predictors)
        if len(sub) < len(predictors) + len(countries) + 5:
            continue
        X = make_X(sub, countries, predictors)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": list(X.columns),
                  "tvalues": res.tvalues,
                  "se": dict(zip(X.columns, res.bse)),
                  "rsq": float(res.rsquared)}
    return out


# ---------------------------------------------------------------------------
# Fit each rolling window
# ---------------------------------------------------------------------------
rows = []
for window in WINDOWS:
    center = window[1]
    sub = panel_all[panel_all["year"].isin(window)]
    if len(sub) < 100:
        continue
    print(f"Window {window} (center {center}): {len(sub)} rows")
    m = fit_per_hour(sub, PREDICTORS)
    for h, mod in m.items():
        for t in DIFF_TECHS:
            name = f"capnorm_{t}"
            if name not in mod["x_cols"]: continue
            idx = mod["x_cols"].index(name)
            rows.append({
                "center_year": center, "hour": h, "tech": t,
                "coef": float(mod["params"].values[idx]),
                "se":   float(mod["se"][name]),
                "t":    float(mod["tvalues"].values[idx]),
                "rsq":  float(mod["rsq"]),
            })

evol = pd.DataFrame(rows)
evol.to_csv(RESULTS / "rolling_window_coefs.csv", index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# A. Solar at key hours (h10 trough, h17 peak) over time
fig, ax = plt.subplots(figsize=(7.0, 3.2))
for h, color, label in [(10, "#1f77b4", "Solar at h10 (noon trough)"),
                        (17, "#d62728", "Solar at h17 (evening peak)")]:
    sub = evol[(evol["tech"] == "solar") & (evol["hour"] == h)].sort_values("center_year")
    ax.errorbar(sub["center_year"], sub["coef"], yerr=1.96 * sub["se"],
                color=color, marker="o", markersize=4, capsize=2,
                linewidth=1.4, label=label)
ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Rolling 3-year window (center year)")
ax.set_ylabel(r"$\beta_{\mathrm{solar}}^{(h)}$ (€/MWh per +1.0 cap/peak_load)")
ax.set_title("Solar's structural-shape coefficient over time (rolling 3-year windows)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="best")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "rolling_window_evolution_solar.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "rolling_window_evolution_solar.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# B. Small-multiples grid: all main techs at key hours
fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.5), sharex=True,
                          constrained_layout=True)
for i, t in enumerate(DISPLAY_TECHS):
    ax = axes.flatten()[i]
    # For each tech pick the hour with the largest |coef| as the "signature"
    full_panel_coef = evol[evol["tech"] == t]
    if full_panel_coef.empty: continue
    # Plot two key hours: where this tech matters most and least
    coefs_by_hour = full_panel_coef.groupby("hour")["coef"].mean().abs()
    if len(coefs_by_hour) == 0: continue
    h_strong = int(coefs_by_hour.idxmax())
    sub = full_panel_coef[full_panel_coef["hour"] == h_strong].sort_values("center_year")
    ax.errorbar(sub["center_year"], sub["coef"], yerr=1.96 * sub["se"],
                color=PALETTE[t], marker="o", markersize=3.5, capsize=2,
                linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_title(f"{TECH_LABEL[t]} at h{h_strong:02d}", fontsize=8, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    if i % 4 == 0:
        ax.set_ylabel(r"$\beta$ (€/MWh per +1.0 capnorm)", fontsize=7)
    if i // 4 == 1:
        ax.set_xlabel("Center year of 3-year window", fontsize=7)
fig.suptitle("Evolution of structural-shape coefficients over rolling 3-year windows",
             fontsize=9, y=1.01, fontweight="bold")
fig.savefig(PLOTS / "rolling_window_evolution_grid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "rolling_window_evolution_grid.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Rolling 3-year window: evolution of structural-shape coefficients")
out.append("=" * 78)
out.append(f"\nWindows: 9 rolling 3-year windows centered on 2016-2024.")
out.append(f"Total (window, hour, tech) coefficient rows: {len(evol)}")

# Solar h10 evolution (the headline)
sol10 = evol[(evol["tech"] == "solar") & (evol["hour"] == 10)].sort_values("center_year")
out.append("\nSolar h10 (noon trough) coefficient over windows:")
out.append(f"  {'center':>7}  {'coef':>8}  {'SE':>6}  {'t':>5}")
out.append("  " + "-" * 35)
for _, r in sol10.iterrows():
    out.append(f"  {int(r['center_year']):>7}  {r['coef']:>+8.2f}  {r['se']:>6.2f}  {r['t']:>+5.2f}")

# Linear trend in solar h10 coefficient
if len(sol10) >= 3:
    slope, intercept = np.polyfit(sol10["center_year"], sol10["coef"], 1)
    out.append(f"\nLinear trend: slope = {slope:+.3f} €/MWh per year, "
               f"intercept = {intercept:+.2f}")
    out.append("Interpretation: if slope is negative, solar's noon-depressing "
               "effect is intensifying over time.")

# Solar h17 evolution
sol17 = evol[(evol["tech"] == "solar") & (evol["hour"] == 17)].sort_values("center_year")
out.append("\nSolar h17 (evening peak) coefficient over windows:")
out.append(f"  {'center':>7}  {'coef':>8}  {'SE':>6}  {'t':>5}")
out.append("  " + "-" * 35)
for _, r in sol17.iterrows():
    out.append(f"  {int(r['center_year']):>7}  {r['coef']:>+8.2f}  {r['se']:>6.2f}  {r['t']:>+5.2f}")

# Summary of which techs evolved
out.append("\nLinear trend in coefficient over time (slope per year), per (tech, hour):")
out.append("Negative slope at duck-hours = effect intensifying.")
out.append(f"\n{'tech':<12s} {'h':>3} {'slope/yr':>8} {'starting':>8} {'ending':>8}")
out.append("-" * 50)
trend_rows = []
for t in DIFF_TECHS:
    for h in [10, 12, 17]:
        sub = evol[(evol["tech"] == t) & (evol["hour"] == h)].sort_values("center_year")
        if len(sub) < 3: continue
        slope, intercept = np.polyfit(sub["center_year"], sub["coef"], 1)
        start_coef = sub["coef"].iloc[0]
        end_coef   = sub["coef"].iloc[-1]
        trend_rows.append({"tech": t, "hour": h, "slope_per_year": slope,
                           "start": start_coef, "end": end_coef})
        out.append(f"{t:<12s} {h:>3} {slope:>+8.2f} {start_coef:>+8.2f} {end_coef:>+8.2f}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "rolling_window_evolution_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'rolling_window_evolution_summary.txt'}")
print(f"       {PLOTS / 'rolling_window_evolution_solar.pdf'}")
print(f"       {PLOTS / 'rolling_window_evolution_grid.pdf'}")
