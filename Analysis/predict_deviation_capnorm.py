"""
Replace `capshare_t` with `capnorm_t = capacity_t / peak_load` as the
per-tech predictor. Peak load = max hourly load per (country, year) from
ENTSO-E load data.

Motivation: capshares sum to 1 -> mechanical collinearity. When solar grows
by 5pp, something else has to fall by 5pp. The regression then attributes
solar's apparent effect to whichever predictor has more idiosyncratic
year-to-year variation (typically lignite/nuclear decline).

`capnorm` decouples each tech from the others -- solar can grow without
forcing anything else to shrink. Identification of the solar coefficient
becomes independent of fossil retirement timing.

Model per hour h (identical structure, new variable definitions):
  dev_h(c, y) = alpha_c^(h)
             + sum_t  beta_t^(h)   * capnorm_t(c, y)
             + sum_t  delta_t^(h)  * exposed_capnorm_t(c, y)
             + gamma^(h)           * ntc_norm(c, y)
             + eps

Outputs:
  Analysis/results/system_state_capnorm_panel.csv
  Analysis/results/predict_deviation_capnorm_summary.txt
  Analysis/results/predict_deviation_capnorm_coefs.csv
  Analysis/results/plots/predict_deviation_capnorm_compare.{pdf,png}
  Analysis/results/plots/predict_deviation_capnorm_tech_slopes.{pdf,png}
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
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE = RESULTS / "hod_deviation_profiles_long.csv"
CAP_RAW  = DATA / "ENTSOE_installed_capacity_2015_2025.csv"
LOAD_HRLY = DATA / "ENTSOE_load_2015_2025.csv"
NTC_RAW  = DATA / "ENTSOE_interconnection_per_border_2015_2025.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]

# Map ENTSO-E carrier labels -> framework tech buckets (same as before)
TECH_MAP = {
    "solar":      ["Solar"],
    "onwind":     ["Wind Onshore"],
    "offwind":    ["Wind Offshore"],
    "hydro_disp": ["Hydro Water Reservoir", "Hydro Pumped Storage"],
    "hydro_ror":  ["Hydro Run-of-river and poundage"],
    "biomass":    ["Biomass"],
    "nuclear":    ["Nuclear"],
    "gas":        ["Fossil Gas", "Fossil Coal-derived gas"],
    "hardcoal":   ["Fossil Hard coal"],
    "lignite":    ["Fossil Brown coal/Lignite", "Fossil Peat", "Fossil Oil shale"],
    "oil":        ["Fossil Oil"],
}
AREA_TO_COUNTRY = {
    "AT":"AT","BE":"BE","BG":"BG","CH":"CH","CZ":"CZ","DE_LU":"DE",
    "DK_1":"DK","DK_2":"DK","EE":"EE","ES":"ES","FI":"FI","FR":"FR","GB":"UK",
    "GR":"GR","HR":"HR","HU":"HU","IE_SEM":"IE","IT_NORD":"IT","IT_BRNN":"IT",
    "LT":"LT","LU":"LU","LV":"LV","NL":"NL","NO_2":"NO","NO_4":"NO","PL":"PL",
    "PT":"PT","RO":"RO","RS":"RS","SE_1":"SE","SE_3":"SE","SE_4":"SE","SI":"SI","SK":"SK",
}


# ---------------------------------------------------------------------------
# 1. Build peak load per (country, year)
# ---------------------------------------------------------------------------
print("Computing peak load per (country, year) ...")
load = pd.read_csv(LOAD_HRLY, index_col="timestamp", parse_dates=True)
load.index = pd.to_datetime(load.index, utc=True)
load["year"] = load.index.year
peak_load = []
for cc in COUNTRIES:
    if cc not in load.columns:
        continue
    for y, sub in load[[cc, "year"]].groupby("year"):
        v = sub[cc].dropna()
        if len(v) < 100:
            continue
        peak_load.append({"country": cc, "year": int(y),
                          "peak_load_mw": float(v.max()),
                          "mean_load_mw": float(v.mean())})
peak_load = pd.DataFrame(peak_load)
print(f"  -> {len(peak_load)} country-year rows; "
      f"mean peak load {peak_load['peak_load_mw'].mean()/1000:.1f} GW")


# ---------------------------------------------------------------------------
# 2. Build capnorm panel
# ---------------------------------------------------------------------------
print("\nBuilding capnorm panel ...")
cap = pd.read_csv(CAP_RAW)
cap = cap[cap["country"].isin(COUNTRIES)].copy()

rows = []
for _, r in cap.iterrows():
    out = {"country": r["country"], "year": int(r["year"])}
    for tech, src_cols in TECH_MAP.items():
        present = [s for s in src_cols if s in cap.columns]
        cap_mw = float(r[present].sum()) if present else 0.0
        out[f"cap_{tech}_mw"] = cap_mw
    rows.append(out)
cap_long = pd.DataFrame(rows).merge(peak_load, on=["country", "year"], how="inner")

for t in DIFF_TECHS:
    cap_long[f"capnorm_{t}"] = cap_long[f"cap_{t}_mw"] / cap_long["peak_load_mw"]
# Also compute total cap / peak_load
cap_long["total_capnorm"] = cap_long[[f"cap_{t}_mw" for t in DIFF_TECHS]].sum(axis=1) / cap_long["peak_load_mw"]

CAPNORM_COLS = [f"capnorm_{t}" for t in DIFF_TECHS]

print(f"  -> {len(cap_long)} (country, year) rows")
print(f"  Example capnorm values (DE):")
print(cap_long[cap_long["country"] == "DE"][["year", "peak_load_mw"] + CAPNORM_COLS].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 3. NTC-weighted neighbour capnorm + ntc_norm
# ---------------------------------------------------------------------------
print("\nBuilding neighbour-weighted exposed_capnorm ...")
ntc_raw = pd.read_csv(NTC_RAW)
ntc_raw["c_from"] = ntc_raw["from"].map(AREA_TO_COUNTRY)
ntc_raw["c_to"]   = ntc_raw["to"].map(AREA_TO_COUNTRY)
ntc_raw = ntc_raw.dropna(subset=["c_from", "c_to"])
ntc_raw = ntc_raw[ntc_raw["c_from"] != ntc_raw["c_to"]]

edges = []
for _, r in ntc_raw.iterrows():
    edges.append({"home": r["c_from"], "neighbor": r["c_to"],
                  "year": int(r["year"]), "ntc_mw": float(r["ntc_mw"])})
    edges.append({"home": r["c_to"], "neighbor": r["c_from"],
                  "year": int(r["year"]), "ntc_mw": float(r["ntc_mw"])})
edges = pd.DataFrame(edges).groupby(["home","neighbor","year"], as_index=False)["ntc_mw"].max()

cap_lookup = cap_long.set_index(["country","year"])[CAPNORM_COLS]
exp_rows = []
for (home, year), grp in edges.groupby(["home", "year"]):
    total_w = grp["ntc_mw"].sum()
    if total_w <= 0:
        continue
    weighted = {f"exposed_capnorm_{t}": 0.0 for t in DIFF_TECHS}
    weight_used = 0.0
    for _, e in grp.iterrows():
        try:
            n_caps = cap_lookup.loc[(e["neighbor"], year)]
        except KeyError:
            continue
        w = e["ntc_mw"]; weight_used += w
        for t in DIFF_TECHS:
            weighted[f"exposed_capnorm_{t}"] += w * float(n_caps[f"capnorm_{t}"])
    if weight_used <= 0: continue
    weighted = {k: v / weight_used for k, v in weighted.items()}
    exp_rows.append({"country": home, "year": int(year),
                     "total_neighbor_ntc": float(total_w), **weighted})
exposure = pd.DataFrame(exp_rows)

EXPOSED_COLS = [f"exposed_capnorm_{t}" for t in DIFF_TECHS]

# ntc_norm = total inbound+outbound / peak_load (per-border-pair already aggregated)
ntc_agg = (ntc_raw.assign(home=lambda d: d["c_from"], yr=lambda d: d["year"])
                  .pivot_table(index=["home", "year"], values="ntc_mw", aggfunc="sum")
                  .reset_index().rename(columns={"home": "country", "ntc_mw": "ntc_sum"}))
# Symmetric so add from "to" side
ntc_agg_to = (ntc_raw.assign(home=lambda d: d["c_to"], yr=lambda d: d["year"])
                     .pivot_table(index=["home", "year"], values="ntc_mw", aggfunc="sum")
                     .reset_index().rename(columns={"home": "country", "ntc_mw": "ntc_sum"}))
ntc_total = pd.concat([ntc_agg, ntc_agg_to]).groupby(["country","year"], as_index=False)["ntc_sum"].sum()
state = (cap_long.merge(exposure, on=["country","year"], how="left")
                 .merge(ntc_total, on=["country","year"], how="left"))
state["ntc_norm"] = state["ntc_sum"] / state["peak_load_mw"]
state.to_csv(RESULTS / "system_state_capnorm_panel.csv", index=False)
print(f"  Final panel: {len(state)} rows; "
      f"with full predictors (no NaN): "
      f"{state.dropna(subset=CAPNORM_COLS + EXPOSED_COLS + ['ntc_norm']).shape[0]}")


# ---------------------------------------------------------------------------
# 4. Merge with deviation panel, fit
# ---------------------------------------------------------------------------
dev = pd.read_csv(DEV_FILE)
PREDICTORS = CAPNORM_COLS + EXPOSED_COLS + ["ntc_norm"]
panel = (dev.merge(state, on=["country","year"], how="inner")
            .query("country in @COUNTRIES")
            .dropna(subset=["deviation"] + PREDICTORS))
print(f"\nMerged panel: {len(panel)} rows; "
      f"{panel['country'].nunique()} countries; "
      f"years {panel['year'].min()}-{panel['year'].max()}")


def make_X(df, train_countries, predictors):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(train_df, predictors):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=["deviation"] + predictors)
        if len(sub) < len(predictors) + len(countries) + 5:
            continue
        X = make_X(sub, countries, predictors)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": list(X.columns),
                  "residuals": (y - res.predict(X)).values,
                  "tvalues": res.tvalues, "rsq": float(res.rsquared),
                  "countries": countries, "n": int(res.nobs)}
    return out


def predict(model_h, country, row, predictors):
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}; x["const"] = 1.0
    if f"FE_{country}" in x: x[f"FE_{country}"] = 1.0
    for s in predictors:
        x[s] = float(row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


print("\nFitting full-panel + LOO on capnorm spec ...")
m_full = fit_per_hour(panel, PREDICTORS)
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
                         "observed": float(r["deviation"]),
                         "predicted": p,
                         "residual": float(r["deviation"]) - p})
loo = pd.DataFrame(loo_rows)


def overall_metrics(df_loo):
    ss_res = (df_loo["residual"] ** 2).sum()
    ss_tot = ((df_loo["observed"] - df_loo["observed"].mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot,
            "rmse": float(np.sqrt((df_loo["residual"] ** 2).mean())),
            "n": int(len(df_loo))}

o = overall_metrics(loo)
print(f"  Overall LOO R2 = {o['r2']:.3f}, RMSE = {o['rmse']:.2f} EUR/MWh")


# Per-hour LOO R^2
def per_hour_r2(df_loo):
    rows = []
    for h, g in df_loo.groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({"hour": int(h),
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rmse": float(np.sqrt((g["residual"] ** 2).mean()))})
    return pd.DataFrame(rows).sort_values("hour")

ph = per_hour_r2(loo)


# ---------------------------------------------------------------------------
# 5. Tech-slope coefficients + standardized magnitudes
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m_full.items():
    for name, coef, t in zip(m["x_cols"], m["params"].values, m["tvalues"].values):
        coef_rows.append({"hour": h, "predictor": name, "coef": float(coef), "t": float(t)})
coefs = pd.DataFrame(coef_rows)
coefs.to_csv(RESULTS / "predict_deviation_capnorm_coefs.csv", index=False)

stds = {t: panel[f"capnorm_{t}"].std() for t in DIFF_TECHS}


# ---------------------------------------------------------------------------
# Compare to capshare baseline (use previously computed LOO from
# predict_deviation_pooled_with_neighbors.py results, if available)
# ---------------------------------------------------------------------------
# Re-compute capshare baseline metrics for matched sample for fairness
caps_old = pd.read_csv(RESULTS / "system_state_panel_with_neighbors.csv")
panel_old = (dev.merge(caps_old, on=["country","year"], how="inner")
                 .query("country in @COUNTRIES"))
CAPS_OLD = [f"capshare_{t}" for t in DIFF_TECHS]
EXP_OLD  = [f"exposed_{t}" for t in DIFF_TECHS]
panel_old = panel_old.dropna(subset=["deviation"] + CAPS_OLD + EXP_OLD + ["ntc_share"])
print(f"\nOld capshare panel for comparison: {len(panel_old)} rows")

m_old_full = fit_per_hour(panel_old, CAPS_OLD + EXP_OLD + ["ntc_share"])
loo_old_rows = []
for y0 in sorted(panel_old["year"].unique()):
    tr = panel_old[panel_old["year"] != y0]
    te = panel_old[panel_old["year"] == y0]
    m_loo = fit_per_hour(tr, CAPS_OLD + EXP_OLD + ["ntc_share"])
    for _, r in te.iterrows():
        h = int(r["hour"])
        if h not in m_loo or r["country"] not in m_loo[h]["countries"]:
            continue
        p = predict(m_loo[h], r["country"], r, CAPS_OLD + EXP_OLD + ["ntc_share"])
        loo_old_rows.append({"country": r["country"], "year": int(y0), "hour": h,
                             "observed": float(r["deviation"]),
                             "predicted": p,
                             "residual": float(r["deviation"]) - p})
loo_old = pd.DataFrame(loo_old_rows)
o_old = overall_metrics(loo_old)
ph_old = per_hour_r2(loo_old)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# A. Per-hour LOO R^2 comparison
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.plot(ph_old["hour"], ph_old["r2"], color="#999999", marker="o", markersize=3,
        linewidth=1.2, label=f"capshare baseline (R2={o_old['r2']:.3f}, RMSE={o_old['rmse']:.2f})")
ax.plot(ph["hour"], ph["r2"], color="#d62728", marker="s", markersize=3,
        linewidth=1.2, label=f"capnorm (cap / peak load) (R2={o['r2']:.3f}, RMSE={o['rmse']:.2f})")
ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("LOO R^2")
ax.set_xticks(range(0, 24, 3))
ax.set_title("LOO R^2: capshare vs capnorm specification")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower center", fontsize=7)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_capnorm_compare.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_capnorm_compare.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# B. Per-tech slope profiles (raw + standardized)
PALETTE = {"solar": "#fdb863", "onwind": "#5e3c99", "offwind": "#3690c0",
           "hydro_disp": "#0570b0", "hydro_ror": "#74a9cf", "biomass": "#33a02c",
           "nuclear": "#e41a1c", "gas": "#fb9a99", "hardcoal": "#525252",
           "lignite": "#8c510a", "oil": "#bf812d"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"].sort_values("hour")
    if sub.empty: continue
    a1.plot(sub["hour"], sub["coef"], color=PALETTE[t], linewidth=1.2,
            marker="o", markersize=2.8, label=t)
    a2.plot(sub["hour"], sub["coef"] * stds[t], color=PALETTE[t], linewidth=1.2,
            marker="o", markersize=2.8, label=t)
for ax, title, ylabel in [
    (a1, "Raw  (EUR/MWh per +1.0 cap/peak_load)", "$\\beta_t^{(h)}$"),
    (a2, "Standardized  (EUR/MWh per +1 std of cap/peak_load)",
     "$\\beta_t^{(h)} \\times \\sigma$"),
]:
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel("Hour of day"); ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=8)
handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=11, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=7)
fig.savefig(PLOTS / "predict_deviation_capnorm_tech_slopes.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_capnorm_tech_slopes.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Pooled-FE regression: capnorm (cap / peak_load) vs capshare baseline")
out.append("=" * 78)
out.append(f"\nPanel (capnorm spec): {len(panel)} rows; "
           f"years {panel['year'].min()}-{panel['year'].max()}")
out.append(f"Panel (capshare baseline, matched): {len(panel_old)} rows")

out.append("\nOverall LOO comparison:")
out.append(f"  capshare baseline : R2 = {o_old['r2']:.3f}, RMSE = {o_old['rmse']:.2f}")
out.append(f"  capnorm spec      : R2 = {o['r2']:.3f}, RMSE = {o['rmse']:.2f}")
out.append(f"  delta             : R2 {o['r2']-o_old['r2']:+.3f}  RMSE {o['rmse']-o_old['rmse']:+.2f}")

out.append("\nTech-slope coefficients on capnorm (full panel fit):")
out.append("  Per-hour, RAW units = EUR/MWh per +1.0 (cap/peak_load) -- comparable across techs")
out.append("  because all techs measured against the SAME denominator.")

out.append("\nPeak-trough range of raw capnorm coefficient (EUR/MWh per +1.0 cap/peak_load):")
for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"]
    if sub.empty: continue
    out.append(f"  {t:<12s}: max-min = {sub['coef'].max() - sub['coef'].min():>+7.1f}  "
               f"(max={sub['coef'].max():+.1f} at h{int(sub.loc[sub['coef'].idxmax(),'hour']):02d}, "
               f"min={sub['coef'].min():+.1f} at h{int(sub.loc[sub['coef'].idxmin(),'hour']):02d})")

out.append("\nStandardized peak-trough range  (EUR/MWh per 1 std of cap/peak_load):")
for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"]
    if sub.empty: continue
    std_coef = sub["coef"].values * stds[t]
    out.append(f"  {t:<12s}: max-min = {std_coef.max() - std_coef.min():>+7.2f}")

out.append("\nSolar per-hour coefficient (full panel):")
sol = coefs[coefs["predictor"] == "capnorm_solar"].sort_values("hour")
for _, r in sol.iterrows():
    out.append(f"  h{int(r['hour']):02d}: coef = {r['coef']:+8.2f}, t = {r['t']:+5.2f}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "predict_deviation_capnorm_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_capnorm_summary.txt'}")
