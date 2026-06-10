"""
Pooled-FE regression EXTENDED with neighbor-weighted capacity exposure.

For each (country c, year y, tech t):
  exposed_capshare_t(c, y) = sum_n NTC(c, n, y) * capshare_t(n, y)
                             / sum_n NTC(c, n, y)

  i.e. NTC-weighted average of neighbors' tech composition.

Model per hour h:
  dev(c, y, h) = alpha_c^(h)
              + sum_t  beta_t^(h)         * capshare_t(c, y)
              + sum_t  delta_t^(h)        * exposed_capshare_t(c, y)
              + gamma^(h)                 * ntc_share(c, y)
              + eps

The delta_t coefficients capture the structural shape "pressure" exerted on
country c by the technology mix it is interconnected to.

Compared against the same-sample baseline (capshares only).

Outputs:
  Analysis/results/system_state_panel_with_neighbors.csv
  Analysis/results/predict_deviation_pooled_with_neighbors_summary.txt
  Analysis/results/predict_deviation_pooled_with_neighbors_coefs.csv
  Analysis/results/forecast_de_2024_2025_with_neighbors.csv
  Analysis/results/plots/predict_deviation_with_neighbors_coefs.{pdf,png}
  Analysis/results/plots/forecast_de_2024_2025_with_neighbors.{pdf,png}
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
rng = np.random.default_rng(seed=42)

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 8, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE  = RESULTS / "hod_deviation_profiles_long.csv"
CAP_PANEL = RESULTS / "system_state_capacity_panel.csv"
NTC_RAW   = DATA / "ENTSOE_interconnection_per_border_2015_2025.csv"
NTC_AGG   = DATA / "ENTSOE_interconnection_capacity_2015_2025.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]
EXPOSED   = [f"exposed_{t}"  for t in DIFF_TECHS]
PREDICTORS = CAPSHARES + EXPOSED + ["ntc_share"]

AREA_TO_COUNTRY = {
    "AT": "AT", "BE": "BE", "BG": "BG", "CH": "CH", "CZ": "CZ",
    "DE_LU": "DE", "DK_1": "DK", "DK_2": "DK", "EE": "EE", "ES": "ES",
    "FI": "FI", "FR": "FR", "GB": "UK", "GR": "GR", "HR": "HR",
    "HU": "HU", "IE_SEM": "IE", "IT_NORD": "IT", "IT_BRNN": "IT",
    "LT": "LT", "LU": "LU", "LV": "LV", "NL": "NL", "NO_2": "NO",
    "NO_4": "NO", "PL": "PL", "PT": "PT", "RO": "RO", "RS": "RS",
    "SE_1": "SE", "SE_3": "SE", "SE_4": "SE", "SI": "SI", "SK": "SK",
}

B = 1000
TRAIN_END = 2023
TARGETS = [2024, 2025]


# ---------------------------------------------------------------------------
# 1. Build neighbor-weighted exposure features
# ---------------------------------------------------------------------------
print("Building neighbor-weighted exposure ...")
ntc_raw = pd.read_csv(NTC_RAW)
cap = pd.read_csv(CAP_PANEL)

# Map areas to country labels and remove self-loops
ntc_raw["c_from"] = ntc_raw["from"].map(AREA_TO_COUNTRY)
ntc_raw["c_to"]   = ntc_raw["to"].map(AREA_TO_COUNTRY)
ntc_raw = ntc_raw.dropna(subset=["c_from", "c_to"])
ntc_raw = ntc_raw[ntc_raw["c_from"] != ntc_raw["c_to"]]

# Build edge list: for each (c, y), list of (neighbor, ntc_mw)
edges = []
for _, r in ntc_raw.iterrows():
    edges.append({"home": r["c_from"], "neighbor": r["c_to"],
                  "year": int(r["year"]), "ntc_mw": float(r["ntc_mw"])})
    edges.append({"home": r["c_to"], "neighbor": r["c_from"],
                  "year": int(r["year"]), "ntc_mw": float(r["ntc_mw"])})
edges = pd.DataFrame(edges)
# Aggregate any duplicate (home, neighbor, year) entries by max
edges = edges.groupby(["home", "neighbor", "year"], as_index=False)["ntc_mw"].max()

# Lookup: capshare per (country, year)
cap_lookup = cap.set_index(["country", "year"])[CAPSHARES]

exposure_rows = []
for (home, year), grp in edges.groupby(["home", "year"]):
    total_w = grp["ntc_mw"].sum()
    if total_w <= 0:
        continue
    weighted = {f"exposed_{t}": 0.0 for t in DIFF_TECHS}
    weight_used = 0.0
    for _, e in grp.iterrows():
        try:
            n_caps = cap_lookup.loc[(e["neighbor"], year)]
        except KeyError:
            continue
        w = e["ntc_mw"]
        weight_used += w
        for t in DIFF_TECHS:
            weighted[f"exposed_{t}"] += w * float(n_caps[f"capshare_{t}"])
    if weight_used <= 0:
        continue
    weighted = {k: v / weight_used for k, v in weighted.items()}
    exposure_rows.append({
        "country": home, "year": int(year),
        "n_neighbors_used": int((grp["ntc_mw"] > 0).sum()),
        "total_neighbor_ntc": float(total_w),
        **weighted,
    })
exposure = pd.DataFrame(exposure_rows)
print(f"  -> {len(exposure)} (country, year) exposure rows for "
      f"{exposure['country'].nunique()} countries")
print("\n  Sample DE exposure (weighted neighbor capshares):")
print(exposure[exposure["country"] == "DE"][["year", "n_neighbors_used"] + EXPOSED]
      .round(3).to_string(index=False))


# Merge exposure + own caps + ntc_share -> full state panel
ntc_agg = pd.read_csv(NTC_AGG)
state = (cap.merge(ntc_agg[["country", "year", "total_ntc_mw"]],
                   on=["country", "year"], how="left")
            .merge(exposure, on=["country", "year"], how="left"))
state["ntc_share"] = state["total_ntc_mw"] / state["total_cap_mw"]
state.to_csv(RESULTS / "system_state_panel_with_neighbors.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Merge with deviation panel
# ---------------------------------------------------------------------------
dev = pd.read_csv(DEV_FILE)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .query("country in @COUNTRIES"))
panel_full = panel.dropna(subset=PREDICTORS + ["deviation"]).copy()
print(f"\nMerged panel (no-NaN on all predictors): {len(panel_full)} rows; "
      f"countries: {panel_full['country'].nunique()}; "
      f"years: {panel_full['year'].min()}-{panel_full['year'].max()}")


# ---------------------------------------------------------------------------
# 3. Fit / LOO helpers (predictors are parametric)
# ---------------------------------------------------------------------------
def make_design(df, train_countries, predictors):
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
        X = make_design(sub, countries, predictors)
        cols = list(X.columns)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": cols,
                  "residuals": (y - res.predict(X)).values,
                  "rsq": float(res.rsquared), "n": int(res.nobs),
                  "rmse": float(np.sqrt(res.mse_resid)),
                  "countries": countries}
    return out


def predict_for_row(model_h, country, feat_row, predictors):
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}
    x["const"] = 1.0
    if f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for s in predictors:
        x[s] = float(feat_row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


def extrapolate_features(panel_c, target_year, features):
    extr = {}
    for s in features:
        sub = panel_c.dropna(subset=[s])
        if len(sub) < 2:
            extr[s] = np.nan
            continue
        slope, intercept = np.polyfit(sub["year"].values, sub[s].values, 1)
        extr[s] = max(0.0, slope * target_year + intercept)
    return pd.Series(extr)


def loo_residuals(panel_in, predictors):
    rows = []
    for y0 in sorted(panel_in["year"].unique()):
        tr = panel_in[panel_in["year"] != y0]
        te = panel_in[panel_in["year"] == y0]
        m_loo = fit_per_hour(tr, predictors)
        for _, r in te.iterrows():
            h = int(r["hour"])
            if h not in m_loo or r["country"] not in m_loo[h]["countries"]:
                continue
            p = predict_for_row(m_loo[h], r["country"], r, predictors)
            rows.append({"country": r["country"], "year": int(y0),
                         "hour": h, "observed": float(r["deviation"]),
                         "predicted": p,
                         "residual": float(r["deviation"]) - p})
    return pd.DataFrame(rows)


def overall_metrics(loo):
    ss_res = (loo["residual"] ** 2).sum()
    ss_tot = ((loo["observed"] - loo["observed"].mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot,
            "rmse": float(np.sqrt((loo["residual"] ** 2).mean()))}


# ---------------------------------------------------------------------------
# 4. Compare three nested models on the SAME sample
# ---------------------------------------------------------------------------
print("\nFitting + LOO three nested models on matched sample ...")
models = {
    "M0_caps_only":         CAPSHARES,
    "M1_caps_ntc":          CAPSHARES + ["ntc_share"],
    "M2_caps_ntc_neighbor": CAPSHARES + ["ntc_share"] + EXPOSED,
}
loos = {}
m_full = {}
for name, preds in models.items():
    print(f"  {name} ({len(preds)} predictors) ...")
    m_full[name] = fit_per_hour(panel_full, preds)
    loos[name] = loo_residuals(panel_full, preds)

print("\nOverall LOO comparison (matched sample):")
for name in models:
    o = overall_metrics(loos[name])
    print(f"  {name:<24}: R² = {o['r2']:.3f}, RMSE = {o['rmse']:.2f}")


# ---------------------------------------------------------------------------
# 5. DE 2024/2025 holdout for the full model
# ---------------------------------------------------------------------------
print(f"\nHoldout: train <= {TRAIN_END}, forecast {TARGETS} for DE (M2 full model) ...")
preds_full = models["M2_caps_ntc_neighbor"]
panel_train = panel_full[panel_full["year"] <= TRAIN_END]
m_hold = fit_per_hour(panel_train, preds_full)
loo_train = loo_residuals(panel_train, preds_full)
loo_pool = {h: g["residual"].values for h, g in loo_train.groupby("hour")}

de_train = (panel_train[panel_train["country"] == "DE"]
            [["year"] + preds_full].drop_duplicates())
de_obs = (panel_full[panel_full["country"] == "DE"]
          [["year", "hour", "deviation"]])

fc_rows = []
for ty in TARGETS:
    extr = extrapolate_features(de_train, ty, preds_full)
    for h in range(24):
        if h not in m_hold or "DE" not in m_hold[h]["countries"]:
            continue
        point = predict_for_row(m_hold[h], "DE", extr, preds_full)
        sampled = rng.choice(loo_pool.get(h, m_hold[h]["residuals"]),
                             size=B, replace=True)
        boot = point + sampled
        obs = de_obs[(de_obs["year"] == ty) & (de_obs["hour"] == h)]
        obs_v = float(obs["deviation"].iloc[0]) if len(obs) else np.nan
        fc_rows.append({
            "year": ty, "hour": h, "point": point,
            "p05": float(np.percentile(boot, 5)),
            "p25": float(np.percentile(boot, 25)),
            "p50": float(np.percentile(boot, 50)),
            "p75": float(np.percentile(boot, 75)),
            "p95": float(np.percentile(boot, 95)),
            "deviation": obs_v,
        })
forecasts = pd.DataFrame(fc_rows)
forecasts.to_csv(RESULTS / "forecast_de_2024_2025_with_neighbors.csv", index=False)


def cov_rmse(d):
    m = d["deviation"].notna()
    if not m.any():
        return {"cov90": np.nan, "cov50": np.nan, "rmse": np.nan}
    return {
        "cov90": float(((d.loc[m, "deviation"] >= d.loc[m, "p05"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p95"])).mean()),
        "cov50": float(((d.loc[m, "deviation"] >= d.loc[m, "p25"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p75"])).mean()),
        "rmse":  float(np.sqrt(((d.loc[m, "point"] - d.loc[m, "deviation"]) ** 2).mean())),
    }


# ---------------------------------------------------------------------------
# 6. Coefficients for M2 (full model)
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m_full["M2_caps_ntc_neighbor"].items():
    for name, coef in zip(m["x_cols"], m["params"].values):
        coef_rows.append({"hour": h, "predictor": name, "coef": float(coef)})
coefs_df = pd.DataFrame(coef_rows)
coefs_df.to_csv(RESULTS / "predict_deviation_pooled_with_neighbors_coefs.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
# 7a. Per-hour LOO R^2 across the three models
fig, ax = plt.subplots(figsize=(6.67, 2.8))
for name, color in [("M0_caps_only", "#999999"),
                    ("M1_caps_ntc", "#1f77b4"),
                    ("M2_caps_ntc_neighbor", "#d62728")]:
    rows = []
    for h, g in loos[name].groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({"hour": int(h), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan})
    df = pd.DataFrame(rows).sort_values("hour")
    ax.plot(df["hour"], df["r2"], color=color, marker="o", markersize=3,
            linewidth=1.2, label=name)
ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("LOO R²")
ax.set_title("Per-hour LOO R²: nested model comparison")
ax.set_xticks(range(0, 24, 3))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower center", ncol=3, fontsize=7)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_with_neighbors_coefs.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_with_neighbors_coefs.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. DE holdout fans
fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
for ax, ty in zip(axes, TARGETS):
    d = forecasts[forecasts["year"] == ty].sort_values("hour")
    ax.fill_between(d["hour"], d["p05"], d["p95"], color="#d62728", alpha=0.18, linewidth=0, label="90% PI")
    ax.fill_between(d["hour"], d["p25"], d["p75"], color="#d62728", alpha=0.30, linewidth=0, label="50% PI")
    ax.plot(d["hour"], d["point"], color="#d62728", linewidth=1.4, marker="s", markersize=3, label="Point forecast")
    if d["deviation"].notna().any():
        ax.plot(d["hour"], d["deviation"], color="black", linewidth=1.4, marker="o", markersize=3, label="Observed")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    cm = cov_rmse(d)
    ax.set_title(f"DE {ty}  (RMSE {cm['rmse']:.1f}, cov90 {cm['cov90']*100:.0f}%)")
    ax.set_xlabel("Hour of day"); ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Deviation (EUR/MWh)")
axes[0].legend(frameon=False, loc="best")
fig.suptitle(f"Pooled-FE WITH neighbor-weighted exposure: DE 2024/2025 holdout (train <= {TRAIN_END})")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_neighbors.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_neighbors.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Pooled-FE WITH neighbor-weighted capacity exposure")
out.append("=" * 78)
out.append(f"\nMatched sample: {len(panel_full)} rows; {panel_full['country'].nunique()} countries; "
           f"years {panel_full['year'].min()}-{panel_full['year'].max()}")
out.append("\nModel comparison (all on the SAME observations):")
out.append(f"  {'model':<26}  {'k':>3}  {'R2_LOO':>8}  {'RMSE_LOO':>10}")
out.append("-" * 78)
for name in models:
    o = overall_metrics(loos[name])
    out.append(f"  {name:<26}  {len(models[name]):>3}  {o['r2']:>8.3f}  {o['rmse']:>10.2f}")

out.append("\nPer-hour LOO R2 (M0 vs M1 vs M2):")
out.append(f"{'h':>3}  {'R2_M0':>7}  {'R2_M1':>7}  {'R2_M2':>7}  "
           f"{'RMSE_M0':>8}  {'RMSE_M1':>8}  {'RMSE_M2':>8}")
out.append("-" * 78)
hh = {}
for name in models:
    rows = []
    for h, g in loos[name].groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({"hour": int(h),
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rmse": float(np.sqrt((g["residual"] ** 2).mean()))})
    hh[name] = pd.DataFrame(rows).set_index("hour").sort_index()
for h in sorted(hh["M0_caps_only"].index):
    r0 = hh["M0_caps_only"].loc[h]
    r1 = hh["M1_caps_ntc"].loc[h]
    r2 = hh["M2_caps_ntc_neighbor"].loc[h]
    out.append(f"{int(h):>3}  {r0['r2']:>7.3f}  {r1['r2']:>7.3f}  {r2['r2']:>7.3f}  "
               f"{r0['rmse']:>8.2f}  {r1['rmse']:>8.2f}  {r2['rmse']:>8.2f}")

# Top neighbor-exposure coefficients (mean abs across hours)
out.append("\nTop neighbor-exposure coefficients (M2, mean |coef| across hours):")
nb = (coefs_df[coefs_df["predictor"].str.startswith("exposed_")]
      .groupby("predictor")["coef"].apply(lambda s: s.abs().mean())
      .sort_values(ascending=False))
for p, v in nb.items():
    out.append(f"  {p:<22s}  mean |coef| = {v:.2f}")

out.append("\nDE 2024/2025 holdout (M2 full model):")
for ty in TARGETS:
    cm = cov_rmse(forecasts[forecasts["year"] == ty])
    out.append(f"  {ty}: RMSE {cm['rmse']:.2f}, 90% cov {cm['cov90']*100:.0f}%, 50% cov {cm['cov50']*100:.0f}%")

out.append("\nFor reference (from previous runs, baseline DE 2024/2025 holdout):")
out.append("  M0 baseline (caps only) DE 2024 RMSE: 10.54   2025 RMSE: 15.78")
out.append("  M1 (caps + ntc_share)   DE 2024 RMSE:  8.04   2025 RMSE: 12.71")

text = "\n".join(out)
print()
print(text)
(RESULTS / "predict_deviation_pooled_with_neighbors_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_pooled_with_neighbors_summary.txt'}")
