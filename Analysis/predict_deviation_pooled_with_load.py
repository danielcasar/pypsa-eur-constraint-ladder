"""
Pooled-FE regression EXTENDED with hourly load share.

For each per-hour OLS at hour h, add ONE additional predictor:
  load_share_h(c, y) = mean(load(c, y, hour==h)) / sum_h mean(load(c, y, hour==h))

So load_share_h is the share of the average daily load that falls at hour h.
Sums to 1.0 across hours within each (country, year). Within a per-hour
regression at hour h, this is a single scalar predictor per (c, y).

Model per hour h:
  dev(c,y,h) = alpha_c^(h)
            + sum_t  beta_t^(h)   * capshare_t(c,y)         (own structure)
            + sum_t  delta_t^(h)  * exposed_t(c,y)          (neighbour structure)
            + gamma^(h)           * ntc_share(c,y)          (interconnection openness)
            + theta^(h)           * load_share_h(c,y)       (NEW: demand shape at this hour)
            + eps

Compares M2 (no load) vs M3 (with load) on the same matched sample.

Outputs:
  Analysis/results/load_share_panel.csv
  Analysis/results/predict_deviation_pooled_with_load_summary.txt
  Analysis/results/predict_deviation_pooled_with_load_coefs.csv
  Analysis/results/forecast_de_2024_2025_with_load.csv
  Analysis/results/plots/predict_deviation_with_load_compare_r2.{pdf,png}
  Analysis/results/plots/forecast_de_2024_2025_with_load.{pdf,png}
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

DEV_FILE   = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_panel_with_neighbors.csv"
LOAD_FILE  = DATA / "ENTSOE_load_2015_2025.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]
EXPOSED   = [f"exposed_{t}"  for t in DIFF_TECHS]
PREDICTORS_M2 = CAPSHARES + EXPOSED + ["ntc_share"]
PREDICTORS_M3 = PREDICTORS_M2 + ["load_share_h"]

B = 1000
TRAIN_END = 2023
TARGETS = [2024, 2025]


# ---------------------------------------------------------------------------
# 1. Build hourly load shares per (country, year, hour)
# ---------------------------------------------------------------------------
print("Building load-share panel ...")
load_raw = pd.read_csv(LOAD_FILE, index_col="timestamp", parse_dates=True)
load_raw.index = pd.to_datetime(load_raw.index, utc=True)
load_raw["year"] = load_raw.index.year
load_raw["hour"] = load_raw.index.hour

# Long format: (country, year, hour) -> mean load
ls_rows = []
for cc in [c for c in COUNTRIES if c in load_raw.columns]:
    sub = load_raw[[cc, "year", "hour"]].dropna()
    g = sub.groupby(["year", "hour"])[cc].mean().reset_index()
    g.columns = ["year", "hour", "load_mw"]
    g["country"] = cc
    ls_rows.append(g[["country", "year", "hour", "load_mw"]])
load_long = pd.concat(ls_rows, ignore_index=True)

# Convert to within-day shares: share_h = load_h / sum_h load_h
def to_share(grp):
    s = grp["load_mw"].sum()
    grp = grp.copy()
    grp["load_share_h"] = grp["load_mw"] / s if s > 0 else np.nan
    return grp

load_long = (load_long.groupby(["country", "year"], group_keys=False)
                       .apply(to_share))
load_long = load_long.dropna(subset=["load_share_h"])
load_long.to_csv(RESULTS / "load_share_panel.csv", index=False)
print(f"  -> {len(load_long)} (country, year, hour) load-share rows; "
      f"{load_long['country'].nunique()} countries; "
      f"years {load_long['year'].min()}-{load_long['year'].max()}")

# Sanity: do they sum to 1 per (c, y)?
sums = load_long.groupby(["country", "year"])["load_share_h"].sum()
print(f"  sum-check: min={sums.min():.4f}  max={sums.max():.4f}  median={sums.median():.4f}")


# ---------------------------------------------------------------------------
# 2. Merge full panel
# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)            # has capshares + exposed + ntc_share
dev = pd.read_csv(DEV_FILE)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .merge(load_long[["country", "year", "hour", "load_share_h"]],
                   on=["country", "year", "hour"], how="left")
            .query("country in @COUNTRIES"))
panel_full = panel.dropna(subset=PREDICTORS_M3 + ["deviation"]).copy()
print(f"\nMerged panel (no-NaN on all M3 predictors): {len(panel_full)} rows; "
      f"{panel_full['country'].nunique()} countries; "
      f"years {panel_full['year'].min()}-{panel_full['year'].max()}")


# ---------------------------------------------------------------------------
# 3. Fit / LOO helpers
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
# 4. Compare M2 vs M3 (matched sample)
# ---------------------------------------------------------------------------
print("\nFitting + LOO M2 (no load) ...")
m2_full = fit_per_hour(panel_full, PREDICTORS_M2)
loo_m2 = loo_residuals(panel_full, PREDICTORS_M2)

print("Fitting + LOO M3 (with load_share_h) ...")
m3_full = fit_per_hour(panel_full, PREDICTORS_M3)
loo_m3 = loo_residuals(panel_full, PREDICTORS_M3)

ov2 = overall_metrics(loo_m2); ov3 = overall_metrics(loo_m3)
print(f"\nOverall LOO  M2: R2={ov2['r2']:.3f}  RMSE={ov2['rmse']:.2f}")
print(f"Overall LOO  M3: R2={ov3['r2']:.3f}  RMSE={ov3['rmse']:.2f}")
print(f"           delta: R2 {ov3['r2']-ov2['r2']:+.3f}  RMSE {ov3['rmse']-ov2['rmse']:+.2f}")

# Per-hour R^2
def per_hour_r2(loo):
    rows = []
    for h, g in loo.groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({"hour": int(h),
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rmse": float(np.sqrt((g["residual"] ** 2).mean()))})
    return pd.DataFrame(rows).sort_values("hour")

h2 = per_hour_r2(loo_m2).set_index("hour")
h3 = per_hour_r2(loo_m3).set_index("hour")
compare = h2.join(h3, lsuffix="_M2", rsuffix="_M3")
compare["dr2"] = compare["r2_M3"] - compare["r2_M2"]
compare["drmse"] = compare["rmse_M3"] - compare["rmse_M2"]


# ---------------------------------------------------------------------------
# 5. Theta coefficient on load_share_h per hour
# ---------------------------------------------------------------------------
theta_rows = []
for h, m in m3_full.items():
    for name, coef in zip(m["x_cols"], m["params"].values):
        if name == "load_share_h":
            theta_rows.append({"hour": h, "theta": float(coef)})
theta = pd.DataFrame(theta_rows).sort_values("hour")


# ---------------------------------------------------------------------------
# 6. DE 2024/2025 holdout for M3
# ---------------------------------------------------------------------------
print(f"\nHoldout: train <= {TRAIN_END}, forecast {TARGETS} for DE (M3) ...")
panel_train = panel_full[panel_full["year"] <= TRAIN_END]
m_hold = fit_per_hour(panel_train, PREDICTORS_M3)
loo_train = loo_residuals(panel_train, PREDICTORS_M3)
loo_pool = {h: g["residual"].values for h, g in loo_train.groupby("hour")}

# DE per-hour predictors history (one row per (year, hour))
de_train = (panel_train[panel_train["country"] == "DE"]
            [["year", "hour"] + PREDICTORS_M3])
de_obs = (panel_full[panel_full["country"] == "DE"]
          [["year", "hour", "deviation"]])

# Extrapolate per (hour, target_year): for each predictor, fit a line over years
fc_rows = []
for ty in TARGETS:
    for h in range(24):
        if h not in m_hold or "DE" not in m_hold[h]["countries"]:
            continue
        sub_h = de_train[de_train["hour"] == h]
        extr = {}
        for s in PREDICTORS_M3:
            line = sub_h.dropna(subset=[s])
            if len(line) < 2:
                extr[s] = np.nan; continue
            slope, intercept = np.polyfit(line["year"].values, line[s].values, 1)
            extr[s] = max(0.0, slope * ty + intercept)
        feat = pd.Series(extr)
        if feat.isna().any():
            continue
        point = predict_for_row(m_hold[h], "DE", feat, PREDICTORS_M3)
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
forecasts.to_csv(RESULTS / "forecast_de_2024_2025_with_load.csv", index=False)


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
# 7. All coefficients table for M3
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m3_full.items():
    for name, coef in zip(m["x_cols"], m["params"].values):
        coef_rows.append({"hour": h, "predictor": name, "coef": float(coef)})
pd.DataFrame(coef_rows).to_csv(RESULTS / "predict_deviation_pooled_with_load_coefs.csv", index=False)


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
# 8a. Per-hour LOO R^2 + theta (two panels side-by-side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)
ax1.plot(h2.index, h2["r2"], color="#999999", marker="o", markersize=3, linewidth=1.2, label="M2 (caps + neighbour + ntc)")
ax1.plot(h3.index, h3["r2"], color="#d62728", marker="s", markersize=3, linewidth=1.2, label="M3 (+ load_share)")
ax1.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax1.set_xlabel("Hour of day"); ax1.set_ylabel("LOO R2")
ax1.set_title("Per-hour LOO R2: M2 vs M3")
ax1.set_xticks(range(0, 24, 3))
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(frameon=False, loc="lower center", fontsize=7)
ax2.bar(theta["hour"], theta["theta"], color="#1f77b4", width=0.7)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Hour of day"); ax2.set_ylabel("theta_h (EUR/MWh per +1.0 share)")
ax2.set_title("Coefficient on load_share_h")
ax2.set_xticks(range(0, 24, 3))
ax2.spines[["top", "right"]].set_visible(False)
fig.savefig(PLOTS / "predict_deviation_with_load_compare_r2.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_with_load_compare_r2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8b. DE holdout fans
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
fig.suptitle(f"M3 (caps + neighbour + ntc + load_share): DE 2024/2025 holdout (train <= {TRAIN_END})")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_load.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_load.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Pooled-FE WITH hourly load share (M3 vs M2)")
out.append("=" * 78)
out.append(f"\nMatched sample: {len(panel_full)} rows; "
           f"{panel_full['country'].nunique()} countries; "
           f"years {panel_full['year'].min()}-{panel_full['year'].max()}")
out.append("\nOverall LOO comparison:")
out.append(f"  M2 (caps + neighbour + ntc):       R2 = {ov2['r2']:.3f}, RMSE = {ov2['rmse']:.2f}")
out.append(f"  M3 (+ load_share_h):                R2 = {ov3['r2']:.3f}, RMSE = {ov3['rmse']:.2f}")
out.append(f"  delta:                              R2 {ov3['r2']-ov2['r2']:+.3f}  RMSE {ov3['rmse']-ov2['rmse']:+.2f}")

out.append("\nPer-hour LOO R2 + theta_h on load share:")
out.append(f"{'h':>3}  {'R2_M2':>7}  {'R2_M3':>7}  {'dR2':>7}  {'theta_h':>10}")
out.append("-" * 78)
theta_lookup = theta.set_index("hour")["theta"].to_dict()
for h in sorted(compare.index):
    th = theta_lookup.get(h, np.nan)
    out.append(f"{int(h):>3}  {compare.loc[h,'r2_M2']:>7.3f}  {compare.loc[h,'r2_M3']:>7.3f}  "
               f"{compare.loc[h,'dr2']:>+7.3f}  {th:>+10.2f}")

out.append("\nDE 2024/2025 holdout (M3 with load):")
for ty in TARGETS:
    cm = cov_rmse(forecasts[forecasts["year"] == ty])
    out.append(f"  {ty}: RMSE {cm['rmse']:.2f}, 90% cov {cm['cov90']*100:.0f}%, 50% cov {cm['cov50']*100:.0f}%")

out.append("\nFor reference (previous models, DE 2024 / 2025 RMSE):")
out.append("  M0 (caps only):                       10.54  /  15.78")
out.append("  M1 (+ ntc_share):                      8.04  /  12.71")
out.append("  M2 (+ neighbour exposure):             6.55  /  10.59")

text = "\n".join(out)
print()
print(text)
(RESULTS / "predict_deviation_pooled_with_load_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_pooled_with_load_summary.txt'}")
