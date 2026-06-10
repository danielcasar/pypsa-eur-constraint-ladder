"""
Pooled-FE capacity-share regression EXTENDED with interconnection share.

Model per hour h:
    dev(c, y, h) = alpha_c^(h)
                 + sum_t  beta_t^(h)    * capshare_t(c, y)
                 + gamma^(h)            * ntc_share(c, y)
                 + eps

  ntc_share(c, y) = total_NTC_MW(c, y) / total_installed_capacity_MW(c, y)

Interpretation of gamma:
  +1.0 in ntc_share means "as much interconnection capacity as domestic
  installed generation capacity". Expect gamma_h to be POSITIVE at trough
  hours (interconnection lifts low prices toward neighbours) and NEGATIVE
  at peak hours (interconnection caps high prices) -> flatter shape with
  more interconnection.

Compared against the no-NTC baseline (predict_deviation_pooled_capacity.py).

Outputs:
  Analysis/results/system_state_capacity_panel_with_ntc.csv
  Analysis/results/predict_deviation_pooled_with_ntc_summary.txt
  Analysis/results/predict_deviation_pooled_with_ntc_coefs.csv
  Analysis/results/forecast_de_2024_2025_with_ntc.csv
  Analysis/results/plots/predict_deviation_with_ntc_coefs.{pdf,png}
  Analysis/results/plots/forecast_de_2024_2025_with_ntc.{pdf,png}
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

DEV_FILE = RESULTS / "hod_deviation_profiles_long.csv"
CAP_PANEL_BASE = RESULTS / "system_state_capacity_panel.csv"
NTC_FILE = DATA / "ENTSOE_interconnection_capacity_2015_2025.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]
PREDICTORS = CAPSHARES + ["ntc_share"]

B = 1000
TRAIN_END = 2023
TARGETS = [2024, 2025]


# ---------------------------------------------------------------------------
# Build augmented panel
# ---------------------------------------------------------------------------
cap = pd.read_csv(CAP_PANEL_BASE)        # already has total_cap_mw and capshare_*
ntc = pd.read_csv(NTC_FILE)              # country, year, total_ntc_mw, n_borders
state = cap.merge(ntc[["country", "year", "total_ntc_mw", "n_borders"]],
                  on=["country", "year"], how="left")
state["ntc_share"] = state["total_ntc_mw"] / state["total_cap_mw"]
state.to_csv(RESULTS / "system_state_capacity_panel_with_ntc.csv", index=False)
n_with_ntc = state["ntc_share"].notna().sum()
print(f"State panel: {len(state)} rows; with ntc_share: {n_with_ntc}/{len(state)}")
print("Sample ntc_share values:")
print(state[state["country"].isin(["DE", "FR", "IE", "NL", "BE", "RO"])]
      [["country", "year", "total_cap_mw", "total_ntc_mw", "ntc_share"]]
      .dropna(subset=["ntc_share"]).head(20).to_string())

dev = pd.read_csv(DEV_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")
panel = panel[panel["country"].isin(COUNTRIES)]
# Drop rows missing any predictor
panel_full = panel.dropna(subset=PREDICTORS + ["deviation"]).copy()
print(f"\nMerged panel (no-NaN on all predictors): {len(panel_full)} rows; "
      f"countries: {panel_full['country'].nunique()}; "
      f"years: {panel_full['year'].min()}-{panel_full['year'].max()}")


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------
def make_design(df: pd.DataFrame, train_countries: list[str],
                predictors: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(train_df: pd.DataFrame, predictors: list[str]) -> dict:
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


def predict_for_row(model_h: dict, country: str,
                    feat_row: pd.Series, predictors: list[str]) -> float:
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}
    x["const"] = 1.0
    if f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for s in predictors:
        x[s] = float(feat_row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


def extrapolate_features(panel_c: pd.DataFrame, target_year: int,
                         features: list[str]) -> pd.Series:
    extr = {}
    for s in features:
        sub = panel_c.dropna(subset=[s])
        if len(sub) < 2:
            extr[s] = np.nan
            continue
        slope, intercept = np.polyfit(sub["year"].values, sub[s].values, 1)
        v = slope * target_year + intercept
        # cap shares constrained to >=0; ntc_share also >=0
        extr[s] = max(0.0, v)
    return pd.Series(extr)


def loo_resids(panel_in: pd.DataFrame, predictors: list[str]) -> dict:
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


# ---------------------------------------------------------------------------
# Fit BASELINE (no NTC) and WITH NTC, in-sample + LOO
# ---------------------------------------------------------------------------
print("\nFitting baseline (no NTC) on the SAME observations (matched sample) ...")
m_base = fit_per_hour(panel_full, CAPSHARES)
print("Fitting with NTC ...")
m_full = fit_per_hour(panel_full, PREDICTORS)

print("\nLOO across years (baseline) ...")
loo_base = loo_resids(panel_full, CAPSHARES)
print("LOO across years (with NTC) ...")
loo_with = loo_resids(panel_full, PREDICTORS)


def overall_metrics(loo: pd.DataFrame) -> dict:
    ss_res = (loo["residual"] ** 2).sum()
    ss_tot = ((loo["observed"] - loo["observed"].mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot, "rmse": float(np.sqrt((loo["residual"] ** 2).mean()))}


b = overall_metrics(loo_base); w = overall_metrics(loo_with)
print(f"\nOverall LOO R^2  baseline: {b['r2']:.3f}    with NTC: {w['r2']:.3f}    delta: {w['r2']-b['r2']:+.3f}")
print(f"Overall LOO RMSE baseline: {b['rmse']:.2f}    with NTC: {w['rmse']:.2f}    delta: {w['rmse']-b['rmse']:+.2f}")


# Per-hour LOO R^2 comparison
def per_hour_r2(loo: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in loo.groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        out.append({"hour": int(h),
                    "loo_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                    "loo_rmse": float(np.sqrt((g["residual"] ** 2).mean()))})
    return pd.DataFrame(out).sort_values("hour")


h_base = per_hour_r2(loo_base).set_index("hour")
h_with = per_hour_r2(loo_with).set_index("hour")
compare = h_base.join(h_with, lsuffix="_base", rsuffix="_with")
compare["delta_r2"] = compare["loo_r2_with"] - compare["loo_r2_base"]
compare["delta_rmse"] = compare["loo_rmse_with"] - compare["loo_rmse_base"]


# ---------------------------------------------------------------------------
# Coefficients from with-NTC fit
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m_full.items():
    for name, coef in zip(m["x_cols"], m["params"].values):
        coef_rows.append({"hour": h, "predictor": name, "coef": float(coef)})
coefs_df = pd.DataFrame(coef_rows)
coefs_df.to_csv(RESULTS / "predict_deviation_pooled_with_ntc_coefs.csv", index=False)

# NTC coefficient pattern across hours
ntc_coefs = coefs_df[coefs_df["predictor"] == "ntc_share"].sort_values("hour")


# ---------------------------------------------------------------------------
# Holdout DE 2024 and 2025 (train <= 2023)
# ---------------------------------------------------------------------------
print(f"\nHoldout: train <= {TRAIN_END}, forecast {TARGETS} for DE (with NTC) ...")
panel_train = panel_full[panel_full["year"] <= TRAIN_END]
m_hold = fit_per_hour(panel_train, PREDICTORS)
loo_train = loo_resids(panel_train, PREDICTORS)
loo_resids_per_hour = {h: g["residual"].values for h, g in loo_train.groupby("hour")}

de_train = (panel_train[panel_train["country"] == "DE"]
            [["year"] + PREDICTORS].drop_duplicates())
de_obs = (panel_full[panel_full["country"] == "DE"]
          [["year", "hour", "deviation"]])

fc_rows = []
for ty in TARGETS:
    extr = extrapolate_features(de_train, ty, PREDICTORS)
    for h in range(24):
        if h not in m_hold or "DE" not in m_hold[h]["countries"]:
            continue
        point = predict_for_row(m_hold[h], "DE", extr, PREDICTORS)
        sampled = rng.choice(loo_resids_per_hour.get(h, m_hold[h]["residuals"]),
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
forecasts.to_csv(RESULTS / "forecast_de_2024_2025_with_ntc.csv", index=False)


def cov_rmse(d: pd.DataFrame) -> dict:
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
# Plot 1: NTC coefficient per hour (does it have the expected shape?)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 2.6))
ax.bar(ntc_coefs["hour"], ntc_coefs["coef"], color="#1f77b4", width=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Coefficient of ntc_share (€/MWh per 1.0)")
ax.set_title("Pooled coefficient on interconnection share (full panel)")
ax.set_xticks(range(0, 24, 3))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_with_ntc_coefs.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_with_ntc_coefs.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Plot 2: DE 2024 + 2025 holdout fans with NTC
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
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Deviation (€/MWh)")
axes[0].legend(frameon=False, loc="best")
fig.suptitle(f"Pooled-FE WITH interconnection share: DE 2024/2025 holdout (train <= {TRAIN_END})")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_ntc.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "forecast_de_2024_2025_with_ntc.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Pooled-FE with interconnection share (ntc_share = total_ntc / total_cap)")
out.append("=" * 78)
out.append(f"\nMatched sample (no NaN on all predictors): {len(panel_full)} rows; "
           f"{panel_full['country'].nunique()} countries; years {panel_full['year'].min()}-{panel_full['year'].max()}")
out.append("\nOverall LOO comparison (same sample, baseline vs with-NTC):")
out.append(f"  Baseline (11 caps + FE)  : LOO R2 = {b['r2']:.3f}, RMSE = {b['rmse']:.2f}")
out.append(f"  With NTC (11 caps + ntc + FE): LOO R2 = {w['r2']:.3f}, RMSE = {w['rmse']:.2f}")
out.append(f"  d R2 = {w['r2']-b['r2']:+.3f}    d RMSE = {w['rmse']-b['rmse']:+.2f} €/MWh")

out.append("\nPer-hour LOO R2 (baseline -> with-NTC):")
out.append(f"{'h':>3}  {'R2_base':>8}  {'R2_ntc':>8}  {'dR2':>7}  "
           f"{'RMSE_base':>9}  {'RMSE_ntc':>9}  {'dRMSE':>7}")
out.append("-" * 78)
for h, r in compare.iterrows():
    out.append(f"{int(h):>3}  {r['loo_r2_base']:>8.3f}  {r['loo_r2_with']:>8.3f}  "
               f"{r['delta_r2']:>+7.3f}  {r['loo_rmse_base']:>9.2f}  "
               f"{r['loo_rmse_with']:>9.2f}  {r['delta_rmse']:>+7.2f}")

out.append("\nNTC share coefficient per hour (pooled across all countries):")
out.append(f"{'h':>3}  {'g_ntc (€/MWh per +1.0 ntc_share)':>35}")
out.append("-" * 78)
for _, r in ntc_coefs.iterrows():
    out.append(f"{int(r['hour']):>3}  {r['coef']:>+35.2f}")

out.append("\nDE 2024/2025 holdout (with NTC):")
for ty in TARGETS:
    cm = cov_rmse(forecasts[forecasts["year"] == ty])
    out.append(f"  {ty}: RMSE {cm['rmse']:.2f} €/MWh, 90% cov {cm['cov90']*100:.0f}%, "
               f"50% cov {cm['cov50']*100:.0f}%")

text = "\n".join(out)
print()
print(text)
(RESULTS / "predict_deviation_pooled_with_ntc_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_pooled_with_ntc_summary.txt'}")
