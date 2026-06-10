"""
Per-hour pooled OLS with country fixed effects on CAPACITY shares.

Model per hour h:
    dev(c, y, h) = alpha_c^(h) + sum_t beta_t^(h) * capshare_t(c, y) + eps

- alpha_c^(h)     : country fixed effect at hour h (absorbs level differences)
- beta_t^(h)      : universal tech slope at hour h, pooled across all countries
- capshare_t(c,y) : capacity share of tech t in country c, year y
                    (raw capacity / sum of all capacities for that country-year)
- omitted tech    : "other" (so the intercept set is identified)

Compared to the v2 model (generation shares + cmean per country):
- generation shares contained weather noise; capacity shares are weather-free.
- v2 fitted a per-(country, hour) cmean intercept; this model fits a per-country
  hour-specific intercept. Universal pooled slopes are a stronger structural
  claim: "across Europe, +1 pp solar capacity changes the deviation at hour h
  by beta_solar^(h) regardless of which country."

Pipeline (single script):
  1. Build capacity-share panel from ENTSOE_installed_capacity_2015_2025.csv
     with tech mapping aligned to the existing 11-tech (+ other) framework.
  2. Merge with hod_deviation_profiles_long.csv -> per-(country, year, hour) panel.
  3. Per-hour pooled OLS with country FE.
  4. LOO across years; holdout: train <= 2023, forecast 2024 and 2025 with PI.
  5. Plot coefficients + holdout fan vs observed.

Outputs:
  Analysis/results/system_state_capacity_panel.csv
  Analysis/results/predict_deviation_pooled_capacity_models.csv
  Analysis/results/predict_deviation_pooled_capacity_coefs.csv
  Analysis/results/predict_deviation_pooled_capacity_loo_predictions.csv
  Analysis/results/forecast_de_2024_2025_pooled_capacity.csv
  Analysis/results/predict_deviation_pooled_capacity_summary.txt
  Analysis/results/plots/predict_deviation_pooled_capacity_coefs.{pdf,png}
  Analysis/results/plots/forecast_de_2024_2025_pooled_capacity.{pdf,png}
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

CAP_FILE = DATA / "ENTSOE_installed_capacity_2015_2025.csv"
DEV_FILE = RESULTS / "hod_deviation_profiles_long.csv"

# 22 countries — same set as the analysis (IT/SE not in panel anyway).
COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]

# Map ENTSO-E carrier names -> framework tech buckets
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
    "other":      ["Geothermal", "Marine", "Other", "Other renewable",
                   "Waste", "Energy storage"],
}
TECH_NAMES = list(TECH_MAP.keys())
DIFF_TECHS = [t for t in TECH_NAMES if t != "other"]
SHARES = [f"capshare_{t}" for t in DIFF_TECHS]

B = 1000
TRAIN_END = 2023
TARGETS = [2024, 2025]


# ---------------------------------------------------------------------------
# 1. Build capacity-share panel
# ---------------------------------------------------------------------------
def build_capacity_panel() -> pd.DataFrame:
    raw = pd.read_csv(CAP_FILE)
    raw = raw[raw["country"].isin(COUNTRIES)].copy()
    rows = []
    for (c, y), grp in raw.groupby(["country", "year"]):
        r = grp.iloc[0]
        bucket_caps = {}
        for tech, src_cols in TECH_MAP.items():
            present = [s for s in src_cols if s in grp.columns]
            bucket_caps[tech] = float(r[present].sum()) if present else 0.0
        total = sum(bucket_caps.values())
        if total <= 0:
            continue
        out = {"country": c, "year": int(y), "total_cap_mw": total}
        for tech, cap in bucket_caps.items():
            out[f"cap_{tech}"]      = cap
            out[f"capshare_{tech}"] = cap / total
        rows.append(out)
    df = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
    return df


print("Building capacity-share panel ...")
state = build_capacity_panel()
state.to_csv(RESULTS / "system_state_capacity_panel.csv", index=False)
print(f"  -> {len(state)} (country, year) rows; "
      f"countries: {state['country'].nunique()}; "
      f"year span: {state['year'].min()}-{state['year'].max()}")


# ---------------------------------------------------------------------------
# 2. Merge with deviation panel
# ---------------------------------------------------------------------------
dev = pd.read_csv(DEV_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")
panel = panel[panel["country"].isin(COUNTRIES)]
print(f"\nMerged panel: {len(panel)} rows ({panel['country'].nunique()} countries, "
      f"years {panel['year'].min()}-{panel['year'].max()})")


# ---------------------------------------------------------------------------
# 3. Per-hour pooled OLS with country FE
# ---------------------------------------------------------------------------
def make_design(df: pd.DataFrame, train_countries: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build [const + country dummies (drop first) + SHARES] design.

    train_countries: list of countries the FE block should accommodate. When
    predicting for a country not in train, its FE row is the baseline (all 0).
    """
    countries = train_countries or sorted(df["country"].unique())
    base = countries[0]
    fe_cols = [f"FE_{c}" for c in countries[1:]]
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for c in countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in SHARES:
        X[s] = df[s].astype(float).values
    return X, ["const"] + fe_cols + SHARES


def fit_per_hour(train_df: pd.DataFrame) -> dict:
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = (train_df[train_df["hour"] == h]
               .dropna(subset=["deviation"] + SHARES))
        if len(sub) < len(SHARES) + len(countries) + 5:
            continue
        X, cols = make_design(sub, train_countries=countries)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {
            "params": res.params, "x_cols": cols,
            "residuals": (y - res.predict(X)).values,
            "rsq": float(res.rsquared), "n": int(res.nobs),
            "rmse": float(np.sqrt(res.mse_resid)),
            "countries": countries,
        }
    return out


def predict_for_row(model_h: dict, country: str, share_vec: pd.Series) -> float:
    countries = model_h["countries"]
    base = countries[0]
    x = {c: 0.0 for c in model_h["x_cols"]}
    x["const"] = 1.0
    if country != base and f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for s in SHARES:
        x[s] = float(share_vec[s])
    vec = np.array([x[c] for c in model_h["x_cols"]])
    return float(np.dot(vec, model_h["params"].values))


print("\nFitting per-hour pooled OLS with country FE on full panel ...")
m_full = fit_per_hour(panel)
print(f"  Hours fit: {len(m_full)}/24")
print(f"  Median in-sample R^2: {np.median([m['rsq'] for m in m_full.values()]):.3f}")


# ---------------------------------------------------------------------------
# 4. LOO across years
# ---------------------------------------------------------------------------
print("\nLOO across years ...")
loo_rows = []
for y0 in sorted(panel["year"].unique()):
    train = panel[panel["year"] != y0]
    test  = panel[panel["year"] == y0]
    m = fit_per_hour(train)
    for _, r in test.iterrows():
        h = int(r["hour"])
        if h not in m:
            continue
        if r["country"] not in m[h]["countries"]:
            continue
        pred = predict_for_row(m[h], r["country"], r[SHARES])
        loo_rows.append({
            "country": r["country"], "year": int(y0), "hour": h,
            "observed": float(r["deviation"]), "predicted": pred,
            "residual": float(r["deviation"]) - pred,
        })
loo = pd.DataFrame(loo_rows)
loo.to_csv(RESULTS / "predict_deviation_pooled_capacity_loo_predictions.csv", index=False)

# Per-hour LOO R^2
def loo_r2_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({
            "hour": int(h),
            "loo_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "loo_rmse": float(np.sqrt((g["residual"] ** 2).mean())),
            "n": len(g),
        })
    return pd.DataFrame(rows).sort_values("hour")


loo_h = loo_r2_per_hour(loo)
overall_ss_res = (loo["residual"] ** 2).sum()
overall_ss_tot = ((loo["observed"] - loo["observed"].mean()) ** 2).sum()
overall_r2 = 1 - overall_ss_res / overall_ss_tot
overall_rmse = float(np.sqrt((loo["residual"] ** 2).mean()))


# ---------------------------------------------------------------------------
# 5. Holdout: train <= 2023, forecast 2024 and 2025 for DE with PI
# ---------------------------------------------------------------------------
print(f"\nHoldout: train years <= {TRAIN_END}, forecast {TARGETS} for DE ...")
panel_train = panel[panel["year"] <= TRAIN_END]
m_holdout = fit_per_hour(panel_train)
print(f"  Train rows: {len(panel_train)}; hours fit: {len(m_holdout)}/24")


def extrapolate_capshare(panel_c: pd.DataFrame, target_year: int) -> pd.Series:
    extr = {}
    for s in SHARES:
        sub = panel_c.dropna(subset=[s])
        if len(sub) < 2:
            extr[s] = np.nan
            continue
        slope, intercept = np.polyfit(sub["year"].values, sub[s].values, 1)
        extr[s] = max(0.0, slope * target_year + intercept)
    s_sum = sum(extr.values())
    if s_sum > 1.0:
        extr = {k: v / s_sum for k, v in extr.items()}
    return pd.Series(extr)


de_train = (panel_train[panel_train["country"] == "DE"]
            [["year"] + SHARES].drop_duplicates())
de_obs   = (panel[panel["country"] == "DE"]
            [["year", "hour", "deviation"]])

# Build per-hour LOO residual pools (calibrated bootstrap source).
# Fitted residuals after country FE understate out-of-sample uncertainty;
# LOO residuals are the proper empirical distribution for a held-out year.
loo_resids_per_hour = {h: g["residual"].values for h, g in loo.groupby("hour")}

forecast_rows = []
for ty in TARGETS:
    extr = extrapolate_capshare(de_train, ty)
    for h in range(24):
        if h not in m_holdout:
            continue
        m = m_holdout[h]
        if "DE" not in m["countries"]:
            continue
        point = predict_for_row(m, "DE", extr)
        resid_pool = loo_resids_per_hour.get(h, m["residuals"])
        sampled = rng.choice(resid_pool, size=B, replace=True)
        boot = point + sampled
        obs = de_obs[(de_obs["year"] == ty) & (de_obs["hour"] == h)]
        obs_v = float(obs["deviation"].iloc[0]) if len(obs) else np.nan
        forecast_rows.append({
            "year": ty, "hour": h, "point": point,
            "p05": float(np.percentile(boot, 5)),
            "p25": float(np.percentile(boot, 25)),
            "p50": float(np.percentile(boot, 50)),
            "p75": float(np.percentile(boot, 75)),
            "p95": float(np.percentile(boot, 95)),
            "deviation": obs_v,
        })
forecasts = pd.DataFrame(forecast_rows)
forecasts.to_csv(RESULTS / "forecast_de_2024_2025_pooled_capacity.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Coefficients table (M_full = full-panel pooled fit)
# ---------------------------------------------------------------------------
coef_rows = []
for h, m in m_full.items():
    for name, coef in zip(m["x_cols"], m["params"].values):
        coef_rows.append({"hour": h, "predictor": name, "coef": float(coef)})
coefs_df = pd.DataFrame(coef_rows)
coefs_df.to_csv(RESULTS / "predict_deviation_pooled_capacity_coefs.csv", index=False)

models_df = pd.DataFrame([
    {"hour": h, "n_in": m["n"], "in_sample_r2": m["rsq"], "in_sample_rmse": m["rmse"]}
    for h, m in m_full.items()
]).merge(loo_h.rename(columns={"n": "n_loo"}), on="hour", how="left")
models_df.to_csv(RESULTS / "predict_deviation_pooled_capacity_models.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
# 7a. Coefficient profiles by hour (one line per tech)
fig, ax = plt.subplots(figsize=(6.67, 3.0))
palette = {
    "solar": "#fdb863", "onwind": "#5e3c99", "offwind": "#3690c0",
    "hydro_disp": "#0570b0", "hydro_ror": "#74a9cf", "biomass": "#33a02c",
    "nuclear": "#e41a1c", "gas": "#fb9a99", "hardcoal": "#525252",
    "lignite": "#8c510a", "oil": "#bf812d",
}
for t in DIFF_TECHS:
    sub = coefs_df[coefs_df["predictor"] == f"capshare_{t}"].sort_values("hour")
    if sub.empty:
        continue
    ax.plot(sub["hour"], sub["coef"], color=palette[t], linewidth=1.2,
            marker="o", markersize=2.8, label=t)
ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Coefficient (€/MWh per +1.0 capshare)")
ax.set_title("Pooled tech-slope profiles (full panel; country FE absorbed)")
ax.set_xticks(range(0, 24, 3))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_pooled_capacity_coefs.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_pooled_capacity_coefs.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 7b. Holdout fan for DE 2024 and 2025
def coverage(d: pd.DataFrame, lo: str, hi: str) -> float:
    m = d["deviation"].notna()
    if not m.any():
        return float("nan")
    return float(((d.loc[m, "deviation"] >= d.loc[m, lo]) &
                  (d.loc[m, "deviation"] <= d.loc[m, hi])).mean())


def rmse(d: pd.DataFrame) -> float:
    m = d["deviation"].notna()
    return float(np.sqrt(((d.loc[m, "point"] - d.loc[m, "deviation"]) ** 2).mean()))


fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 3.0),
                         sharey=True, constrained_layout=True)
for ax, ty in zip(axes, TARGETS):
    d = forecasts[forecasts["year"] == ty].sort_values("hour")
    ax.fill_between(d["hour"], d["p05"], d["p95"], color="#d62728", alpha=0.18, linewidth=0, label="90% PI")
    ax.fill_between(d["hour"], d["p25"], d["p75"], color="#d62728", alpha=0.30, linewidth=0, label="50% PI")
    ax.plot(d["hour"], d["point"], color="#d62728", linewidth=1.4, marker="s", markersize=3, label="Point forecast")
    if d["deviation"].notna().any():
        ax.plot(d["hour"], d["deviation"], color="black", linewidth=1.4, marker="o", markersize=3, label="Observed")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    cov90 = coverage(d, "p05", "p95"); cov50 = coverage(d, "p25", "p75"); rm = rmse(d)
    ax.set_title(f"DE {ty}  (90% cov = {cov90*100:.0f}%, 50% cov = {cov50*100:.0f}%, "
                 f"RMSE = {rm:.1f} €/MWh)")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Deviation from annual mean (€/MWh)")
axes[0].legend(frameon=False, loc="best", handlelength=1.5)
fig.suptitle(f"Pooled-FE capacity-share model: holdout forecast (train 2015-{TRAIN_END})")
fig.savefig(PLOTS / "forecast_de_2024_2025_pooled_capacity.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "forecast_de_2024_2025_pooled_capacity.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Pooled per-hour OLS with country FE on capacity shares")
out.append("=" * 78)
out.append(f"\nPanel: {len(panel)} rows, {panel['country'].nunique()} countries, "
           f"{panel['year'].min()}-{panel['year'].max()}")
out.append(f"Predictors: {len(SHARES)} capacity shares + "
           f"{panel['country'].nunique()-1} country FE dummies + const")
out.append(f"\nOverall LOO R²:  {overall_r2:.3f}    Overall LOO RMSE: {overall_rmse:.2f} €/MWh")
out.append(f"In-sample R² range: "
           f"{min(m['rsq'] for m in m_full.values()):.3f} ... "
           f"{max(m['rsq'] for m in m_full.values()):.3f}")
out.append("")
out.append("Per-hour fit:")
out.append(f"{'h':>3}  {'n':>4}  {'in-R²':>7}  {'in-RMSE':>8}  {'LOO-R²':>7}  {'LOO-RMSE':>9}")
out.append("-" * 78)
for _, r in models_df.sort_values("hour").iterrows():
    out.append(f"{int(r['hour']):>3}  {int(r['n_in']):>4}  {r['in_sample_r2']:>7.3f}  "
               f"{r['in_sample_rmse']:>8.2f}  {r['loo_r2']:>7.3f}  {r['loo_rmse']:>9.2f}")

# Top tech slopes (mean abs coef across hours)
out.append("\nTop tech slopes (mean |coef| across hours):")
mean_abs = (coefs_df[coefs_df["predictor"].str.startswith("capshare_")]
            .groupby("predictor")["coef"].apply(lambda s: s.abs().mean())
            .sort_values(ascending=False))
for p, v in mean_abs.items():
    out.append(f"  {p:<24s}: mean |coef| = {v:.2f}")

out.append(f"\nDE 2024/2025 holdout (train years <= {TRAIN_END}):")
for ty in TARGETS:
    d = forecasts[forecasts["year"] == ty]
    out.append(f"  {ty}: 90% PI cov = {coverage(d,'p05','p95')*100:.0f}%, "
               f"50% PI cov = {coverage(d,'p25','p75')*100:.0f}%, "
               f"point RMSE = {rmse(d):.2f} €/MWh")

text = "\n".join(out)
print(text)
(RESULTS / "predict_deviation_pooled_capacity_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_pooled_capacity_summary.txt'}")
