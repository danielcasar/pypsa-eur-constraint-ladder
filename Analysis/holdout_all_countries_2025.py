"""
1-year-ahead holdout for 2025: train pooled-FE capacity-share model on
2015-2024, forecast 2025 deviation profile for each country, compare to
observed.

Mirrors holdout_all_countries_2024.py with TARGET=2025 and TRAIN_END=2024.

Outputs:
  Analysis/results/holdout_all_countries_2025.csv
  Analysis/results/holdout_all_countries_2025_summary.txt
  Analysis/results/plots/holdout_all_countries_2025_panels.{pdf,png}
  Analysis/results/plots/holdout_all_countries_2025_rmse.{pdf,png}
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
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE  = RESULTS / "hod_deviation_profiles_long.csv"
CAP_PANEL = RESULTS / "system_state_capacity_panel.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
SHARES = [f"capshare_{t}" for t in DIFF_TECHS]

B = 1000
TRAIN_END = 2024
TARGET = 2025


# ---------------------------------------------------------------------------
state = pd.read_csv(CAP_PANEL)
dev = pd.read_csv(DEV_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")
panel = panel[panel["country"].isin(COUNTRIES)]
panel_train = panel[panel["year"] <= TRAIN_END]
print(f"Panel: {len(panel)} rows; train ({len(panel_train)} rows, "
      f"years <= {TRAIN_END}); target year {TARGET}")


def make_design(df: pd.DataFrame, train_countries: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in SHARES:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(train_df: pd.DataFrame) -> dict:
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = (train_df[train_df["hour"] == h]
               .dropna(subset=["deviation"] + SHARES))
        if len(sub) < len(SHARES) + len(countries) + 5:
            continue
        X = make_design(sub, countries)
        cols = list(X.columns)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": cols,
                  "residuals": (y - res.predict(X)).values,
                  "countries": countries}
    return out


def predict_for_row(model_h: dict, country: str, share_vec: pd.Series) -> float:
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}
    x["const"] = 1.0
    if f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for s in SHARES:
        x[s] = float(share_vec[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


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


print("Fitting pooled OLS on train panel ...")
m = fit_per_hour(panel_train)

print("LOO across train years ...")
loo_rows = []
for y0 in sorted(panel_train["year"].unique()):
    tr = panel_train[panel_train["year"] != y0]
    te = panel_train[panel_train["year"] == y0]
    m_loo = fit_per_hour(tr)
    for _, r in te.iterrows():
        h = int(r["hour"])
        if h not in m_loo or r["country"] not in m_loo[h]["countries"]:
            continue
        p = predict_for_row(m_loo[h], r["country"], r[SHARES])
        loo_rows.append({"hour": h, "residual": float(r["deviation"]) - p})
loo = pd.DataFrame(loo_rows)
loo_resids_per_hour = {h: g["residual"].values for h, g in loo.groupby("hour")}


print(f"\nForecasting {TARGET} for {len(COUNTRIES)} countries ...")
rows = []
for cc in COUNTRIES:
    pc_train = (panel_train[panel_train["country"] == cc]
                [["year"] + SHARES].drop_duplicates())
    if len(pc_train) < 2:
        continue
    extr = extrapolate_capshare(pc_train, TARGET)
    obs_country = panel[(panel["country"] == cc) & (panel["year"] == TARGET)]
    for h in range(24):
        if h not in m or cc not in m[h]["countries"]:
            continue
        point = predict_for_row(m[h], cc, extr)
        sampled = rng.choice(loo_resids_per_hour.get(h, m[h]["residuals"]),
                             size=B, replace=True)
        boot = point + sampled
        obs = obs_country[obs_country["hour"] == h]
        obs_v = float(obs["deviation"].iloc[0]) if len(obs) else np.nan
        rows.append({
            "country": cc, "hour": h, "point": point,
            "p05": float(np.percentile(boot, 5)),
            "p25": float(np.percentile(boot, 25)),
            "p50": float(np.percentile(boot, 50)),
            "p75": float(np.percentile(boot, 75)),
            "p95": float(np.percentile(boot, 95)),
            "deviation": obs_v,
        })
fc = pd.DataFrame(rows)
fc.to_csv(RESULTS / f"holdout_all_countries_{TARGET}.csv", index=False)


def metrics_per_country(d: pd.DataFrame) -> dict:
    m = d["deviation"].notna()
    if not m.any():
        return {"n_hours": 0, "rmse": np.nan, "mae": np.nan,
                "cov90": np.nan, "cov50": np.nan,
                "obs_range": np.nan, "obs_std": np.nan}
    err = d.loc[m, "point"] - d.loc[m, "deviation"]
    return {
        "n_hours": int(m.sum()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae":  float(err.abs().mean()),
        "cov90": float(((d.loc[m, "deviation"] >= d.loc[m, "p05"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p95"])).mean()),
        "cov50": float(((d.loc[m, "deviation"] >= d.loc[m, "p25"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p75"])).mean()),
        "obs_range": float(d.loc[m, "deviation"].max() - d.loc[m, "deviation"].min()),
        "obs_std":   float(d.loc[m, "deviation"].std()),
    }


met_rows = []
for cc, g in fc.groupby("country"):
    met_rows.append({"country": cc, **metrics_per_country(g.sort_values("hour"))})
met = pd.DataFrame(met_rows).sort_values("rmse")


# Plot 1: per-country RMSE bar chart
fig, ax = plt.subplots(figsize=(6.67, 3.0))
m_plot = met.dropna(subset=["rmse"]).sort_values("rmse")
ax.bar(range(len(m_plot)), m_plot["rmse"], color="#1f77b4", width=0.7)
ax.axhline(m_plot["rmse"].median(), color="#d62728", linewidth=0.8, linestyle="--",
           label=f"median = {m_plot['rmse'].median():.1f} €/MWh")
ax.set_xticks(range(len(m_plot)))
ax.set_xticklabels(m_plot["country"])
ax.set_ylabel("RMSE (€/MWh, deviation)")
ax.set_title(f"1-year-ahead holdout RMSE per country (train 2015-{TRAIN_END}, target {TARGET})")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / f"holdout_all_countries_{TARGET}_rmse.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / f"holdout_all_countries_{TARGET}_rmse.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Plot 2: per-country small multiples
ccs = sorted(fc["country"].unique())
NCOLS = 6
NROWS = int(np.ceil(len(ccs) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0, 1.4 * NROWS),
                         sharex=True, constrained_layout=True)
axes = axes.flatten()
for i, cc in enumerate(ccs):
    ax = axes[i]
    d = fc[fc["country"] == cc].sort_values("hour")
    ax.fill_between(d["hour"], d["p05"], d["p95"], color="#d62728", alpha=0.18, linewidth=0)
    ax.fill_between(d["hour"], d["p25"], d["p75"], color="#d62728", alpha=0.30, linewidth=0)
    ax.plot(d["hour"], d["point"], color="#d62728", linewidth=1.0, marker="s", markersize=1.8)
    if d["deviation"].notna().any():
        ax.plot(d["hour"], d["deviation"], color="black", linewidth=1.0, marker="o", markersize=1.8)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    r = metrics_per_country(d)
    rmse_str = f"{r['rmse']:.1f}" if not np.isnan(r['rmse']) else "n/a"
    cov_str  = f"{int(r['cov90']*100)}%" if not np.isnan(r['cov90']) else "n/a"
    ax.set_title(f"{cc}: RMSE {rmse_str}, cov {cov_str}", fontsize=7, pad=2)
    ax.set_xticks([0, 6, 12, 18])
    ax.spines[["top", "right"]].set_visible(False)
for j in range(len(ccs), len(axes)):
    axes[j].set_visible(False)
fig.supxlabel("Hour of day", fontsize=8, y=0.01)
fig.supylabel("Deviation (€/MWh)", fontsize=8, x=0.005)
fig.suptitle(f"1y-ahead holdout per country (train 2015-{TRAIN_END}, target {TARGET}, 90% PI)",
             fontsize=8, y=1.005)
fig.savefig(PLOTS / f"holdout_all_countries_{TARGET}_panels.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / f"holdout_all_countries_{TARGET}_panels.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Summary table
out = []
out.append("=" * 78)
out.append(f"1-year-ahead holdout, pooled-FE capacity-share model")
out.append(f"Train: 2015-{TRAIN_END}.  Target: {TARGET}.")
out.append("=" * 78)
out.append(f"\nMedian RMSE: {met['rmse'].median():.2f} €/MWh")
out.append(f"Mean   RMSE: {met['rmse'].mean():.2f} €/MWh")
out.append(f"Min / Max:   {met['rmse'].min():.2f}  /  {met['rmse'].max():.2f} €/MWh")
out.append(f"Median 90% PI cov: {met['cov90'].median()*100:.0f}%")
out.append(f"Median 50% PI cov: {met['cov50'].median()*100:.0f}%")
out.append("")
out.append(f"{'cc':<4} {'n':>3} {'RMSE':>7} {'MAE':>7} {'cov90':>7} {'cov50':>7} "
           f"{'obs_range':>10} {'obs_std':>9}  {'RMSE/std':>9}")
out.append("-" * 78)
for _, r in met.iterrows():
    rs = r["rmse"] / r["obs_std"] if r["obs_std"] > 0 else np.nan
    out.append(f"{r['country']:<4} {int(r['n_hours']):>3} {r['rmse']:>7.2f} "
               f"{r['mae']:>7.2f} {r['cov90']*100:>6.0f}% {r['cov50']*100:>6.0f}% "
               f"{r['obs_range']:>10.1f} {r['obs_std']:>9.2f}  {rs:>9.2f}")

text = "\n".join(out)
print()
print(text)
(RESULTS / f"holdout_all_countries_{TARGET}_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / f'holdout_all_countries_{TARGET}_summary.txt'}")
