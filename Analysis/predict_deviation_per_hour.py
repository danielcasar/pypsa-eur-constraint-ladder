"""
For each hour-of-day h, fit a standard OLS that predicts the mean deviation
from per-(country, year) generation-share predictors only:

    dev(c, y, h) = a_h + Σ_t  beta_{t,h} · share_t(c, y) + ε

Predictors are 10 of the 11 carrier shares (we drop 'other' as the omitted
category to avoid perfect collinearity since shares sum to 1). No fuel prices,
no weather, no country FE — purely energy-system-structural variables.

The model is fit twice per hour:
  - on the full panel (in-sample R^2)
  - leave-one-year-out (out-of-sample R^2 — the validation we care about)

Outputs:
  Analysis/results/predict_deviation_per_hour_coefs.csv
  Analysis/results/predict_deviation_per_hour_models.csv
  Analysis/results/predict_deviation_per_hour_loo_predictions.csv
  Analysis/results/predict_deviation_per_hour_summary.txt
  Analysis/results/plots/predict_deviation_per_hour_DE_loo.{pdf,png}
  Analysis/results/plots/predict_deviation_per_hour_loo_scatter.{pdf,png}
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
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  8,
    "figure.titlesize": 8,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE   = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_panel_per_tech.csv"

# Same tech list as v2; drop "other" as omitted category (shares sum to 1).
TECH_NAMES = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil", "other"]
PREDICTORS = [f"share_{t}" for t in TECH_NAMES if t != "other"]


# ---------------------------------------------------------------------------
# Load + merge
# ---------------------------------------------------------------------------
dev   = pd.read_csv(DEV_FILE)
state = pd.read_csv(STATE_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")
print(f"Merged panel: {len(panel)} rows "
      f"({panel['country'].nunique()} countries x "
      f"{panel['year'].nunique()} years x 24 hours).")


def fit_ols(df: pd.DataFrame, predictors: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(df[predictors].astype(float))
    y = df["deviation"].astype(float)
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# 1. In-sample fit per hour
# ---------------------------------------------------------------------------
in_sample_rows = []
coefs_rows = []
in_sample_preds = []
for h in range(24):
    sub = panel[panel["hour"] == h].dropna(subset=["deviation"] + PREDICTORS).copy()
    if len(sub) < len(PREDICTORS) + 5:
        continue
    res = fit_ols(sub, PREDICTORS)
    in_sample_rows.append({
        "hour": h, "n": int(res.nobs),
        "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj),
        "rmse": float(np.sqrt(res.mse_resid)),
        "aic": float(res.aic), "bic": float(res.bic),
    })
    for name, coef, pval in zip(["const"] + PREDICTORS, res.params, res.pvalues):
        coefs_rows.append({
            "hour": h, "predictor": name,
            "coef": float(coef), "p_value": float(pval),
        })
    sub["predicted"] = res.predict(sm.add_constant(sub[PREDICTORS].astype(float)))
    sub["residual"]  = sub["deviation"] - sub["predicted"]
    in_sample_preds.append(sub[["country", "year", "hour",
                                "deviation", "predicted", "residual"]])

in_sample = pd.DataFrame(in_sample_rows)
coefs = pd.DataFrame(coefs_rows)
in_sample.to_csv(RESULTS / "predict_deviation_per_hour_models.csv", index=False)
coefs.to_csv(RESULTS / "predict_deviation_per_hour_coefs.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Leave-one-year-out cross-validation
#    For each year y0: train on years != y0, predict y0, collect (obs, pred).
# ---------------------------------------------------------------------------
loo_rows = []
all_years = sorted(panel["year"].unique())
for h in range(24):
    sub_h = panel[panel["hour"] == h].dropna(subset=["deviation"] + PREDICTORS).copy()
    for y0 in all_years:
        train = sub_h[sub_h["year"] != y0]
        test  = sub_h[sub_h["year"] == y0]
        if len(train) < len(PREDICTORS) + 5 or test.empty:
            continue
        res = fit_ols(train, PREDICTORS)
        pred = res.predict(sm.add_constant(test[PREDICTORS].astype(float)))
        for c, dev_obs, dev_pred in zip(test["country"], test["deviation"], pred):
            loo_rows.append({
                "country": c, "year": int(y0), "hour": h,
                "observed": float(dev_obs), "predicted": float(dev_pred),
                "residual": float(dev_obs - dev_pred),
            })

loo = pd.DataFrame(loo_rows)
loo.to_csv(RESULTS / "predict_deviation_per_hour_loo_predictions.csv", index=False)

# Per-hour LOO R^2
loo_r2 = []
for h in range(24):
    sub = loo[loo["hour"] == h]
    if sub.empty:
        continue
    ss_res = (sub["residual"] ** 2).sum()
    ss_tot = ((sub["observed"] - sub["observed"].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    loo_r2.append({
        "hour": h,
        "loo_r2": float(r2),
        "loo_rmse": float(np.sqrt((sub["residual"] ** 2).mean())),
        "n_predictions": int(len(sub)),
    })
loo_r2 = pd.DataFrame(loo_r2)

models_combined = in_sample.merge(loo_r2, on="hour", how="left")
models_combined.to_csv(RESULTS / "predict_deviation_per_hour_models.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Plot: DE LOO predicted vs observed profile, one panel per held-out year
# ---------------------------------------------------------------------------
de_loo = loo[loo["country"] == "DE"].sort_values(["year", "hour"])
years = sorted(de_loo["year"].unique())
NCOLS = 4
NROWS = int(np.ceil(len(years) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(6.67, 3.6),
                         sharex=True, sharey=True, constrained_layout=True)
axes = axes.flatten()
for i, yr in enumerate(years):
    ax = axes[i]
    sub = de_loo[de_loo["year"] == yr]
    ax.plot(sub["hour"], sub["observed"], color="black",
            linewidth=1.2, marker="o", markersize=2.5, label="observed")
    ax.plot(sub["hour"], sub["predicted"], color="#d62728",
            linewidth=1.2, marker="s", markersize=2.5,
            linestyle="--", label="LOO prediction")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_title(f"DE {yr}", fontsize=8, pad=2)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.spines[["top", "right"]].set_visible(False)

for j in range(len(years), len(axes)):
    axes[j].set_visible(False)

fig.supxlabel("Hour of day", fontsize=8, y=0.02)
fig.supylabel("Deviation from annual mean (€/MWh)", fontsize=8, x=0.005)
fig.suptitle("DE — leave-one-year-out predicted vs observed HOD deviation profile",
             fontsize=8, y=1.005)
axes[0].legend(loc="upper left", frameon=False, handlelength=1.5)
fig.savefig(PLOTS / "predict_deviation_per_hour_DE_loo.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_per_hour_DE_loo.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Plot: LOO scatter (predicted vs observed) for all (country, year, hour)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.scatter(loo["observed"], loo["predicted"], s=2, alpha=0.30,
           color="#1f77b4", edgecolor="none")
xlo, xhi = loo["observed"].min(), loo["observed"].max()
ax.plot([xlo, xhi], [xlo, xhi], color="black", linewidth=0.8,
        linestyle="--", label="y = x")
overall_r2 = (1 - ((loo["residual"] ** 2).sum() /
                   ((loo["observed"] - loo["observed"].mean()) ** 2).sum()))
ax.set_xlabel("Observed deviation (€/MWh)", fontsize=8)
ax.set_ylabel("LOO-predicted deviation (€/MWh)", fontsize=8)
ax.set_title(f"LOO out-of-sample fit  (R² = {overall_r2:.3f})", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.3, alpha=0.4)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_per_hour_loo_scatter.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_per_hour_loo_scatter.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Human summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Per-hour OLS:  dev(c,y,h) ~ tech-shares(c,y)  -- structural, no fuel/FE")
out.append("=" * 78)
out.append(f"\nN per hour: ~{int(in_sample['n'].mean())}; "
           f"predictors: {len(PREDICTORS)} (omitted category: 'other')")
out.append(f"Total observations across all hours: {len(panel.dropna(subset=PREDICTORS))}")
out.append(f"Overall LOO out-of-sample R²: {overall_r2:.3f}")
out.append("")
out.append(f"{'h':>3}  {'in-R²':>8}  {'in-RMSE':>9}  {'LOO-R²':>8}  {'LOO-RMSE':>10}")
out.append("-" * 78)
for _, r in models_combined.iterrows():
    out.append(
        f"{int(r['hour']):>3}  "
        f"{r['r2']:>8.3f}  {r['rmse']:>9.2f}  "
        f"{r['loo_r2']:>8.3f}  {r['loo_rmse']:>10.2f}"
    )
out.append("")
out.append("Top 5 most important coefficients (largest |coef|, average over hours):")
mean_abs = (coefs[coefs["predictor"] != "const"]
            .groupby("predictor")["coef"].apply(lambda s: s.abs().mean())
            .sort_values(ascending=False))
for p, v in mean_abs.head(5).items():
    out.append(f"  {p}:  mean |coef| across hours = {v:.2f}")

text = "\n".join(out)
print(text)
(RESULTS / "predict_deviation_per_hour_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'predict_deviation_per_hour_summary.txt'}")
print(f"Plots: {PLOTS / 'predict_deviation_per_hour_DE_loo.pdf'}")
print(f"       {PLOTS / 'predict_deviation_per_hour_loo_scatter.pdf'}")
