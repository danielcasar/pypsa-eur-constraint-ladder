"""
Per-hour OLS to predict the HOD mean-deviation profile, v2.

Adds country-mean shares (cmean_t) as time-invariant country-structure
predictors, on top of the v1 per-(country, year) shares. Both predictor sets
remain energy-system-only (projectable).

Models per hour h:
  M_shares     dev_h ~ shares                              (v1)
  M_full       dev_h ~ shares + cmean                      (v2; adds country structure)

Both are evaluated in-sample and via leave-one-year-out (LOO).

Outputs:
  Analysis/results/predict_deviation_per_hour_v2_models.csv
  Analysis/results/predict_deviation_per_hour_v2_coefs.csv
  Analysis/results/predict_deviation_per_hour_v2_loo_predictions.csv
  Analysis/results/predict_deviation_per_hour_v2_summary.txt
  Analysis/results/plots/predict_deviation_per_hour_v2_DE_loo.{pdf,png}
  Analysis/results/plots/predict_deviation_per_hour_v2_loo_r2_by_hour.{pdf,png}
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

TECH_NAMES = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil", "other"]
DIFF_TECHS = [t for t in TECH_NAMES if t != "other"]
SHARES = [f"share_{t}" for t in DIFF_TECHS]
CMEAN  = [f"cmean_{t}"  for t in DIFF_TECHS]


# ---------------------------------------------------------------------------
# Load + merge + add cmean
# ---------------------------------------------------------------------------
dev   = pd.read_csv(DEV_FILE)
state = pd.read_csv(STATE_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")

# Country-mean shares (time-invariant per country)
cmean_table = (state.groupby("country")[SHARES].mean().reset_index()
               .rename(columns={f"share_{t}": f"cmean_{t}" for t in TECH_NAMES if f"share_{t}" in state.columns}))
panel = panel.merge(cmean_table, on="country", how="left")
print(f"Merged panel: {len(panel)} rows; "
      f"shares + cmean columns ready ({len(SHARES)} + {len(CMEAN)} predictors).")


def fit(df: pd.DataFrame, predictors: list[str]):
    X = sm.add_constant(df[predictors].astype(float))
    y = df["deviation"].astype(float)
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# Run two specs, both in-sample and LOO
# ---------------------------------------------------------------------------
def run_spec(name: str, predictors: list[str]):
    in_sample, coefs_rows, loo_rows = [], [], []
    all_years = sorted(panel["year"].unique())
    for h in range(24):
        sub_h = (panel[panel["hour"] == h]
                  .dropna(subset=["deviation"] + predictors).copy())
        if len(sub_h) < len(predictors) + 5:
            continue
        # in-sample
        res = fit(sub_h, predictors)
        in_sample.append({
            "spec": name, "hour": h, "n": int(res.nobs),
            "k": len(predictors),
            "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj),
            "rmse": float(np.sqrt(res.mse_resid)),
            "aic": float(res.aic),
        })
        for p, c, pv in zip(["const"] + predictors, res.params, res.pvalues):
            coefs_rows.append({"spec": name, "hour": h, "predictor": p,
                               "coef": float(c), "p_value": float(pv)})
        # LOO
        for y0 in all_years:
            train = sub_h[sub_h["year"] != y0]
            test  = sub_h[sub_h["year"] == y0]
            if len(train) < len(predictors) + 5 or test.empty:
                continue
            res_t = fit(train, predictors)
            pred = res_t.predict(sm.add_constant(test[predictors].astype(float)))
            for c, dev_obs, dev_pred in zip(test["country"], test["deviation"], pred):
                loo_rows.append({"spec": name, "country": c, "year": int(y0),
                                 "hour": h,
                                 "observed": float(dev_obs),
                                 "predicted": float(dev_pred),
                                 "residual": float(dev_obs - dev_pred)})
    return pd.DataFrame(in_sample), pd.DataFrame(coefs_rows), pd.DataFrame(loo_rows)


print("\nFitting M_shares ...")
ins_a, coefs_a, loo_a = run_spec("M_shares", SHARES)
print("Fitting M_full ...")
ins_b, coefs_b, loo_b = run_spec("M_full", SHARES + CMEAN)

ins   = pd.concat([ins_a, ins_b], ignore_index=True)
coefs = pd.concat([coefs_a, coefs_b], ignore_index=True)
loo   = pd.concat([loo_a, loo_b], ignore_index=True)


# Per-hour LOO R^2 per spec
def loo_r2(grp: pd.DataFrame) -> pd.Series:
    ss_res = (grp["residual"] ** 2).sum()
    ss_tot = ((grp["observed"] - grp["observed"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt((grp["residual"] ** 2).mean())
    return pd.Series({"loo_r2": r2, "loo_rmse": rmse, "n": len(grp)})


loo_summary = loo.groupby(["spec", "hour"]).apply(loo_r2).reset_index()
models = ins.merge(loo_summary, on=["spec", "hour"], how="left")
models.to_csv(RESULTS / "predict_deviation_per_hour_v2_models.csv", index=False)
coefs.to_csv(RESULTS / "predict_deviation_per_hour_v2_coefs.csv", index=False)
loo.to_csv(RESULTS / "predict_deviation_per_hour_v2_loo_predictions.csv",
           index=False)


# Overall LOO R^2 per spec
overall = []
for s in ["M_shares", "M_full"]:
    sub = loo[loo["spec"] == s]
    ss_res = (sub["residual"] ** 2).sum()
    ss_tot = ((sub["observed"] - sub["observed"].mean()) ** 2).sum()
    overall.append({
        "spec": s,
        "overall_loo_r2": 1 - ss_res / ss_tot,
        "overall_rmse":   float(np.sqrt((sub["residual"] ** 2).mean())),
        "n":              int(len(sub)),
    })
overall_df = pd.DataFrame(overall)


# ---------------------------------------------------------------------------
# Plot 1: DE LOO predicted vs observed, per held-out year, M_full
# ---------------------------------------------------------------------------
de_loo = loo[(loo["spec"] == "M_full") & (loo["country"] == "DE")]
years = sorted(de_loo["year"].unique())
NCOLS, NROWS = 4, int(np.ceil(len(years) / 4))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(6.67, 3.6),
                         sharex=True, sharey=True, constrained_layout=True)
axes = axes.flatten()
for i, yr in enumerate(years):
    ax = axes[i]
    sub = de_loo[de_loo["year"] == yr].sort_values("hour")
    ax.plot(sub["hour"], sub["observed"], color="black",
            linewidth=1.2, marker="o", markersize=2.5, label="observed")
    ax.plot(sub["hour"], sub["predicted"], color="#d62728",
            linewidth=1.2, marker="s", markersize=2.5,
            linestyle="--", label="LOO prediction (M_full)")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_title(f"DE {yr}", fontsize=8, pad=2)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.spines[["top", "right"]].set_visible(False)
for j in range(len(years), len(axes)):
    axes[j].set_visible(False)
fig.supxlabel("Hour of day", fontsize=8, y=0.02)
fig.supylabel("Deviation from annual mean (€/MWh)", fontsize=8, x=0.005)
fig.suptitle("DE — leave-one-year-out predicted vs observed (M_full = shares + cmean)",
             fontsize=8, y=1.005)
axes[0].legend(loc="upper left", frameon=False, handlelength=1.5)
fig.savefig(PLOTS / "predict_deviation_per_hour_v2_DE_loo.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_per_hour_v2_DE_loo.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: LOO R^2 per hour, M_shares vs M_full
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.67, 2.8))
for spec, color, marker in [("M_shares", "#1f77b4", "o"),
                            ("M_full",   "#d62728", "s")]:
    sub = loo_summary[loo_summary["spec"] == spec].sort_values("hour")
    ax.plot(sub["hour"], sub["loo_r2"], color=color, marker=marker,
            markersize=3.5, linewidth=1.2, label=spec)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day", fontsize=8)
ax.set_ylabel("LOO R²", fontsize=8)
ax.set_xticks(range(0, 24, 3))
ax.set_xlim(-0.5, 23.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
ax.legend(frameon=False, loc="lower center", ncol=2)
ax.set_title("Per-hour LOO out-of-sample R²", fontsize=8)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "predict_deviation_per_hour_v2_loo_r2_by_hour.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "predict_deviation_per_hour_v2_loo_r2_by_hour.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Per-hour OLS v2:  dev(c,y,h) ~ shares  vs  shares + cmean")
out.append("=" * 78)
out.append("\nOverall LOO R² (across all hours, all observations):")
for _, r in overall_df.iterrows():
    out.append(f"  {r['spec']:<10s}: LOO R² = {r['overall_loo_r2']:.3f}, "
               f"RMSE = {r['overall_rmse']:.2f}, n = {int(r['n'])}")
out.append("")

# Side-by-side per hour
out.append("Per-hour LOO R² comparison:")
out.append(f"{'h':>3}  {'M_shares':>10}  {'M_full':>10}  {'delta':>8}")
out.append("-" * 78)
piv = loo_summary.pivot(index="hour", columns="spec", values="loo_r2")
for h, row in piv.iterrows():
    delta = row.get("M_full", np.nan) - row.get("M_shares", np.nan)
    out.append(f"{int(h):>3}  {row.get('M_shares', np.nan):>10.3f}  "
               f"{row.get('M_full', np.nan):>10.3f}  {delta:>+8.3f}")

# Top predictors in M_full
out.append("\nTop coefficients in M_full (avg |coef| across hours):")
mean_abs = (coefs[(coefs["spec"] == "M_full") & (coefs["predictor"] != "const")]
            .groupby("predictor")["coef"].apply(lambda s: s.abs().mean())
            .sort_values(ascending=False))
for p, v in mean_abs.head(8).items():
    out.append(f"  {p:<25s}: mean |coef| = {v:.2f}")

text = "\n".join(out)
print(text)
(RESULTS / "predict_deviation_per_hour_v2_summary.txt").write_text(text, encoding="utf-8")
print(f"\nFiles written to {RESULTS}")
