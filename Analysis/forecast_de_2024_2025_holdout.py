"""
Strict holdout: train on 2018-2023 only, forecast DE 2024 AND 2025 with
residual-bootstrap PI, compare to observed.

Difference vs forecast_de_with_pi.py:
  - That script's "validation" trains on 2018-2024 to predict 2025 (only one
    year held out).
  - Here we hold out the LAST TWO years simultaneously: train on 2018-2023,
    forecast both 2024 and 2025. cmean is also recomputed from 2018-2023 only
    to avoid any leakage of future-year structure into the country-mean
    predictors.

Procedure:
  1. Filter panel to year <= 2023.
  2. Recompute cmean per country from filtered state.
  3. Fit per-hour OLS  dev ~ shares + cmean  on the 2018-2023 panel.
  4. Linearly extrapolate DE shares from 2018-2023 to target years 2024, 2025.
  5. Predict + bootstrap PI per hour for each target year.
  6. Compare to observed DE 2024 / 2025; report 90% PI coverage.

Outputs:
  Analysis/results/forecast_de_2024_holdout.csv
  Analysis/results/forecast_de_2025_holdout.csv
  Analysis/results/plots/forecast_de_2024_2025_holdout.{pdf,png}
  Analysis/results/forecast_de_2024_2025_holdout_summary.txt
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
PREDICTORS = SHARES + CMEAN

B = 1000
TARGET_COUNTRY = "DE"
TRAIN_END = 2023
TARGETS = [2024, 2025]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
dev   = pd.read_csv(DEV_FILE)
state = pd.read_csv(STATE_FILE)
panel_full = dev.merge(state, on=["country", "year"], how="inner")

# Train slice + leak-free cmean
state_train = state[state["year"] <= TRAIN_END]
cmean_table = (state_train.groupby("country")[SHARES].mean().reset_index()
               .rename(columns={f"share_{t}": f"cmean_{t}" for t in DIFF_TECHS}))
panel_full = panel_full.merge(cmean_table, on="country", how="left")

panel_train = panel_full[panel_full["year"] <= TRAIN_END].copy()
print(f"Train panel: {len(panel_train)} rows  (years <= {TRAIN_END}, "
      f"{panel_train['country'].nunique()} countries)")
print(f"Holdout target years: {TARGETS}")


def fit_per_hour(df: pd.DataFrame):
    out = {}
    for h in range(24):
        sub = df[df["hour"] == h].dropna(subset=["deviation"] + PREDICTORS)
        if len(sub) < len(PREDICTORS) + 5:
            continue
        X = sm.add_constant(sub[PREDICTORS].astype(float))
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {
            "params": res.params,
            "x_cols": list(X.columns),
            "residuals": (y - res.predict(X)).values,
        }
    return out


def predict_with_pi(model_per_hour: dict, feature_row: pd.Series,
                    B: int = 1000) -> pd.DataFrame:
    rows = []
    for h in range(24):
        if h not in model_per_hour:
            continue
        m = model_per_hour[h]
        x = pd.Series({c: (1.0 if c == "const" else feature_row[c])
                       for c in m["x_cols"]})
        point = float(np.dot(x.values, m["params"].values))
        sampled = rng.choice(m["residuals"], size=B, replace=True)
        boot = point + sampled
        rows.append({
            "hour": h,
            "point": point,
            "p05": float(np.percentile(boot, 5)),
            "p25": float(np.percentile(boot, 25)),
            "p50": float(np.percentile(boot, 50)),
            "p75": float(np.percentile(boot, 75)),
            "p95": float(np.percentile(boot, 95)),
            "n_residuals": int(len(m["residuals"])),
        })
    return pd.DataFrame(rows).sort_values("hour")


def extrapolate_shares(panel_c: pd.DataFrame, target_year: int) -> pd.Series:
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


# ---------------------------------------------------------------------------
# Fit on 2018-2023
# ---------------------------------------------------------------------------
print("\nFitting per-hour OLS on years <= 2023 ...")
m_holdout = fit_per_hour(panel_train)
print(f"  Hours fit: {len(m_holdout)}/24")

de_train = (panel_train[panel_train["country"] == TARGET_COUNTRY]
            [["year"] + SHARES].drop_duplicates())
print(f"  DE training years available: {sorted(de_train['year'].unique())}")
cmean_de = panel_full[panel_full["country"] == TARGET_COUNTRY][CMEAN].iloc[0]


# ---------------------------------------------------------------------------
# Forecast both targets
# ---------------------------------------------------------------------------
forecasts = {}
for ty in TARGETS:
    extr = extrapolate_shares(de_train, ty)
    feats = pd.concat([extr, cmean_de])
    fc = predict_with_pi(m_holdout, feats, B=B)
    obs = (panel_full[(panel_full["country"] == TARGET_COUNTRY) &
                      (panel_full["year"] == ty)]
           [["hour", "deviation"]].sort_values("hour"))
    fc = fc.merge(obs, on="hour", how="left")
    fc.to_csv(RESULTS / f"forecast_de_{ty}_holdout.csv", index=False)
    forecasts[ty] = (fc, extr)
    print(f"\nDE {ty} extrapolated shares (slope from {de_train['year'].min()}-"
          f"{de_train['year'].max()}):")
    for k, v in extr.items():
        print(f"  {k}: {v:.3f}")
    print(f"  (sum = {extr.sum():.3f}; remainder is implicit 'other')")


def coverage(fc: pd.DataFrame, lo: str, hi: str) -> float:
    mask = fc["deviation"].notna()
    if not mask.any():
        return float("nan")
    return float(((fc.loc[mask, "deviation"] >= fc.loc[mask, lo]) &
                  (fc.loc[mask, "deviation"] <= fc.loc[mask, hi])).mean())


def rmse(fc: pd.DataFrame) -> float:
    mask = fc["deviation"].notna()
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((fc.loc[mask, "point"] - fc.loc[mask, "deviation"]) ** 2)
                         .mean()))


# ---------------------------------------------------------------------------
# Plot: side-by-side panels, both target years
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 3.0),
                         sharey=True, constrained_layout=True)
if len(TARGETS) == 1:
    axes = [axes]
for ax, ty in zip(axes, TARGETS):
    df, _ = forecasts[ty]
    ax.fill_between(df["hour"], df["p05"], df["p95"], color="#d62728",
                    alpha=0.18, linewidth=0, label="90% PI")
    ax.fill_between(df["hour"], df["p25"], df["p75"], color="#d62728",
                    alpha=0.30, linewidth=0, label="50% PI")
    ax.plot(df["hour"], df["point"], color="#d62728", linewidth=1.4,
            marker="s", markersize=3, label="Point forecast")
    if df["deviation"].notna().any():
        ax.plot(df["hour"], df["deviation"], color="black", linewidth=1.4,
                marker="o", markersize=3, label="Observed")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    cov90 = coverage(df, "p05", "p95")
    cov50 = coverage(df, "p25", "p75")
    rm = rmse(df)
    ax.set_title(f"DE {ty}  (90% cov = {cov90*100:.0f}%, "
                 f"50% cov = {cov50*100:.0f}%, RMSE = {rm:.1f} €/MWh)",
                 fontsize=8)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Deviation from annual mean (€/MWh)", fontsize=8)
axes[0].legend(frameon=False, loc="best", handlelength=1.5)
fig.suptitle(f"Holdout forecast of DE 2024 + 2025 trained on 2018-{TRAIN_END}",
             fontsize=8, y=1.02)
fig.savefig(PLOTS / "forecast_de_2024_2025_holdout.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "forecast_de_2024_2025_holdout.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append(f"Holdout forecast: train years 2018-{TRAIN_END}, forecast {TARGETS}")
out.append("=" * 78)
for ty in TARGETS:
    fc, _ = forecasts[ty]
    cov90 = coverage(fc, "p05", "p95")
    cov50 = coverage(fc, "p25", "p75")
    rm = rmse(fc)
    out.append(f"\nDE {ty}: 90% PI coverage = {cov90*100:.0f}% "
               f"(nominal 90%), 50% PI = {cov50*100:.0f}% (nominal 50%), "
               f"point RMSE = {rm:.2f} €/MWh")
    out.append(f"{'h':>3}  {'point':>8}  {'p05':>8}  {'p95':>8}  {'obs':>8}  "
               f"{'in_PI':>6}")
    out.append("-" * 64)
    for _, r in fc.iterrows():
        in_pi = (("Y" if (r["deviation"] >= r["p05"] and
                          r["deviation"] <= r["p95"]) else "N")
                 if pd.notna(r["deviation"]) else "-")
        obs_str = f"{r['deviation']:+8.2f}" if pd.notna(r["deviation"]) else f"{'na':>8}"
        out.append(f"{int(r['hour']):>3}  {r['point']:+8.2f}  "
                   f"{r['p05']:+8.2f}  {r['p95']:+8.2f}  {obs_str}  "
                   f"{in_pi:>6}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "forecast_de_2024_2025_holdout_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'forecast_de_2024_2025_holdout_summary.txt'}")
print(f"Plot:  {PLOTS / 'forecast_de_2024_2025_holdout.pdf'}")
