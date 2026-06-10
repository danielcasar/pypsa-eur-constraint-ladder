"""
Forward forecast of DE's HOD mean-deviation profile, with residual-bootstrap
prediction intervals.

Procedure:
  1. Fit M_full (per-hour OLS: dev ~ shares + cmean) on the full 2018-2025
     panel. Store fitted parameters per hour and residuals per hour.
  2. Build DE's projected tech mix for the target year by linear extrapolation
     of each share over 2018-2025, clipped to [0, 1] and renormalised.
  3. Point-predict the 24-element deviation profile.
  4. Bootstrap PI: for each of B=1000 iterations and each hour h, sample one
     residual from the empirical residual distribution AT THAT HOUR and add
     to the point prediction. Take quantiles (p5, p25, p50, p75, p95) of
     the bootstrap fan per hour.
  5. Sanity check: do the same forecast for DE 2025 using 2018-2024
     extrapolation, and compare to actual DE 2025 observations.

Outputs:
  Analysis/results/forecast_de_<TARGET_YEAR>.csv
  Analysis/results/plots/forecast_de_<TARGET_YEAR>.{pdf,png}
  Analysis/results/forecast_de_2025_validation.csv
  Analysis/results/plots/forecast_de_2025_validation.{pdf,png}
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

B = 1000   # bootstrap iterations
TARGET_COUNTRY = "DE"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
dev   = pd.read_csv(DEV_FILE)
state = pd.read_csv(STATE_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")

# cmean per country (mean of shares across all available years for that country)
cmean_table = state.groupby("country")[SHARES].mean().reset_index()
cmean_table.columns = ["country"] + [f"cmean_{t}" for t in DIFF_TECHS]
panel = panel.merge(cmean_table, on="country", how="left")
print(f"Panel: {len(panel)} rows.")


def fit_per_hour(train_df: pd.DataFrame):
    """Fit OLS per hour. Returns dict h -> (params, X_columns, residuals)."""
    out = {}
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=["deviation"] + PREDICTORS)
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


def predict_with_pi(model_per_hour: dict,
                    feature_row: pd.Series,
                    B: int = 1000):
    """Return DataFrame indexed by hour with point + p5/p25/p50/p75/p95."""
    rows = []
    for h in range(24):
        if h not in model_per_hour:
            continue
        m = model_per_hour[h]
        # Build the X vector matching m["x_cols"] (which includes 'const')
        x = pd.Series({c: (1.0 if c == "const" else feature_row[c])
                       for c in m["x_cols"]})
        point = float(np.dot(x.values, m["params"].values))

        # Residual bootstrap
        resids = m["residuals"]
        sampled = rng.choice(resids, size=B, replace=True)
        boot_preds = point + sampled

        rows.append({
            "hour": h,
            "point": point,
            "p05": float(np.percentile(boot_preds, 5)),
            "p25": float(np.percentile(boot_preds, 25)),
            "p50": float(np.percentile(boot_preds, 50)),
            "p75": float(np.percentile(boot_preds, 75)),
            "p95": float(np.percentile(boot_preds, 95)),
            "n_residuals": int(len(resids)),
        })
    return pd.DataFrame(rows).sort_values("hour")


def extrapolate_de_shares(panel_de: pd.DataFrame, target_year: int) -> pd.Series:
    """Linearly extrapolate each share, clip to [0,1], renormalise to sum 1."""
    extr = {}
    for s in SHARES:
        sub = panel_de.dropna(subset=[s])
        if len(sub) < 2:
            extr[s] = np.nan
            continue
        slope, intercept = np.polyfit(sub["year"].values, sub[s].values, 1)
        extr[s] = max(0.0, slope * target_year + intercept)
    # Add the 'other' category implicitly (renormalise around it if non-zero)
    s_sum = sum(extr.values())
    if s_sum > 1.0:
        # rescale to sum 1 (assumes 'other' goes to zero)
        extr = {k: v / s_sum for k, v in extr.items()}
    elif s_sum < 1.0:
        # leave as is — implicit 'other' fills the gap
        pass
    return pd.Series(extr)


def country_features(country: str, target_year: int,
                     panel_subset: pd.DataFrame, cmean_row: pd.Series) -> pd.Series:
    panel_c = panel_subset[panel_subset["country"] == country][["year"] + SHARES].drop_duplicates()
    extr = extrapolate_de_shares(panel_c, target_year)
    out = pd.Series({**extr, **cmean_row.to_dict()})
    return out


# ---------------------------------------------------------------------------
# 1. VALIDATION: predict DE 2025 using 2018-2024 model + 2018-2024 extrap
# ---------------------------------------------------------------------------
print("\n--- Validation: DE 2025 from 2018-2024 ---")
train24 = panel[panel["year"] != 2025]
m24 = fit_per_hour(train24)
de_panel_24 = train24[train24["country"] == "DE"][["year"] + SHARES].drop_duplicates()
extrap_de_25 = extrapolate_de_shares(de_panel_24, 2025)
cmean_de = panel[panel["country"] == "DE"][CMEAN].iloc[0]
features_25 = pd.concat([extrap_de_25, cmean_de])
forecast_25 = predict_with_pi(m24, features_25, B=B)

# Observed DE 2025 deviation
obs_25 = (panel[(panel["country"] == "DE") & (panel["year"] == 2025)]
          [["hour", "deviation"]].sort_values("hour"))
forecast_25 = forecast_25.merge(obs_25, on="hour", how="left")
forecast_25.to_csv(RESULTS / "forecast_de_2025_validation.csv", index=False)

# Coverage check (does observed fall in 90% PI?)
covered = ((forecast_25["deviation"] >= forecast_25["p05"]) &
           (forecast_25["deviation"] <= forecast_25["p95"])).mean()
print(f"DE 2025 90% PI coverage: {covered*100:.0f}% of hours within band")

# ---------------------------------------------------------------------------
# 2. FORWARD FORECAST: DE 2030 from full 2018-2025 model + 2018-2025 extrap
# ---------------------------------------------------------------------------
print("\n--- Forecast: DE 2030 from full 2018-2025 ---")
m_full = fit_per_hour(panel)
de_panel_full = panel[panel["country"] == "DE"][["year"] + SHARES].drop_duplicates()
extrap_de_30 = extrapolate_de_shares(de_panel_full, 2030)
features_30 = pd.concat([extrap_de_30, cmean_de])
forecast_30 = predict_with_pi(m_full, features_30, B=B)
forecast_30.to_csv(RESULTS / "forecast_de_2030.csv", index=False)

print("DE 2030 extrapolated tech shares:")
for k, v in extrap_de_30.items():
    print(f"  {k}: {v:.3f}")
print(f"  (sum = {extrap_de_30.sum():.3f}; remainder is 'other')")


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_forecast(df: pd.DataFrame, observed: pd.Series | None,
                  title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    df = df.sort_values("hour")
    ax.fill_between(df["hour"], df["p05"], df["p95"],
                    color="#d62728", alpha=0.18, linewidth=0,
                    label="90% PI (residual bootstrap)")
    ax.fill_between(df["hour"], df["p25"], df["p75"],
                    color="#d62728", alpha=0.30, linewidth=0,
                    label="50% PI")
    ax.plot(df["hour"], df["point"], color="#d62728",
            linewidth=1.4, marker="s", markersize=3,
            label="Point forecast")
    if observed is not None:
        ax.plot(observed.index, observed.values, color="black",
                linewidth=1.4, marker="o", markersize=3,
                label="Observed")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_ylabel("Deviation from annual mean (€/MWh)", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", handlelength=1.5)
    plt.tight_layout(pad=0.4)
    fig.savefig(outpath.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# Validation plot
obs_series = obs_25.set_index("hour")["deviation"]
plot_forecast(
    forecast_25, obs_series,
    f"Validation: DE 2025 forecast from 2018–2024 (90% PI coverage = {covered*100:.0f}%)",
    PLOTS / "forecast_de_2025_validation",
)
print(f"Wrote {PLOTS / 'forecast_de_2025_validation.pdf'}")

# Forward plot
plot_forecast(
    forecast_30, None,
    "Forward forecast: DE 2030 — projected HOD deviation profile, residual-bootstrap PI",
    PLOTS / "forecast_de_2030",
)
print(f"Wrote {PLOTS / 'forecast_de_2030.pdf'}")


# Print compact summary
out = []
out.append("=" * 78)
out.append("DE 2030 forecast — point prediction and prediction intervals")
out.append("=" * 78)
out.append(f"\n{'h':>3}  {'point':>8}  {'p05':>8}  {'p25':>8}  {'p50':>8}  "
           f"{'p75':>8}  {'p95':>8}")
out.append("-" * 64)
for _, r in forecast_30.iterrows():
    out.append(f"{int(r['hour']):>3}  "
               f"{r['point']:>+8.2f}  {r['p05']:>+8.2f}  {r['p25']:>+8.2f}  "
               f"{r['p50']:>+8.2f}  {r['p75']:>+8.2f}  {r['p95']:>+8.2f}")

out.append(f"\nDE 2025 validation: 90% PI captured "
           f"{covered*100:.0f}% of observed hours.")
text = "\n".join(out)
print()
print(text)
(RESULTS / "forecast_de_summary.txt").write_text(text, encoding="utf-8")
