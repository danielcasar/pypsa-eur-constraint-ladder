"""
Test whether the load-share vs capacity-share collinearity is weakening
over time. Split the 2015-2025 panel into two halves:

  Early window  : 2015-2019  (low BESS, low EV penetration, low DR)
  Late  window  : 2020-2025  (BESS scaling up, EV growth, DR expanding)

For each window, fit three specs per hour:
  S1 : dev_h ~ FE + load_share_h
  S2 : dev_h ~ FE + caps
  S3 : dev_h ~ FE + caps + load_share_h

Look at:
  - load_share coefficient in S3 (partial after caps): if it grows / becomes
    more significant in the late window -> collinearity weakening / decoupling
    has started
  - R^2 gain from adding load to caps (S3 - S2): same logic, aggregate version

Outputs:
  Analysis/results/load_cap_collinearity_decay_summary.txt
  Analysis/results/plots/load_cap_collinearity_decay.{pdf,png}
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
    "legend.fontsize": 8, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DEV_FILE   = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_panel_with_neighbors.csv"
LOAD_PANEL = RESULTS / "load_share_panel.csv"

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]

WINDOWS = {
    "early_2015_2019": (2015, 2019),
    "late_2020_2025":  (2020, 2025),
}


# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
load = pd.read_csv(LOAD_PANEL)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .merge(load[["country", "year", "hour", "load_share_h"]],
                   on=["country", "year", "hour"], how="left")
            .query("country in @COUNTRIES")
            .dropna(subset=["deviation", "load_share_h"] + CAPSHARES))
print(f"Panel: {len(panel)} rows, {panel['country'].nunique()} countries, "
      f"years {panel['year'].min()}-{panel['year'].max()}")


def make_X(df, train_countries, predictors):
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X


def fit_per_hour(df, predictors):
    out = {}
    countries = sorted(df["country"].unique())
    for h in range(24):
        sub = df[df["hour"] == h].dropna(subset=["deviation"] + predictors)
        if len(sub) < len(predictors) + len(countries) + 5:
            continue
        X = make_X(sub, countries, predictors)
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {
            "params": res.params, "x_cols": list(X.columns),
            "tvalues": res.tvalues, "rsq": float(res.rsquared),
            "n": int(res.nobs),
        }
    return out


def load_coef_t(model_dict):
    rows = []
    for h, m in model_dict.items():
        if "load_share_h" not in m["x_cols"]:
            continue
        idx = m["x_cols"].index("load_share_h")
        rows.append({"hour": h,
                     "coef": float(m["params"].values[idx]),
                     "t": float(m["tvalues"].values[idx])})
    return pd.DataFrame(rows).sort_values("hour")


# ---------------------------------------------------------------------------
# Fit each spec in each window
# ---------------------------------------------------------------------------
results = {}
for win_name, (y_lo, y_hi) in WINDOWS.items():
    sub = panel[(panel["year"] >= y_lo) & (panel["year"] <= y_hi)]
    print(f"\nWindow {win_name}: {len(sub)} rows, "
          f"{sub['country'].nunique()} countries, "
          f"years {sub['year'].min()}-{sub['year'].max()}")
    m_load     = fit_per_hour(sub, ["load_share_h"])
    m_caps     = fit_per_hour(sub, CAPSHARES)
    m_caps_ld  = fit_per_hour(sub, CAPSHARES + ["load_share_h"])
    results[win_name] = {
        "rows": len(sub),
        "m_load": m_load, "m_caps": m_caps, "m_caps_ld": m_caps_ld,
        "load_only_t": load_coef_t(m_load),
        "load_partial_t": load_coef_t(m_caps_ld),
        "r2_caps":  pd.DataFrame([{"hour": h, "r2": m["rsq"]} for h, m in m_caps.items()]),
        "r2_caps_ld": pd.DataFrame([{"hour": h, "r2": m["rsq"]} for h, m in m_caps_ld.items()]),
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Load-shape vs capacity-shape collinearity decay test")
out.append("Early window: 2015-2019.  Late window: 2020-2025.")
out.append("=" * 78)
for win in WINDOWS:
    r = results[win]
    out.append(f"\n--- {win} (n = {r['rows']}) ---")
    out.append("Median R^2 across hours:")
    out.append(f"  M_load_only (FE + load_share):      "
               f"{np.median([m['rsq'] for m in r['m_load'].values()]):.3f}")
    out.append(f"  M_caps_only (FE + caps):             "
               f"{np.median([m['rsq'] for m in r['m_caps'].values()]):.3f}")
    out.append(f"  M_caps + load:                       "
               f"{np.median([m['rsq'] for m in r['m_caps_ld'].values()]):.3f}")
    delta = (np.median([m['rsq'] for m in r['m_caps_ld'].values()]) -
             np.median([m['rsq'] for m in r['m_caps'].values()]))
    out.append(f"  delta (caps+load vs caps only):      {delta:+.4f}  "
               f"(this is the marginal R^2 gain from adding load on top of caps)")

# Partial-coefficient comparison
out.append("\n" + "=" * 78)
out.append("Load coefficient after caps are controlled (S3 partial coefficient)")
out.append("If the late window has LARGER coefficient / |t| -> decoupling started.")
out.append("=" * 78)
out.append(f"\n{'h':>3}  {'early_coef':>11}  {'early_t':>8}  "
           f"{'late_coef':>10}  {'late_t':>8}  "
           f"{'coef_change':>12}  {'|t|_change':>11}")
out.append("-" * 78)
e = results["early_2015_2019"]["load_partial_t"].set_index("hour")
l = results["late_2020_2025"]["load_partial_t"].set_index("hour")
joined = e.join(l, lsuffix="_e", rsuffix="_l").dropna()
for h in sorted(joined.index):
    ce, te = joined.loc[h, "coef_e"], joined.loc[h, "t_e"]
    cl, tl = joined.loc[h, "coef_l"], joined.loc[h, "t_l"]
    out.append(f"{int(h):>3}  {ce:>+11.0f}  {te:>+8.2f}  "
               f"{cl:>+10.0f}  {tl:>+8.2f}  "
               f"{(cl - ce):>+12.0f}  {(abs(tl) - abs(te)):>+11.2f}")

# Summary statistic: did the average |t| or coef increase in the late window?
mean_abs_t_e = joined["t_e"].abs().mean()
mean_abs_t_l = joined["t_l"].abs().mean()
mean_abs_c_e = joined["coef_e"].abs().mean()
mean_abs_c_l = joined["coef_l"].abs().mean()
out.append("\nSummary:")
out.append(f"  mean |partial coef| early -> late: {mean_abs_c_e:.0f} -> {mean_abs_c_l:.0f}  "
           f"({(mean_abs_c_l/mean_abs_c_e - 1)*100:+.1f}%)")
out.append(f"  mean |partial t-stat| early -> late: {mean_abs_t_e:.2f} -> {mean_abs_t_l:.2f}  "
           f"({(mean_abs_t_l/mean_abs_t_e - 1)*100:+.1f}%)")
out.append("")
if mean_abs_t_l > mean_abs_t_e * 1.15:
    out.append("=> Late-window load coefficient is meaningfully STRONGER after controlling")
    out.append("   for caps. This is consistent with the start of supply-demand decoupling.")
elif mean_abs_t_l < mean_abs_t_e * 0.85:
    out.append("=> Late-window load coefficient is WEAKER. Collinearity has TIGHTENED, not")
    out.append("   loosened (e.g., behind-the-meter PV self-consumption is making load more")
    out.append("   tightly tied to capacity).")
else:
    out.append("=> No clear shift. Collinearity strength is approximately stable across windows.")

text = "\n".join(out)
print(text)
(RESULTS / "load_cap_collinearity_decay_summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plot: partial t-stat per hour, early vs late
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.plot(joined.index, joined["t_e"], color="#1f77b4", marker="o", markersize=3,
        linewidth=1.2, label="2015-2019 (early)")
ax.plot(joined.index, joined["t_l"], color="#d62728", marker="s", markersize=3,
        linewidth=1.2, label="2020-2025 (late)")
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(+2, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
ax.axhline(-2, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day")
ax.set_ylabel("t-stat on load_share_h (after caps controlled)")
ax.set_xticks(range(0, 24, 3))
ax.set_title("Partial load coefficient t-stat: early vs late window\n"
             "(rising |t| in late window = supply-demand decoupling beginning)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="best")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "load_cap_collinearity_decay.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "load_cap_collinearity_decay.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote summary + plot to {RESULTS}, {PLOTS}")
