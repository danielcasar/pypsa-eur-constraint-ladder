"""
All-22-country 1y holdout, but with COUNTRY-SPECIFIC residual pools for the PI.

Same model and point forecasts as holdout_all_countries_2024.py — only the
PI bootstrap source changes:

  Old: pool LOO residuals across ALL countries × years per hour
       (DE PI inherits RO/BG outlier variance — too wide)

  New: pool LOO residuals from a SINGLE country (or country-cluster) per hour
       (DE PI uses only DE's past prediction errors)

Two variants reported:
  - per_country: each country bootstraps from its own LOO residuals
                 (9 residuals per (country, hour) — thin but most relevant)
  - per_country_allhours: pool all 24*9=216 residuals per country, hour-blind
                          (more stable, loses hour-specific heteroscedasticity)

Outputs:
  Analysis/results/holdout_all_countries_2024_country_pi.csv
  Analysis/results/holdout_all_countries_2024_country_pi_summary.txt
  Analysis/results/plots/holdout_all_countries_2024_country_pi_panels.{pdf,png}
  Analysis/results/plots/holdout_pi_width_comparison.{pdf,png}
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
TRAIN_END = 2023
TARGET = 2024


# ---------------------------------------------------------------------------
state = pd.read_csv(CAP_PANEL)
dev = pd.read_csv(DEV_FILE)
panel = dev.merge(state, on=["country", "year"], how="inner")
panel = panel[panel["country"].isin(COUNTRIES)]
panel_train = panel[panel["year"] <= TRAIN_END]


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


# ---------------------------------------------------------------------------
# Build LOO residuals on train data, with COUNTRY label preserved
# ---------------------------------------------------------------------------
print("Fitting + LOO over train years (preserving country) ...")
m_holdout = fit_per_hour(panel_train)
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
        loo_rows.append({"country": r["country"], "year": int(y0),
                         "hour": h, "residual": float(r["deviation"]) - p})
loo = pd.DataFrame(loo_rows)
print(f"  LOO rows: {len(loo)}; residuals per (country, hour): "
      f"{loo.groupby(['country', 'hour']).size().describe().loc[['min', '50%', 'max']].to_dict()}")


# ---------------------------------------------------------------------------
# Forecast 2024 for each country, three PI variants:
#   pool_all       : original — all-country, per-hour pool
#   country_hour   : country-specific, per-hour pool (~9 residuals/cell)
#   country_pooled : country-specific, all-hour pool (~216 residuals/country)
# ---------------------------------------------------------------------------
def pi_from_pool(point: float, pool: np.ndarray, B: int = B) -> dict:
    if len(pool) < 5:
        return {k: np.nan for k in ["p05", "p25", "p50", "p75", "p95"]}
    sampled = rng.choice(pool, size=B, replace=True)
    boot = point + sampled
    return {
        "p05": float(np.percentile(boot, 5)),
        "p25": float(np.percentile(boot, 25)),
        "p50": float(np.percentile(boot, 50)),
        "p75": float(np.percentile(boot, 75)),
        "p95": float(np.percentile(boot, 95)),
    }


pool_all       = {h: g["residual"].values for h, g in loo.groupby("hour")}
pool_chour     = {(c, h): g["residual"].values for (c, h), g in loo.groupby(["country", "hour"])}
pool_cpooled   = {c: g["residual"].values for c, g in loo.groupby("country")}

print(f"\nForecasting {TARGET} for {len(COUNTRIES)} countries (3 PI variants) ...")
rows = []
for cc in COUNTRIES:
    pc_train = (panel_train[panel_train["country"] == cc]
                [["year"] + SHARES].drop_duplicates())
    if len(pc_train) < 2:
        continue
    extr = extrapolate_capshare(pc_train, TARGET)
    obs_country = panel[(panel["country"] == cc) & (panel["year"] == TARGET)]
    for h in range(24):
        if h not in m_holdout or cc not in m_holdout[h]["countries"]:
            continue
        point = predict_for_row(m_holdout[h], cc, extr)
        obs = obs_country[obs_country["hour"] == h]
        obs_v = float(obs["deviation"].iloc[0]) if len(obs) else np.nan
        for variant, pool in [
            ("pool_all",       pool_all.get(h, np.array([]))),
            ("country_hour",   pool_chour.get((cc, h), np.array([]))),
            ("country_pooled", pool_cpooled.get(cc, np.array([]))),
        ]:
            pi = pi_from_pool(point, pool)
            rows.append({
                "country": cc, "hour": h, "variant": variant,
                "point": point, **pi, "deviation": obs_v,
                "n_residuals": int(len(pool)),
            })
fc = pd.DataFrame(rows)
fc.to_csv(RESULTS / "holdout_all_countries_2024_country_pi.csv", index=False)


# ---------------------------------------------------------------------------
# Per-(country, variant) coverage + PI width metrics
# ---------------------------------------------------------------------------
def metrics(d: pd.DataFrame) -> dict:
    m = d["deviation"].notna() & d["p05"].notna()
    if not m.any():
        return {"n": 0, "rmse": np.nan, "cov90": np.nan, "cov50": np.nan,
                "pi90_width": np.nan, "pi50_width": np.nan}
    err = d.loc[m, "point"] - d.loc[m, "deviation"]
    return {
        "n": int(m.sum()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "cov90": float(((d.loc[m, "deviation"] >= d.loc[m, "p05"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p95"])).mean()),
        "cov50": float(((d.loc[m, "deviation"] >= d.loc[m, "p25"]) &
                        (d.loc[m, "deviation"] <= d.loc[m, "p75"])).mean()),
        "pi90_width": float((d.loc[m, "p95"] - d.loc[m, "p05"]).mean()),
        "pi50_width": float((d.loc[m, "p75"] - d.loc[m, "p25"]).mean()),
    }


met_rows = []
for (cc, var), g in fc.groupby(["country", "variant"]):
    met_rows.append({"country": cc, "variant": var, **metrics(g)})
met = pd.DataFrame(met_rows)


# ---------------------------------------------------------------------------
# Compare PI widths across variants
# ---------------------------------------------------------------------------
print("\nMedian PI width across countries (smaller = tighter, more useful):")
piv_w90 = met.pivot(index="country", columns="variant", values="pi90_width")
piv_w50 = met.pivot(index="country", columns="variant", values="pi50_width")
piv_c90 = met.pivot(index="country", columns="variant", values="cov90")
piv_c50 = met.pivot(index="country", columns="variant", values="cov50")

print("\n  90% PI width (€/MWh, mean per hour, median across countries):")
print(piv_w90.median().to_string())
print("\n  90% PI coverage (target 90%, median across countries):")
print((piv_c90.median() * 100).to_string())
print("\n  50% PI width (€/MWh):")
print(piv_w50.median().to_string())
print("\n  50% PI coverage (target 50%):")
print((piv_c50.median() * 100).to_string())


# ---------------------------------------------------------------------------
# Plot 1: bar chart comparing 90% PI width per country across variants
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.0))
piv_sorted = piv_w90.sort_values("pool_all")
x = np.arange(len(piv_sorted))
W = 0.27
ax.bar(x - W, piv_sorted["pool_all"], width=W, color="#999999",
       label="pool_all (original)")
ax.bar(x,     piv_sorted["country_pooled"], width=W, color="#1f77b4",
       label="country_pooled")
ax.bar(x + W, piv_sorted["country_hour"], width=W, color="#d62728",
       label="country_hour")
ax.set_xticks(x); ax.set_xticklabels(piv_sorted.index, rotation=0)
ax.set_ylabel("Mean 90% PI width (€/MWh)")
ax.set_title(f"PI width comparison across countries (1y-ahead holdout {TARGET})")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "holdout_pi_width_comparison.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "holdout_pi_width_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-country small multiples comparing pool_all vs country_hour PI
# ---------------------------------------------------------------------------
ccs = sorted(fc["country"].unique())
NCOLS = 6
NROWS = int(np.ceil(len(ccs) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.5, 1.5 * NROWS),
                         sharex=True, constrained_layout=True)
axes = axes.flatten()
for i, cc in enumerate(ccs):
    ax = axes[i]
    d_all = fc[(fc["country"] == cc) & (fc["variant"] == "pool_all")].sort_values("hour")
    d_ch  = fc[(fc["country"] == cc) & (fc["variant"] == "country_hour")].sort_values("hour")
    ax.fill_between(d_all["hour"], d_all["p05"], d_all["p95"],
                    color="#999999", alpha=0.30, linewidth=0, label="90% PI (pool_all)")
    ax.fill_between(d_ch["hour"], d_ch["p05"], d_ch["p95"],
                    color="#d62728", alpha=0.30, linewidth=0, label="90% PI (country_hour)")
    ax.plot(d_all["hour"], d_all["point"], color="black", linewidth=0.8,
            marker="s", markersize=1.5)
    if d_all["deviation"].notna().any():
        ax.plot(d_all["hour"], d_all["deviation"], color="black",
                linewidth=0.8, marker="o", markersize=1.5)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    r_all = metrics(d_all); r_ch = metrics(d_ch)
    def fmt_pct(x): return f"{int(x*100)}%" if not np.isnan(x) else "n/a"
    def fmt_w(x): return f"{x:.0f}" if not np.isnan(x) else "n/a"
    ax.set_title(f"{cc}: cov90 {fmt_pct(r_all['cov90'])}->{fmt_pct(r_ch['cov90'])}, "
                 f"w90 {fmt_w(r_all['pi90_width'])}->{fmt_w(r_ch['pi90_width'])}",
                 fontsize=7, pad=2)
    ax.set_xticks([0, 6, 12, 18])
    ax.spines[["top", "right"]].set_visible(False)
for j in range(len(ccs), len(axes)):
    axes[j].set_visible(False)
axes[0].legend(frameon=False, fontsize=6, loc="upper left")
fig.supxlabel("Hour of day", fontsize=8, y=0.01)
fig.supylabel("Deviation (€/MWh)", fontsize=8, x=0.005)
fig.suptitle(f"PI tightening by switching to country-specific residual pool (target {TARGET})",
             fontsize=8, y=1.005)
fig.savefig(PLOTS / "holdout_all_countries_2024_country_pi_panels.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "holdout_all_countries_2024_country_pi_panels.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append(f"PI variants for 1y-ahead holdout (target {TARGET}, train 2015-{TRAIN_END})")
out.append("=" * 78)
out.append("\nResidual pool variants:")
out.append("  pool_all       : LOO residuals across ALL countries x ALL years, per hour")
out.append("                   (every country inherits all other countries' variance)")
out.append("  country_hour   : LOO residuals from THIS country THIS hour only (~9 each)")
out.append("                   (most relevant; thin pool, possibly noisy)")
out.append("  country_pooled : LOO residuals from THIS country, all hours (~216 each)")
out.append("                   (stable; loses hour-specific heteroscedasticity)")

out.append(f"\n{'variant':<18}  {'med w90':>9}  {'med w50':>9}  "
           f"{'med cov90':>10}  {'med cov50':>10}")
out.append("-" * 78)
for v in ["pool_all", "country_hour", "country_pooled"]:
    g = met[met["variant"] == v]
    out.append(f"{v:<18}  {g['pi90_width'].median():>9.1f}  "
               f"{g['pi50_width'].median():>9.1f}  "
               f"{g['cov90'].median()*100:>9.0f}%  {g['cov50'].median()*100:>9.0f}%")

out.append(f"\nDE comparison:")
out.append(f"{'variant':<18}  {'w90':>9}  {'w50':>9}  {'cov90':>9}  {'cov50':>9}  "
           f"{'rmse':>7}")
out.append("-" * 78)
for v in ["pool_all", "country_hour", "country_pooled"]:
    r = met[(met["country"] == "DE") & (met["variant"] == v)].iloc[0]
    out.append(f"{v:<18}  {r['pi90_width']:>9.1f}  {r['pi50_width']:>9.1f}  "
               f"{r['cov90']*100:>8.0f}%  {r['cov50']*100:>8.0f}%  {r['rmse']:>7.2f}")

out.append("\nPer-country, per-variant detail:")
out.append(f"{'cc':<4}  {'variant':<18}  {'w90':>7}  {'cov90':>7}  {'rmse':>6}")
out.append("-" * 78)
for cc in sorted(met["country"].unique()):
    for v in ["pool_all", "country_hour", "country_pooled"]:
        g = met[(met["country"] == cc) & (met["variant"] == v)]
        if g.empty:
            continue
        r = g.iloc[0]
        if pd.isna(r["pi90_width"]):
            continue
        out.append(f"{cc:<4}  {v:<18}  {r['pi90_width']:>7.1f}  "
                   f"{r['cov90']*100:>6.0f}%  {r['rmse']:>6.1f}")

text = "\n".join(out)
print()
print(text)
(RESULTS / "holdout_all_countries_2024_country_pi_summary.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {RESULTS / 'holdout_all_countries_2024_country_pi_summary.txt'}")
