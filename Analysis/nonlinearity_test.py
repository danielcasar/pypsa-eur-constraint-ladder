"""
Test whether non-linear specifications of the structural-shape model
substantially outperform the linear baseline (M2).

Specifications compared per hour (all include country FE):
  S1  Linear (M2 baseline)           : caps + exposed + ntc
  S2  + quadratic main techs         : S1 + capshare_t^2 for t in {solar, onwind,
                                       gas, nuclear, lignite}
  S3  + targeted interactions        : S1 + capshare_solar * {capshare_gas,
                                       capshare_nuclear, capshare_onwind}
  S4  Full nonlinear (S2 union S3)
  S5  Gradient-boosted machine (GBM) : sklearn HistGradientBoostingRegressor on
                                       SAME features as S1 + country one-hot.
                                       Per-hour separate GBM, depth=4, 200 trees.

Compare:
  - Overall LOO R^2 across all hours
  - Per-hour LOO R^2 with linear as reference
  - Coefficients on the new non-linear terms (S2/S3)

Outputs:
  Analysis/results/nonlinearity_test_summary.txt
  Analysis/results/nonlinearity_test_predictions.csv
  Analysis/results/plots/nonlinearity_test_r2_by_hour.{pdf,png}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

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

COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
             "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
             "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPSHARES = [f"capshare_{t}" for t in DIFF_TECHS]
EXPOSED   = [f"exposed_{t}"  for t in DIFF_TECHS]
NTC       = "ntc_share"

QUAD_TECHS  = ["solar", "onwind", "gas", "nuclear", "lignite"]
INTERACTIONS = [("solar", "gas"), ("solar", "nuclear"), ("solar", "onwind")]


# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .query("country in @COUNTRIES")
            .dropna(subset=["deviation"] + CAPSHARES + EXPOSED + [NTC]))

# Build derived features (quadratic + interactions) up front
panel = panel.copy()
for t in QUAD_TECHS:
    panel[f"capshare_{t}_sq"] = panel[f"capshare_{t}"] ** 2
for a, b in INTERACTIONS:
    panel[f"int_{a}_{b}"] = panel[f"capshare_{a}"] * panel[f"capshare_{b}"]

QUAD_FEATS = [f"capshare_{t}_sq" for t in QUAD_TECHS]
INT_FEATS  = [f"int_{a}_{b}" for a, b in INTERACTIONS]

SPECS = {
    "S1_linear":     CAPSHARES + EXPOSED + [NTC],
    "S2_quadratic":  CAPSHARES + EXPOSED + [NTC] + QUAD_FEATS,
    "S3_interactions": CAPSHARES + EXPOSED + [NTC] + INT_FEATS,
    "S4_full_nonlin": CAPSHARES + EXPOSED + [NTC] + QUAD_FEATS + INT_FEATS,
    "S5_gbm":        CAPSHARES + EXPOSED + [NTC],   # same features, GBM fitter
}

print(f"Panel: {len(panel)} rows; {panel['country'].nunique()} countries; "
      f"years {panel['year'].min()}-{panel['year'].max()}")


# ---------------------------------------------------------------------------
# Linear fit per hour (S1-S4)
# ---------------------------------------------------------------------------
def fit_ols_per_hour(train_df, features):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=["deviation"] + features)
        if len(sub) < len(features) + len(countries) + 5:
            continue
        X = pd.DataFrame(index=sub.index); X["const"] = 1.0
        for c in countries[1:]:
            X[f"FE_{c}"] = (sub["country"].values == c).astype(float)
        for f in features:
            X[f] = sub[f].astype(float).values
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        out[h] = {"params": res.params, "x_cols": list(X.columns),
                  "countries": countries}
    return out


def predict_ols(model_h, country, row, features):
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}; x["const"] = 1.0
    if f"FE_{country}" in x:
        x[f"FE_{country}"] = 1.0
    for f in features:
        x[f] = float(row[f])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


# ---------------------------------------------------------------------------
# GBM fit per hour (S5) -- features + country one-hot
# ---------------------------------------------------------------------------
def gbm_features_matrix(df, features, countries):
    M = df[features].values.astype(float)
    # one-hot country dummies (drop first to avoid singular)
    cdum = np.column_stack([(df["country"].values == c).astype(float)
                            for c in countries[1:]])
    return np.hstack([M, cdum])


def fit_gbm_per_hour(train_df, features):
    out = {}
    countries = sorted(train_df["country"].unique())
    for h in range(24):
        sub = train_df[train_df["hour"] == h].dropna(subset=["deviation"] + features)
        if len(sub) < len(features) + len(countries) + 5:
            continue
        X = gbm_features_matrix(sub, features, countries)
        y = sub["deviation"].astype(float).values
        model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=200, learning_rate=0.05,
            min_samples_leaf=10, random_state=42,
        ).fit(X, y)
        out[h] = {"model": model, "countries": countries, "features": features}
    return out


def predict_gbm(model_h, df_test):
    M = gbm_features_matrix(df_test, model_h["features"], model_h["countries"])
    return model_h["model"].predict(M)


# ---------------------------------------------------------------------------
# LOO across years for every spec
# ---------------------------------------------------------------------------
def loo_for_spec(name, features, fitter_kind):
    print(f"  LOO for {name} ({len(features)} features) ...")
    rows = []
    years = sorted(panel["year"].unique())
    for y0 in years:
        tr = panel[panel["year"] != y0]
        te = panel[panel["year"] == y0]
        if fitter_kind == "ols":
            mods = fit_ols_per_hour(tr, features)
            for _, r in te.iterrows():
                h = int(r["hour"])
                if h not in mods or r["country"] not in mods[h]["countries"]:
                    continue
                p = predict_ols(mods[h], r["country"], r, features)
                rows.append({"spec": name, "country": r["country"], "year": int(y0),
                             "hour": h, "observed": float(r["deviation"]),
                             "predicted": p,
                             "residual": float(r["deviation"]) - p})
        elif fitter_kind == "gbm":
            mods = fit_gbm_per_hour(tr, features)
            for h, m in mods.items():
                te_h = te[(te["hour"] == h) & (te["country"].isin(m["countries"]))]
                if te_h.empty: continue
                preds = predict_gbm(m, te_h)
                for (_, r), p in zip(te_h.iterrows(), preds):
                    rows.append({"spec": name, "country": r["country"], "year": int(y0),
                                 "hour": h, "observed": float(r["deviation"]),
                                 "predicted": float(p),
                                 "residual": float(r["deviation"]) - float(p)})
    return pd.DataFrame(rows)


all_loo = []
for name, feats in SPECS.items():
    kind = "gbm" if name == "S5_gbm" else "ols"
    all_loo.append(loo_for_spec(name, feats, kind))
loo = pd.concat(all_loo, ignore_index=True)
loo.to_csv(RESULTS / "nonlinearity_test_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def overall(df_loo):
    ss_res = (df_loo["residual"] ** 2).sum()
    ss_tot = ((df_loo["observed"] - df_loo["observed"].mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot,
            "rmse": float(np.sqrt((df_loo["residual"] ** 2).mean())),
            "n": int(len(df_loo))}


def per_hour(df_loo):
    rows = []
    for h, g in df_loo.groupby("hour"):
        ss_res = (g["residual"] ** 2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
        rows.append({"hour": int(h),
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rmse": float(np.sqrt((g["residual"] ** 2).mean()))})
    return pd.DataFrame(rows).sort_values("hour")


summary = {name: overall(loo[loo["spec"] == name]) for name in SPECS}
per_h   = {name: per_hour(loo[loo["spec"] == name]) for name in SPECS}


# ---------------------------------------------------------------------------
# Coefficient inspection for S2/S3/S4 (does quad/interaction reach significance?)
# ---------------------------------------------------------------------------
print("\nFitting full-panel models for coefficient inspection ...")
full_fits = {name: fit_ols_per_hour(panel, feats)
             for name, feats in SPECS.items() if name != "S5_gbm"}


def quad_int_significance(model_dict, term_list):
    rows = []
    for h, m in model_dict.items():
        # Fit with t-stats: re-run with full result
        countries = m["countries"]
        sub = panel[panel["hour"] == h].dropna(subset=["deviation"] + m["x_cols"][1+len(countries)-1:])
        X = pd.DataFrame(index=sub.index); X["const"] = 1.0
        for c in countries[1:]:
            X[f"FE_{c}"] = (sub["country"].values == c).astype(float)
        feats = [c for c in m["x_cols"] if c not in (["const"] + [f"FE_{c}" for c in countries[1:]])]
        for f in feats:
            X[f] = sub[f].astype(float).values
        y = sub["deviation"].astype(float)
        res = sm.OLS(y, X).fit()
        for t in term_list:
            if t in res.params.index:
                rows.append({"hour": h, "term": t,
                             "coef": float(res.params[t]),
                             "t": float(res.tvalues[t]),
                             "p": float(res.pvalues[t])})
    return pd.DataFrame(rows)


sig_S2 = quad_int_significance(full_fits["S2_quadratic"], QUAD_FEATS)
sig_S3 = quad_int_significance(full_fits["S3_interactions"], INT_FEATS)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("Non-linearity test: linear vs quadratic vs interactions vs GBM")
out.append("=" * 78)
out.append(f"\nPanel: {len(panel)} rows; {panel['country'].nunique()} countries; "
           f"years {panel['year'].min()}-{panel['year'].max()}")

out.append("\nOverall LOO performance:")
out.append(f"{'spec':<20s} {'k_feat':>7} {'R2':>8} {'RMSE':>8}  {'dR2 vs S1':>11}  {'dRMSE vs S1':>13}")
out.append("-" * 78)
ref = summary["S1_linear"]
for name, feats in SPECS.items():
    o = summary[name]
    k = len(feats)
    if name == "S5_gbm": k = f"{k} (trees)"
    dR2 = o["r2"] - ref["r2"]
    dRMSE = o["rmse"] - ref["rmse"]
    out.append(f"{name:<20s} {str(k):>7} {o['r2']:>8.3f} {o['rmse']:>8.2f}  "
               f"{dR2:>+11.4f}  {dRMSE:>+13.4f}")

out.append("\nPer-hour LOO R^2 across specs:")
out.append(f"{'h':>3}  {'S1_lin':>7}  {'S2_quad':>8}  {'S3_int':>7}  {'S4_full':>8}  {'S5_gbm':>7}")
out.append("-" * 78)
for h in range(24):
    s1 = per_h["S1_linear"].set_index("hour").get("r2", pd.Series()).get(h, np.nan)
    s2 = per_h["S2_quadratic"].set_index("hour").get("r2", pd.Series()).get(h, np.nan)
    s3 = per_h["S3_interactions"].set_index("hour").get("r2", pd.Series()).get(h, np.nan)
    s4 = per_h["S4_full_nonlin"].set_index("hour").get("r2", pd.Series()).get(h, np.nan)
    s5 = per_h["S5_gbm"].set_index("hour").get("r2", pd.Series()).get(h, np.nan)
    out.append(f"{h:>3}  {s1:>7.3f}  {s2:>8.3f}  {s3:>7.3f}  {s4:>8.3f}  {s5:>7.3f}")

out.append("\nQuadratic term significance (S2 full-panel fit):")
out.append(f"{'h':>3}  {'term':<22}  {'coef':>10}  {'t':>7}  {'p':>7}")
out.append("-" * 78)
for _, r in sig_S2.iterrows():
    out.append(f"{int(r['hour']):>3}  {r['term']:<22}  {r['coef']:>+10.2f}  "
               f"{r['t']:>+7.2f}  {r['p']:>7.4f}")

out.append("\nInteraction term significance (S3 full-panel fit):")
out.append(f"{'h':>3}  {'term':<22}  {'coef':>10}  {'t':>7}  {'p':>7}")
out.append("-" * 78)
for _, r in sig_S3.iterrows():
    out.append(f"{int(r['hour']):>3}  {r['term']:<22}  {r['coef']:>+10.2f}  "
               f"{r['t']:>+7.2f}  {r['p']:>7.4f}")

# Count significant terms
sig2_count = (sig_S2["p"] < 0.05).sum()
sig3_count = (sig_S3["p"] < 0.05).sum()
out.append(f"\nQuadratic terms significant (p<0.05): {sig2_count}/{len(sig_S2)}")
out.append(f"Interaction terms significant (p<0.05): {sig3_count}/{len(sig_S3)}")

# Interpretation
gbm_gain = summary["S5_gbm"]["r2"] - summary["S1_linear"]["r2"]
out.append("\n" + "=" * 78)
out.append("INTERPRETATION:")
if gbm_gain > 0.03:
    out.append(f"  GBM beats linear by dR2 = {gbm_gain:+.3f} -> non-linearity matters")
    out.append("  Consider keeping a polynomial / interaction spec or moving to GBM.")
elif gbm_gain > 0.01:
    out.append(f"  GBM beats linear by dR2 = {gbm_gain:+.3f} -> modest non-linearity.")
    out.append("  Linear is mostly fine; add targeted quadratic/interactions if cheap.")
else:
    out.append(f"  GBM beats linear by dR2 = {gbm_gain:+.3f} -> ESSENTIALLY NO NON-LINEARITY")
    out.append("  Linear specification is justified; flexibility doesn't help.")

text = "\n".join(out)
print()
print(text)
(RESULTS / "nonlinearity_test_summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))
colors = {"S1_linear": "#999999", "S2_quadratic": "#1f77b4",
          "S3_interactions": "#2ca02c", "S4_full_nonlin": "#ff7f0e",
          "S5_gbm": "#d62728"}
for name in SPECS:
    df = per_h[name]
    ax.plot(df["hour"], df["r2"], color=colors[name], marker="o", markersize=3,
            linewidth=1.2, label=name)
ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("LOO R^2")
ax.set_xticks(range(0, 24, 3))
ax.set_title("Per-hour LOO R^2: does non-linearity beat linear?")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=7, ncol=2)
plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "nonlinearity_test_r2_by_hour.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "nonlinearity_test_r2_by_hour.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote: {RESULTS / 'nonlinearity_test_summary.txt'}")
print(f"       {PLOTS / 'nonlinearity_test_r2_by_hour.pdf'}")
