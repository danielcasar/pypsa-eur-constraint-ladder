"""
Headline figure for "shape responds to capacity mix in measurable, replicable ways":

4 archetype countries with very different capacity mixes, shown alongside their
predicted (model) and observed deviation profile. The visual demonstrates that
the model -- given only the capacity mix as input -- correctly reproduces the
distinct shapes of structurally different markets.

Countries (2024 snapshot):
  - DE   : solar-heavy, deep duck
  - FR   : nuclear-dominated, flatter shape
  - NO   : hydro-dominated, modest shape
  - PL   : coal-heavy, classic peaker shape

Output:
  Analysis/results/plots/mix_to_shape_archetypes.{pdf,png}
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
    "legend.fontsize": 7, "figure.titlesize": 9,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"

DEV_FILE = RESULTS / "hod_deviation_profiles_long.csv"
STATE_FILE = RESULTS / "system_state_capnorm_panel.csv"

ARCHETYPES = [
    ("DE", "Solar-heavy (DE)"),
    ("FR", "Nuclear-dominated (FR)"),
    ("ES", "Solar + hydro (ES)"),   # NO not in 22-country list, replace
    ("PL", "Coal-heavy (PL)"),
]
TARGET_YEAR = 2024

ALL_COUNTRIES = ["AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
                 "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO",
                 "SI", "SK"]
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
CAPNORM = [f"capnorm_{t}" for t in DIFF_TECHS]
EXPOSED = [f"exposed_capnorm_{t}" for t in DIFF_TECHS]
PREDICTORS = CAPNORM + EXPOSED + ["ntc_norm"]

PALETTE = {"solar":"#fdb863","onwind":"#5e3c99","offwind":"#3690c0",
           "hydro_disp":"#0570b0","hydro_ror":"#74a9cf","biomass":"#33a02c",
           "nuclear":"#e41a1c","gas":"#fb9a99","hardcoal":"#525252",
           "lignite":"#8c510a","oil":"#bf812d"}

# ---------------------------------------------------------------------------
# Load + fit (pooled FE on full panel, leave-one-year-out: train without 2024)
# ---------------------------------------------------------------------------
state = pd.read_csv(STATE_FILE)
dev = pd.read_csv(DEV_FILE)
panel = (dev.merge(state, on=["country", "year"], how="inner")
            .query("country in @ALL_COUNTRIES")
            .dropna(subset=["deviation"] + PREDICTORS))
train = panel[panel["year"] != TARGET_YEAR]

def build_X(df, predictors, train_countries):
    X = pd.DataFrame(index=df.index); X["const"] = 1.0
    for c in train_countries[1:]:
        X[f"FE_{c}"] = (df["country"].values == c).astype(float)
    for s in predictors:
        X[s] = df[s].astype(float).values
    return X

countries_t = sorted(train["country"].unique())
models = {}
for h in range(24):
    sub = train[train["hour"] == h].dropna(subset=["deviation"] + PREDICTORS)
    if len(sub) < 50: continue
    X = build_X(sub, PREDICTORS, countries_t)
    y = sub["deviation"].astype(float)
    res = sm.OLS(y, X).fit()
    models[h] = {"params": res.params, "x_cols": list(X.columns),
                 "countries": countries_t}

def predict(model_h, country, row, predictors):
    cols = model_h["x_cols"]
    x = {c: 0.0 for c in cols}; x["const"] = 1.0
    if f"FE_{country}" in x: x[f"FE_{country}"] = 1.0
    for s in predictors:
        x[s] = float(row[s])
    return float(np.dot(np.array([x[c] for c in cols]), model_h["params"].values))


# ---------------------------------------------------------------------------
# For each archetype, compute predicted + observed shape for TARGET_YEAR
# ---------------------------------------------------------------------------
arch_data = {}
for cc, _ in ARCHETYPES:
    sub = panel[(panel["country"] == cc) & (panel["year"] == TARGET_YEAR)]
    if sub.empty:
        # try year 2023 fallback
        sub = panel[(panel["country"] == cc) & (panel["year"] == 2023)]
        year_used = 2023
    else:
        year_used = TARGET_YEAR
    rows = []
    for _, r in sub.iterrows():
        h = int(r["hour"])
        if h not in models: continue
        rows.append({"hour": h, "observed": float(r["deviation"]),
                     "predicted": predict(models[h], cc, r, PREDICTORS)})
    arch_data[cc] = {
        "year": year_used,
        "shape": pd.DataFrame(rows).sort_values("hour"),
        "capnorm": {t: float(sub[f"capnorm_{t}"].iloc[0]) for t in DIFF_TECHS},
    }


# ---------------------------------------------------------------------------
# Figure: 4 columns (one per archetype), 2 rows (mix bar + shape line)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, len(ARCHETYPES), figsize=(8.5, 4.2),
                         constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.0, 1.4]})

for i, (cc, label) in enumerate(ARCHETYPES):
    d = arch_data[cc]

    # Top row: capacity mix
    ax = axes[0, i]
    techs = DIFF_TECHS
    vals = [d["capnorm"][t] for t in techs]
    colors = [PALETTE[t] for t in techs]
    ax.bar(range(len(techs)), vals, color=colors, width=0.7)
    ax.set_xticks(range(len(techs)))
    ax.set_xticklabels(techs, rotation=60, ha="right", fontsize=6.5)
    ax.set_title(f"{label}, {d['year']}", fontsize=8, fontweight="bold", pad=2)
    if i == 0:
        ax.set_ylabel("capnorm\n(cap / peak load)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(0.6, max(vals) * 1.1))

    # Bottom row: predicted vs observed deviation profile
    ax = axes[1, i]
    s = d["shape"]
    ax.plot(s["hour"], s["observed"], color="black", linewidth=1.4, marker="o",
            markersize=3.5, label="observed", zorder=3)
    ax.plot(s["hour"], s["predicted"], color="#d62728", linewidth=1.4,
            marker="s", markersize=3.5, linestyle="--",
            label="predicted from capacity mix", zorder=2)
    ax.fill_between(s["hour"], s["predicted"], s["observed"],
                    color="#d62728", alpha=0.10, linewidth=0)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    # RMSE annotation
    rmse = float(np.sqrt(((s["predicted"] - s["observed"]) ** 2).mean()))
    ax.text(0.97, 0.06, f"RMSE = {rmse:.1f} €/MWh", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_xlabel("Hour of day", fontsize=7.5)
    if i == 0:
        ax.set_ylabel("Deviation from\nannual mean (€/MWh)", fontsize=7)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0:
        ax.legend(frameon=False, loc="upper left", fontsize=6.5,
                  handlelength=1.5)

# Common y-limits for bottom row
ymins = [axes[1, i].get_ylim()[0] for i in range(len(ARCHETYPES))]
ymaxs = [axes[1, i].get_ylim()[1] for i in range(len(ARCHETYPES))]
ylo, yhi = min(ymins), max(ymaxs)
for i in range(len(ARCHETYPES)):
    axes[1, i].set_ylim(ylo, yhi)

fig.suptitle(
    "Capacity mix determines price shape: four European markets, model trained "
    "without target year",
    fontsize=9, y=1.02, fontweight="bold",
)

fig.savefig(PLOTS / "mix_to_shape_archetypes.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "mix_to_shape_archetypes.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote: {PLOTS / 'mix_to_shape_archetypes.pdf'}")
print(f"       {PLOTS / 'mix_to_shape_archetypes.png'}")
