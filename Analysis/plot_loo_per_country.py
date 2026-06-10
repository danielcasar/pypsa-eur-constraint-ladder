"""
Per-country small-multiples plot of LOO predicted vs observed HOD deviation
profile, one panel per held-out year. Same layout as predict_deviation_per_hour_v2_DE_loo.pdf
but for every country and using the current pooled-FE capacity-share model.

Reads the LOO predictions written by predict_deviation_pooled_capacity.py.

Outputs (one PDF per country):
  Analysis/results/plots/loo_per_country/loo_<COUNTRY>.pdf
  Analysis/results/plots/loo_per_country/loo_<COUNTRY>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 8, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS_OUT = RESULTS / "plots" / "loo_per_country"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

LOO_FILE = RESULTS / "predict_deviation_pooled_capacity_loo_predictions.csv"

loo = pd.read_csv(LOO_FILE)
print(f"LOO predictions: {len(loo)} rows; "
      f"countries: {loo['country'].nunique()}; "
      f"years: {sorted(loo['year'].unique())}")


def loo_r2(g: pd.DataFrame) -> float:
    ss_res = (g["residual"] ** 2).sum()
    ss_tot = ((g["observed"] - g["observed"].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(g: pd.DataFrame) -> float:
    return float(np.sqrt((g["residual"] ** 2).mean()))


for country in sorted(loo["country"].unique()):
    sub = loo[loo["country"] == country]
    years = sorted(sub["year"].unique())
    NCOLS = 4
    NROWS = int(np.ceil(len(years) / NCOLS))
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(6.67, 1.0 * NROWS + 0.6),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten() if NROWS * NCOLS > 1 else [axes]

    overall_r2 = loo_r2(sub)
    overall_rm = rmse(sub)

    for i, yr in enumerate(years):
        ax = axes[i]
        s = sub[sub["year"] == yr].sort_values("hour")
        ax.plot(s["hour"], s["observed"], color="black", linewidth=1.2,
                marker="o", markersize=2.5, label="observed")
        ax.plot(s["hour"], s["predicted"], color="#d62728", linewidth=1.2,
                marker="s", markersize=2.5, linestyle="--",
                label="LOO prediction")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
        rm_y = rmse(s)
        ax.set_title(f"{country} {yr}  (RMSE {rm_y:.1f})", fontsize=7, pad=2)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(years), len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel("Hour of day", fontsize=8, y=0.02)
    fig.supylabel("Deviation from annual mean (€/MWh)", fontsize=8, x=0.005)
    fig.suptitle(f"{country} — LOO predicted vs observed HOD deviation "
                 f"(pooled-FE, capacity shares; "
                 f"all-years LOO R² = {overall_r2:.2f}, RMSE = {overall_rm:.1f} €/MWh)",
                 fontsize=8, y=1.005)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.5)

    out_pdf = PLOTS_OUT / f"loo_{country}.pdf"
    out_png = PLOTS_OUT / f"loo_{country}.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {country}: {len(years)} years, LOO R² = {overall_r2:.3f}, "
          f"RMSE = {overall_rm:.2f}  -> {out_pdf.name}")

print(f"\nWrote {len(loo['country'].unique())} per-country plots to {PLOTS_OUT}")
