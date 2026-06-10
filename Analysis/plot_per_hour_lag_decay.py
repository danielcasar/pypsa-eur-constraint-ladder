"""
For each hour-of-day h, plot the year-lag decay of the deviation:

    Delta_dev_h(c, y1, y2) = dev_h(c, y2) - dev_h(c, y1)

x-axis: lag = y2 - y1
y-axis: Delta_dev_h (EUR/MWh) -- median across (country, year-pair) at each lag
        with q25/q75 shaded band

24 panels in a 4x6 grid. Reproduces the structure of Paper 1's Figure 8
(median Pearson rho vs lag, with IQR), but conditioned on a single hour.

Inputs:
  Analysis/results/hod_deviation_profiles_long.csv

Outputs:
  Analysis/results/plots/per_hour_lag_decay.{pdf,png}
  Analysis/results/per_hour_lag_decay.csv  (long: hour, lag, median, q25, q75)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

INPUT = RESULTS / "hod_deviation_profiles_long.csv"
df = pd.read_csv(INPUT)

# ---------------------------------------------------------------------------
# Build all (country, h, y1, y2) Delta_dev rows
# ---------------------------------------------------------------------------
rows = []
for (country, hour), sub in df.groupby(["country", "hour"]):
    sub = sub.sort_values("year")
    years = sub["year"].tolist()
    devs  = dict(zip(years, sub["deviation"]))
    for i, y1 in enumerate(years):
        for y2 in years[i + 1:]:
            rows.append({
                "country": country, "hour": int(hour),
                "year1": int(y1), "year2": int(y2),
                "lag": int(y2 - y1),
                "delta_dev": float(devs[y2] - devs[y1]),
            })
pairs = pd.DataFrame(rows)

# Per (hour, lag) summary across country x year-pair observations
summary = (pairs.groupby(["hour", "lag"])["delta_dev"]
           .agg(median="median",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
                n="count")
           .reset_index().sort_values(["hour", "lag"]))
summary.to_csv(RESULTS / "per_hour_lag_decay.csv", index=False)

# ---------------------------------------------------------------------------
# Plot: 4 rows x 6 cols, one panel per hour
# ---------------------------------------------------------------------------
NCOLS, NROWS = 6, 4
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(6.67, 4.6),
                         sharex=True, sharey=True,
                         constrained_layout=True)
axes = axes.flatten()

# Symmetric global y-range so all panels share the same scale
y_max = max(abs(summary["q25"].min()), abs(summary["q75"].max())) * 1.05

for h in range(24):
    ax = axes[h]
    sub = summary[summary["hour"] == h].sort_values("lag")
    ax.fill_between(sub["lag"], sub["q25"], sub["q75"],
                    color="#1f77b4", alpha=0.20, linewidth=0)
    ax.plot(sub["lag"], sub["median"],
            color="#1f77b4", linewidth=1.2,
            marker="o", markersize=2.5)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--",
               alpha=0.5, zorder=0)
    ax.set_title(f"h = {h:02d}", fontsize=8, pad=2)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(-y_max, y_max)
    ax.tick_params(axis="both", which="both", length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    if h % NCOLS != 0:
        ax.tick_params(labelleft=False)
    if h // NCOLS != NROWS - 1:
        ax.tick_params(labelbottom=False)

fig.supxlabel("Year lag (years)", fontsize=8, y=0.02)
fig.supylabel("Δ deviation = dev$_h$(y$_2$) − dev$_h$(y$_1$) [€/MWh]",
              fontsize=8, x=0.005)
fig.suptitle("Year-lag decay per hour-of-day (median ± IQR across "
             "country × year-pairs)", fontsize=8, y=1.005)

fig.savefig(PLOTS / "per_hour_lag_decay.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "per_hour_lag_decay.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PLOTS / 'per_hour_lag_decay.pdf'}")
print(f"Wrote {RESULTS / 'per_hour_lag_decay.csv'}")
