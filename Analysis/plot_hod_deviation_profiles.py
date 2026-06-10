"""
Visualise year-by-year evolution of the HOD mean-deviation profile.

Two figures:
  1. Single panel for Germany: 8 year-curves overlaid.
  2. 4x6 small-multiples panel: one subplot per country.

Inputs:
  Analysis/results/hod_deviation_profiles_long.csv  (built by hod_deviation_profiles.py)

Outputs:
  Analysis/results/plots/hod_deviation_profile_DE.{pdf,png}
  Analysis/results/plots/hod_deviation_profile_grid.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daniel's house style: 8pt everywhere, 6.67in width
matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.titlesize": 8,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":  1.0,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "hod_deviation_profiles_long.csv"
df = pd.read_csv(INPUT)

YEARS = sorted(df["year"].unique())
COUNTRIES = sorted(df["country"].unique())

# Sequential colormap year-coded; darker = earlier
cmap = matplotlib.colormaps["viridis"]
year_colors = {y: cmap(i / (len(YEARS) - 1)) for i, y in enumerate(YEARS)}


# ---------------------------------------------------------------------------
# Figure 1: DE single-panel
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.67, 3.0))

de = df[df["country"] == "DE"]
for year in YEARS:
    sub = de[de["year"] == year].sort_values("hour")
    ax.plot(sub["hour"], sub["deviation"],
            color=year_colors[year], label=str(year),
            linewidth=1.2,
            marker="o", markersize=2.5,
            zorder=10 if year in (2018, 2025) else 5)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6, zorder=0)
ax.set_xlabel("Hour of day", fontsize=8)
ax.set_ylabel("Mean deviation from annual mean (€/MWh)", fontsize=8)
ax.set_title("Germany — emergence of the duck curve in HOD deviation profile",
             fontsize=8)
ax.set_xticks(range(0, 24, 3))
ax.set_xlim(-0.5, 23.5)
ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="Year", loc="upper left", ncol=2, frameon=False,
          handlelength=1.2, columnspacing=1.0)

plt.tight_layout(pad=0.4)
fig.savefig(PLOTS / "hod_deviation_profile_DE.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "hod_deviation_profile_DE.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PLOTS / 'hod_deviation_profile_DE.pdf'}")


# ---------------------------------------------------------------------------
# Figure 2: 22-country small multiples (4 x 6 grid)
# ---------------------------------------------------------------------------
NCOLS, NROWS = 6, 4
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(6.67, 4.6),
                         sharex=True,
                         constrained_layout=True)

axes = axes.flatten()

# Get global y-range so all panels share comparable scale
ymin = df["deviation"].quantile(0.005)   # exclude wild outliers (DE 2022 etc.)
ymax = df["deviation"].quantile(0.995)

for i, country in enumerate(COUNTRIES):
    ax = axes[i]
    sub_c = df[df["country"] == country]
    for year in YEARS:
        sub = sub_c[sub_c["year"] == year].sort_values("hour")
        ax.plot(sub["hour"], sub["deviation"],
                color=year_colors[year], linewidth=0.7, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--",
               alpha=0.5, zorder=0)
    ax.set_title(country, fontsize=8, pad=2)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.tick_params(axis="both", which="both", length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    if i // NCOLS != NROWS - 1 and i + NCOLS < len(COUNTRIES):
        ax.tick_params(labelbottom=False)

# Hide the unused trailing axes
for j in range(len(COUNTRIES), len(axes)):
    axes[j].set_visible(False)

# Shared x-label in the last visible row
fig.supxlabel("Hour of day", fontsize=8, y=0.02)
fig.supylabel("Mean deviation from annual mean (€/MWh)", fontsize=8, x=0.005)

# Year legend at the bottom (uses dummy lines for the colormap bar)
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], color=year_colors[y], linewidth=1.2,
                         label=str(y)) for y in YEARS]
fig.legend(handles=handles, ncol=len(YEARS), loc="lower center",
           bbox_to_anchor=(0.5, -0.04), frameon=False,
           handlelength=1.2, columnspacing=1.0, fontsize=8)

fig.suptitle("HOD mean-deviation profile, 22 European markets, 2018–2025",
             fontsize=8, y=1.005)

fig.savefig(PLOTS / "hod_deviation_profile_grid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "hod_deviation_profile_grid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PLOTS / 'hod_deviation_profile_grid.pdf'}")
