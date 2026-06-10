"""
Show -- for each hour of day, for each technology -- how it shifts the
deviation profile. Two complementary views:

  Panel A: Heatmap (tech x hour). Color = standardized coefficient
           (beta_t^h * std(capnorm_t across panel)). Standardization makes
           techs comparable on a single scale.

  Panel B: Small-multiples, one panel per tech. Bar chart of the
           standardized coefficient at each hour. Same information as the
           heatmap row, with explicit y-axis to read magnitudes.

Output:
  Analysis/results/plots/hourly_tech_shifts.{pdf,png}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

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

COEFS_FILE = RESULTS / "predict_deviation_capnorm_coefs.csv"
STATE_FILE = RESULTS / "system_state_capnorm_panel.csv"

# Display set: drop lignite, oil, biomass (small-N or strong collinearity, noisy).
# The regression that produced these coefficients STILL controls for them;
# we only suppress them from the visualisation.
DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "nuclear", "gas", "hardcoal"]
TECH_LABEL = {
    "solar": "Solar", "onwind": "Onshore wind", "offwind": "Offshore wind",
    "hydro_disp": "Hydro (reservoir+pumped)", "hydro_ror": "Hydro (RoR)",
    "nuclear": "Nuclear", "gas": "Gas", "hardcoal": "Hard coal",
}

coefs = pd.read_csv(COEFS_FILE)
state = pd.read_csv(STATE_FILE)
stds = {t: state[f"capnorm_{t}"].std() for t in DIFF_TECHS}

# Build matrix: rows = techs, cols = hours, values = standardized coefficient
M = np.zeros((len(DIFF_TECHS), 24))
for i, t in enumerate(DIFF_TECHS):
    sub = coefs[coefs["predictor"] == f"capnorm_{t}"].sort_values("hour")
    for _, r in sub.iterrows():
        M[i, int(r["hour"])] = r["coef"] * stds[t]


# ---------------------------------------------------------------------------
# Panel A: Heatmap
# ---------------------------------------------------------------------------
vmax = float(np.max(np.abs(M)))
fig, (ax_hm, ax_sm) = plt.subplots(
    2, 1, figsize=(8.0, 7.5), constrained_layout=True,
    gridspec_kw={"height_ratios": [1.1, 2.4]},
)

cmap = "RdBu_r"
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax_hm.imshow(M, aspect="auto", cmap=cmap, norm=norm)
ax_hm.set_yticks(range(len(DIFF_TECHS)))
ax_hm.set_yticklabels([TECH_LABEL[t] for t in DIFF_TECHS])
ax_hm.set_xticks(range(0, 24, 3))
ax_hm.set_xticklabels(range(0, 24, 3))
ax_hm.set_xlabel("Hour of day")
ax_hm.set_title(
    "How each technology shifts the within-day price deviation profile  "
    "(standardized: €/MWh per +1 std of cap/peak_load)",
    fontsize=8.5,
)
# Numbers in cells
for i in range(len(DIFF_TECHS)):
    for j in range(24):
        v = M[i, j]
        color = "white" if abs(v) > 0.55 * vmax else "black"
        ax_hm.text(j, i, f"{v:+.0f}", ha="center", va="center",
                   fontsize=5.5, color=color)
cb = fig.colorbar(im, ax=ax_hm, fraction=0.030, pad=0.01)
cb.set_label("€/MWh per +1 std capnorm", fontsize=7)
cb.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Panel B: Small multiples (one per tech)
# ---------------------------------------------------------------------------
NCOLS = 4
NROWS = int(np.ceil(len(DIFF_TECHS) / NCOLS))
gs = ax_sm.get_subplotspec().subgridspec(NROWS, NCOLS, hspace=0.5, wspace=0.3)
ax_sm.axis("off")

# Common y-limits
yabs = float(np.max(np.abs(M))) * 1.1

PALETTE = {"solar":"#fdb863","onwind":"#5e3c99","offwind":"#3690c0",
           "hydro_disp":"#0570b0","hydro_ror":"#74a9cf","biomass":"#33a02c",
           "nuclear":"#e41a1c","gas":"#fb9a99","hardcoal":"#525252",
           "lignite":"#8c510a","oil":"#bf812d"}

for i, t in enumerate(DIFF_TECHS):
    r, c = divmod(i, NCOLS)
    ax = fig.add_subplot(gs[r, c])
    bars = ax.bar(range(24), M[i, :], color=PALETTE[t], width=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-yabs, yabs)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xticklabels([0, 6, 12, 18], fontsize=6.5)
    ax.set_title(TECH_LABEL[t], fontsize=7.5, fontweight="bold", pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    # Annotate peak / trough
    h_max = int(np.argmax(M[i, :]))
    h_min = int(np.argmin(M[i, :]))
    ax.annotate(f"+{M[i, h_max]:.0f}", xy=(h_max, M[i, h_max]),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, color="black")
    ax.annotate(f"{M[i, h_min]:.0f}", xy=(h_min, M[i, h_min]),
                xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=6, color="black")
    if c == 0:
        ax.set_ylabel("€/MWh", fontsize=6.5)
    if r == NROWS - 1:
        ax.set_xlabel("Hour", fontsize=6.5)

fig.suptitle(
    "Per-hour structural-shape contribution by technology   "
    "(positive = lifts price relative to annual mean; negative = depresses)",
    fontsize=9, y=1.01, fontweight="bold",
)

fig.savefig(PLOTS / "hourly_tech_shifts.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "hourly_tech_shifts.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote: {PLOTS / 'hourly_tech_shifts.pdf'}")
print(f"       {PLOTS / 'hourly_tech_shifts.png'}")
