"""
Produce a corrected version of the "Pooled tech-slope profiles" plot
(predict_deviation_pooled_capacity_coefs.pdf).

The raw coefficient β_t^(h) is in €/MWh per +1.0 share. Comparing techs
on this scale is misleading: techs with tiny share variance (biomass ~2pp std)
need very large coefficients to register, while high-variance techs
(solar ~10pp std) appear "flat" even when their realised market impact is large.

The standardized counterpart is the predicted price-shape effect from a
typical (1 standard deviation) change in that tech's share:

    standardized_t^(h) = beta_t^(h) * std(capshare_t across all c, y)

This makes techs comparable on a single scale ("€/MWh per typical-magnitude
shift in tech t's share").

Outputs:
  Analysis/results/plots/standardized_tech_slopes.{pdf,png}
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
    "legend.fontsize": 7, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"

COEFS = RESULTS / "predict_deviation_pooled_capacity_coefs.csv"
STATE = RESULTS / "system_state_capacity_panel.csv"

DIFF_TECHS = ["solar", "onwind", "offwind", "hydro_disp", "hydro_ror",
              "biomass", "nuclear", "gas", "hardcoal", "lignite", "oil"]
PALETTE = {
    "solar": "#fdb863", "onwind": "#5e3c99", "offwind": "#3690c0",
    "hydro_disp": "#0570b0", "hydro_ror": "#74a9cf", "biomass": "#33a02c",
    "nuclear": "#e41a1c", "gas": "#fb9a99", "hardcoal": "#525252",
    "lignite": "#8c510a", "oil": "#bf812d",
}

coefs = pd.read_csv(COEFS)
state = pd.read_csv(STATE)

stds = {t: state[f"capshare_{t}"].std() for t in DIFF_TECHS}
print("Capacity-share standard deviations:")
for t in DIFF_TECHS:
    print(f"  {t:<12s}: std = {stds[t]:.4f}")


# ---------------------------------------------------------------------------
# Two-panel figure: raw vs standardized, side by side
# ---------------------------------------------------------------------------
fig, (ax_raw, ax_std) = plt.subplots(1, 2, figsize=(8.5, 3.4),
                                     constrained_layout=True)

for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capshare_{t}"].sort_values("hour")
    if sub.empty:
        continue
    # raw
    ax_raw.plot(sub["hour"], sub["coef"], color=PALETTE[t], linewidth=1.2,
                marker="o", markersize=2.8, label=t)
    # standardized
    ax_std.plot(sub["hour"], sub["coef"] * stds[t], color=PALETTE[t],
                linewidth=1.2, marker="o", markersize=2.8, label=t)

for ax, title, ylabel in [
    (ax_raw, "Raw tech-slope profiles\n(€/MWh per +1.0 share)",
     "$\\beta_t^{(h)}$ (€/MWh per +1.0 share)"),
    (ax_std, "Standardized tech-slope profiles\n(€/MWh per +1 std of share)",
     "$\\beta_t^{(h)} \\times \\sigma(\\mathrm{share}_t)$  (€/MWh)"),
]:
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)

# Single legend below both
handles, labels = ax_raw.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=11,
           frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=7)
fig.savefig(PLOTS / "standardized_tech_slopes.pdf", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS / "standardized_tech_slopes.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Also dump the numbers for inspection
# ---------------------------------------------------------------------------
out = []
out.append("Standardized tech-slope profiles  (€/MWh per +1 std of share):")
out.append(f"{'h':>3}  " + "  ".join(f"{t:>9s}" for t in DIFF_TECHS))
out.append("-" * 130)
for h in range(24):
    row = [f"{h:>3}"]
    for t in DIFF_TECHS:
        sub = coefs[(coefs["predictor"] == f"capshare_{t}") & (coefs["hour"] == h)]
        if sub.empty:
            row.append(f"{'NA':>9}")
        else:
            row.append(f"{sub['coef'].iloc[0] * stds[t]:>+9.2f}")
    out.append("  ".join(row))

# Peak-trough across hours per tech (standardized)
out.append("\nPeak-to-trough range of standardized coefficient (€/MWh):")
for t in DIFF_TECHS:
    sub = coefs[coefs["predictor"] == f"capshare_{t}"]
    if sub.empty:
        continue
    std_coef = sub["coef"].values * stds[t]
    out.append(f"  {t:<12s}: max-min = {std_coef.max() - std_coef.min():.2f}  "
               f"(max={std_coef.max():+.2f} at h{int(sub.loc[sub['coef'].idxmax(), 'hour']):02d}, "
               f"min={std_coef.min():+.2f} at h{int(sub.loc[sub['coef'].idxmin(), 'hour']):02d})")

text = "\n".join(out)
print()
print(text)
(RESULTS / "standardized_tech_slopes.txt").write_text(text, encoding="utf-8")
print(f"\nWrote: {PLOTS / 'standardized_tech_slopes.pdf'}")
