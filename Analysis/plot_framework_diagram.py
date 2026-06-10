"""
Framework figure for the Electricity Price Evolution research programme.

Layout follows Daniel's hand-drawn sketch:

  Historical electricity prices
            │  (technology shares, country, NTC, neighbour exposure)
            ↓
  ┌────────────────────────┐         ┌──────────────────────────┐
  │  Hourly mean           │ ←────── │ ESOM                     │
  │  deviation model       │  new-   │ (PyPSA-Eur, GENeSYS-MOD) │
  │                        │  tech   │                          │
  └─────────┬──────────────┘  preds  └─────────────┬────────────┘
            │            (BESS, electrolyzers,…)   │ (shadow prices)
            ↓                                      ↓
  ┌────────────────────────┐                ┌─────────────────────┐
  │ Future mean hourly     │ ─ validation → │ Future hourly       │
  │ price structure        │                │ electricity prices  │
  └────────────────────────┘                └─────────────────────┘

The ESOM has a DUAL role:
  (1) supplies coefficients/predictors for new technologies that the empirical
      regression cannot identify from history
  (2) independently produces shadow prices for future scenarios
The validation step compares the two future outputs -> model artefact catalogue.

Outputs:
  Analysis/results/plots/framework_diagram.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8,
    "axes.linewidth": 0.0,
})

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COL_HIST_F     = "#f4e8d8"; COL_HIST_E     = "#a8826a"
COL_EMP_F      = "#e2ecf6"; COL_EMP_E      = "#1f5fa8"
COL_ESOM_F     = "#fde7e7"; COL_ESOM_E     = "#b22222"
COL_FUT_EMP_F  = "#cfe1f1"; COL_FUT_EMP_E  = "#1f5fa8"
COL_FUT_ESOM_F = "#f5cccc"; COL_FUT_ESOM_E = "#b22222"
COL_VAL        = "#7a6c2c"

# ---------------------------------------------------------------------------
# Figure & coordinate system [0..100] x [0..100]
# Use a wider canvas so all labels fit without overlap.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect("auto"); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8, fw="normal", lw=0.9):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color="#202020", linespacing=1.35)


def arrow(x1, y1, x2, y2, color="#404040", lw=1.0, style="-|>", mut=12):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def edge_label(x, y, text, color="#404040", fs=7, ha="center", va="center",
               italic=True):
    style = "italic" if italic else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, style=style,
            linespacing=1.25)


# ---------------------------------------------------------------------------
# Layout coordinates
#   Top row     y=92   (historical)
#   Middle row  y=58   (empirical | esom)
#   Bottom row  y=22   (future emp | future esom)
#   Left col    x=22
#   Right col   x=78
#   Box widths  ~28; gap between cols = 56 - 28 = 28 (plenty of room)
# ---------------------------------------------------------------------------

# (1) Historical prices — top centre-left
box(22, 92, 30, 7,
    "Historical electricity prices\n(ENTSO-E day-ahead, 2015–2025)",
    fc=COL_HIST_F, ec=COL_HIST_E, fw="bold")

# Arrow down to empirical model with explanatory label
arrow(22, 88.5, 22, 65)
edge_label(40, 78,
           "Predictors:  capacity shares  •  country FE\n"
           "NTC openness  •  neighbour-weighted exposure",
           color=COL_HIST_E, ha="left", italic=True, fs=7)

# (2) Empirical model — middle-left
box(22, 58, 30, 11,
    "Hourly mean-deviation model\n"
    "$\\widehat{\\mathrm{dev}}_h(c, y) = \\alpha_c^{(h)} + \\sum_t \\beta_t^{(h)} \\, s_t(c,y)$\n"
    "(pooled per-hour OLS, country FE)",
    fc=COL_EMP_F, ec=COL_EMP_E, fw="bold", fs=7.5)

# (3) ESOM — middle-right
box(78, 58, 30, 11,
    "ESOM\n"
    "PyPSA-Eur  •  GENeSYS-MOD  •  …\n"
    "cost-min dispatch under future scenario",
    fc=COL_ESOM_F, ec=COL_ESOM_E, fw="bold", fs=7.5)

# (3a) ESOM -> empirical (new-tech predictors, dashed/red horizontal arrow)
arrow(63, 58, 37, 58, color=COL_ESOM_E, lw=1.2)
edge_label(50, 62.5,
           "new-tech predictors  ($\\beta_t^{(h)}$ for BESS,\nelectrolyzers, V2G, demand response)",
           color=COL_ESOM_E, italic=True, fs=7)

# (4) Empirical -> future shape (down arrow)
arrow(22, 52.5, 22, 32)

# (5) ESOM -> future hourly prices (down arrow)
arrow(78, 52.5, 78, 32, color=COL_ESOM_E)
edge_label(82, 42, "shadow prices\n(EB constraint duals)",
           color=COL_ESOM_E, italic=True, fs=7, ha="left")

# (6) Future mean hourly price structure (bottom-left)
box(22, 22, 30, 12,
    "Future mean hourly\nprice structure  +  90 % PI\n(deviation profile per country/year)",
    fc=COL_FUT_EMP_F, ec=COL_FUT_EMP_E, fw="bold", fs=8)

# (7) Future hourly electricity prices (bottom-right)
box(78, 22, 30, 12,
    "Future hourly\nelectricity prices\n(shadow-price profile per scenario/year)",
    fc=COL_FUT_ESOM_F, ec=COL_FUT_ESOM_E, fw="bold", fs=8)

# (8) Validation arrow with labels above and below
arrow(37, 22, 63, 22, color=COL_VAL, lw=1.6, style="<|-|>", mut=14)
edge_label(50, 25, "validation",
           color=COL_VAL, italic=False, fs=8.5)
edge_label(50, 19,
           "Δ shape  =  ESOM − empirical\n→ model-artefact catalogue",
           color=COL_VAL, italic=True, fs=7)

# (9) Footer (well below boxes)
ax.text(50, 8, "Electricity Price Evolution — empirical-anchored hybrid framework",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="#404040")
ax.text(50, 4,
        "ESOM has a dual role:  (i) supplies coefficients for new techs the empirical model cannot identify from history;  "
        "(ii) independently produces shadow prices for the same scenario.\nThe gap quantifies the ESOM's structural-shape "
        "distortion relative to historical-style market dynamics.",
        ha="center", va="center", fontsize=7, color="#606060",
        style="italic", linespacing=1.4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_pdf = PLOTS / "framework_diagram.pdf"
out_png = PLOTS / "framework_diagram.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote: {out_pdf}")
print(f"       {out_png}")
