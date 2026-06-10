"""
Predict the cross-year HOD Pearson rho between two years from system-state
differences (VRE share, fossil share, demand level, ...).

Premise (established by linearity_of_pearson_decay.py):
  - HOD has a clean linear cross-year-rho decay with R^2 = 0.99 within each
    of the 22 ENTSO-E countries.
  - Slope -0.085/year => about 9 pp loss in HOD profile correlation per year
    of separation.
  - Different countries have different baselines (intercepts) but the same slope.

Hypothesis to test now:
  The "year-of-separation" lag is a *proxy* for system-state change. The real
  driver is how much the merit order has shifted between y1 and y2 — primarily
  measured by the change in VRE (wind+solar) share, fossil share, demand.

Specifications fitted:
  M0  rho ~ lag                                           (lag-only baseline)
  M1  rho ~ |dVRE|                                        (just VRE shift)
  M2  rho ~ |dVRE| + lag                                  (does lag still matter?)
  M3  rho ~ |dVRE| + |dFossil| + |dDemand| + lag          (full set)
  Each repeated WITH country fixed effects (M0_FE etc.) and WITHOUT.

Reads (READ-ONLY from first paper):
  Code/results/calendar_effects/entsoe_cross_year_pearson.csv
  Code/data/ENTSOE_generation_mix.csv

Writes:
  Analysis/results/system_state_panel.csv
  Analysis/results/predict_pearson_shift_models.csv
  Analysis/results/predict_pearson_shift_summary.txt
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER1 = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)
PEARSON_FILE = PAPER1 / "Code" / "results" / "calendar_effects" / "entsoe_cross_year_pearson.csv"
GEN_FILE = PAPER1 / "Code" / "data" / "ENTSOE_generation_mix.csv"

PAPER_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO", "SI", "SK",
]


# ---------------------------------------------------------------------------
# Build (country, year) system-state panel from generation-mix CSV
# ---------------------------------------------------------------------------
def categorize(tech: str) -> str:
    """Map a technology name to one of {VRE, RES_other, Fossil, Nuclear, Other}."""
    t = tech.lower()
    if "solar" in t or "wind" in t:
        return "VRE"
    if "biomass" in t or "hydro" in t or "geothermal" in t or "marine" in t:
        return "RES_other"
    if "fossil" in t or "peat" in t:
        return "Fossil"
    if "nuclear" in t:
        return "Nuclear"
    return "Other"


def build_state_panel() -> pd.DataFrame:
    print("Loading generation mix CSV (large)...")
    df = pd.read_csv(GEN_FILE, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = df["timestamp"].dt.year

    # Keep only simple-named columns:  XX_<technology>
    # Skip the older tuple-format columns like "DK_('Biomass', ...)"
    simple_cols = [c for c in df.columns
                   if c not in ("timestamp", "year")
                   and "(" not in c
                   and re.match(r"^[A-Z]{2}_", c)]

    rows = []
    for col in simple_cols:
        country = col[:2]
        if country not in PAPER_COUNTRIES:
            continue
        tech = col[3:]
        cat = categorize(tech)
        annual = (df.groupby("year")[col].sum(min_count=1) / 1000.0)  # MWh -> GWh
        for year, mwh in annual.items():
            rows.append({"country": country, "year": int(year),
                         "category": cat, "tech": tech, "gwh": float(mwh)})
    long_panel = pd.DataFrame(rows).dropna(subset=["gwh"])

    # Aggregate to country-year-category
    cat_sum = (
        long_panel.groupby(["country", "year", "category"])["gwh"].sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    cat_sum["total_gen_gwh"] = cat_sum.drop(columns=["country", "year"]).sum(axis=1)

    # Fractions
    for c in ("VRE", "RES_other", "Fossil", "Nuclear", "Other"):
        if c not in cat_sum.columns:
            cat_sum[c] = 0.0
    cat_sum["vre_share"]    = cat_sum["VRE"]    / cat_sum["total_gen_gwh"]
    cat_sum["res_share"]    = (cat_sum["VRE"] + cat_sum["RES_other"]) / cat_sum["total_gen_gwh"]
    cat_sum["fossil_share"] = cat_sum["Fossil"] / cat_sum["total_gen_gwh"]

    panel = cat_sum[[
        "country", "year", "total_gen_gwh",
        "vre_share", "res_share", "fossil_share",
    ]].copy()

    # Filter implausible rows (e.g. coverage < few TWh suggests data gaps)
    panel = panel[panel["total_gen_gwh"] > 1000].copy()

    return panel


# ---------------------------------------------------------------------------
# Build pairwise (country, y1, y2) panel: HOD rho + system-state diffs
# ---------------------------------------------------------------------------
def build_pairwise_panel(state: pd.DataFrame) -> pd.DataFrame:
    rho = pd.read_csv(PEARSON_FILE)
    rho = rho[(rho["dimension"] == "hour") & (rho["year2"] > rho["year1"])].copy()
    rho["lag"] = rho["year2"] - rho["year1"]

    s1 = state.add_suffix("_y1").rename(
        columns={"country_y1": "country", "year_y1": "year1"})
    s2 = state.add_suffix("_y2").rename(
        columns={"country_y2": "country", "year_y2": "year2"})

    df = (rho
          .merge(s1, on=["country", "year1"], how="inner")
          .merge(s2, on=["country", "year2"], how="inner"))

    # Differences (signed and absolute)
    for v in ("vre_share", "res_share", "fossil_share", "total_gen_gwh"):
        df[f"d_{v}"]   = df[f"{v}_y2"]   - df[f"{v}_y1"]
        df[f"abs_d_{v}"] = df[f"d_{v}"].abs()

    return df


# ---------------------------------------------------------------------------
# Fit & report a model
# ---------------------------------------------------------------------------
def fit_ols(df: pd.DataFrame, predictors: list[str], country_fe: bool) -> dict:
    y = df["rho"].values
    X = df[predictors].copy()
    if country_fe:
        # FE via dummies (drop_first=True to avoid perfect collinearity with const)
        dummies = pd.get_dummies(df["country"], prefix="c", drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X.astype(float))
    res = sm.OLS(y, X).fit()
    return {
        "predictors": ", ".join(predictors) + (" + country_FE" if country_fe else ""),
        "n": int(res.nobs),
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "aic": float(res.aic),
        "coef_const": float(res.params.get("const", np.nan)),
        # Keep raw params/p-values for the predictors of interest
        **{f"coef_{p}": float(res.params[p]) for p in predictors if p in res.params},
        **{f"p_{p}":    float(res.pvalues[p]) for p in predictors if p in res.pvalues},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    state = build_state_panel()
    state.to_csv(RESULTS / "system_state_panel.csv", index=False)
    print(f"\nState panel: {len(state)} country-year rows; "
          f"{state['country'].nunique()} countries, "
          f"years {state['year'].min()}..{state['year'].max()}")

    pair = build_pairwise_panel(state)
    print(f"Pairwise HOD rho panel: {len(pair)} (country, y1, y2) rows")

    # ---- Fit all specifications -------------------------------------------
    specs = [
        ("M0_lag",                  ["lag"]),
        ("M1_dVRE",                 ["abs_d_vre_share"]),
        ("M2_dVRE_lag",             ["abs_d_vre_share", "lag"]),
        ("M3_dVRE_dFossil_dDem",    ["abs_d_vre_share", "abs_d_fossil_share",
                                     "abs_d_total_gen_gwh"]),
        ("M4_full",                 ["abs_d_vre_share", "abs_d_fossil_share",
                                     "abs_d_total_gen_gwh", "lag"]),
    ]
    rows = []
    for name, preds in specs:
        for fe in (False, True):
            r = fit_ols(pair, preds, fe)
            r["spec"] = name + ("_FE" if fe else "")
            rows.append(r)
    res_df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    res_df.to_csv(RESULTS / "predict_pearson_shift_models.csv", index=False)

    # ---- Human-readable summary -------------------------------------------
    out = []
    out.append("=" * 78)
    out.append("Predicting cross-year HOD Pearson rho from system-state differences")
    out.append("=" * 78)
    out.append(f"\nN = {len(pair)} pairwise observations across "
               f"{pair['country'].nunique()} countries, "
               f"{pair['year1'].min()}..{pair['year2'].max()}")
    out.append(f"Mean rho        = {pair['rho'].mean():.3f}")
    out.append(f"Mean |dVRE_share| = {pair['abs_d_vre_share'].mean():.3f}  "
               f"(range: {pair['abs_d_vre_share'].min():.3f}.."
               f"{pair['abs_d_vre_share'].max():.3f})")
    out.append("")
    out.append(f"{'Specification':<55} {'n':>5} {'R^2':>7} {'R^2_adj':>8}")
    out.append("-" * 78)
    for _, r in res_df.iterrows():
        out.append(f"{r['spec']:<55} {int(r['n']):>5} {r['r2']:>7.3f} {r['r2_adj']:>8.3f}")

    out.append("")
    out.append("Coefficients on the interesting variable (|dVRE_share|, lag):")
    out.append(f"{'Specification':<55} {'coef_dVRE':>10} {'p_dVRE':>10} "
               f"{'coef_lag':>10} {'p_lag':>10}")
    out.append("-" * 78)
    for _, r in res_df.iterrows():
        cv = r.get("coef_abs_d_vre_share")
        pv = r.get("p_abs_d_vre_share")
        cl = r.get("coef_lag")
        pl = r.get("p_lag")
        cvs = f"{cv:>+10.3f}" if pd.notna(cv) else f"{'---':>10}"
        pvs = f"{pv:>10.3g}" if pd.notna(pv) else f"{'---':>10}"
        cls = f"{cl:>+10.5f}" if pd.notna(cl) else f"{'---':>10}"
        pls = f"{pl:>10.3g}" if pd.notna(pl) else f"{'---':>10}"
        out.append(f"{r['spec']:<55} {cvs} {pvs} {cls} {pls}")

    out.append("")
    out.append("Reading guide:")
    out.append("  - If |dVRE| has a strong negative coef and lag becomes insignificant")
    out.append("    once |dVRE| is added (M2 vs M0), the lag effect was a proxy for VRE.")
    out.append("  - If FE specifications have much higher R^2, country baselines matter.")
    out.append("  - coef on |dVRE| of, say, -2.0 means: 1 percentage-point larger VRE")
    out.append("    share difference between two years -> rho lower by 2 (i.e., 0.02 less)")

    text = "\n".join(out)
    print(text)
    (RESULTS / "predict_pearson_shift_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
