"""
Predict cross-year HOD Pearson rho from per-technology generation shares
(disaggregated by carrier) and country-specific time-invariant levels.

Structure of predictors (each (country, y1, y2) pair):
  d_<tech>     = share_<tech>(y2) - share_<tech>(y1)     signed difference
  abs_d_<tech> = |share_<tech>(y2) - share_<tech>(y1)|   absolute difference
  level_<tech> = mean(share_<tech>(y1), share_<tech>(y2))     local two-year mean
  cmean_<tech> = country mean of share_<tech> over all available years
                 (proxy for time-invariant country structure -> replaces FE)

Optionally merges a fuel-price CSV (the user will provide). Expected schema:
  year,ttf_eur_per_mwh,eua_eur_per_t,api2_eur_per_t
into Analysis/data/fuel_prices.csv. If present, automatic diffs added.

Reads (READ-ONLY from first paper):
  Code/results/calendar_effects/entsoe_cross_year_pearson.csv
  Code/data/ENTSOE_generation_mix.csv

Writes:
  Analysis/results/system_state_panel_per_tech.csv
  Analysis/results/predict_pearson_shift_v2_models.csv
  Analysis/results/predict_pearson_shift_v2_coefs.csv
  Analysis/results/predict_pearson_shift_v2_summary.txt
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
DATA_LOCAL = HERE / "data"
DATA_LOCAL.mkdir(parents=True, exist_ok=True)

PAPER1 = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)
PEARSON_FILE = PAPER1 / "Code" / "results" / "calendar_effects" / "entsoe_cross_year_pearson.csv"
GEN_FILE = PAPER1 / "Code" / "data" / "ENTSOE_generation_mix.csv"
FUEL_FILE = DATA_LOCAL / "fuel_prices.csv"      # user-provided (optional)

PAPER_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "LT", "LV", "NL", "PL", "PT", "RO", "SI", "SK",
]

# ---------------------------------------------------------------------------
# Per-technology category mapping
# Group ENTSO-E carrier names into a parsimonious set of buckets.
# ---------------------------------------------------------------------------
TECH_GROUPS: list[tuple[str, list[str]]] = [
    ("solar",        ["Solar"]),
    ("onwind",       ["Wind_Onshore"]),
    ("offwind",      ["Wind_Offshore"]),
    ("hydro_disp",   ["Hydro_Water_Reservoir", "Hydro_Pumped_Storage"]),
    ("hydro_ror",    ["Hydro_Run_of_river_and_poundage"]),
    ("biomass",      ["Biomass"]),
    ("nuclear",      ["Nuclear"]),
    ("gas",          ["Fossil_Gas", "Fossil_Coal_derived_gas"]),
    ("hardcoal",     ["Fossil_Hard_coal"]),
    ("lignite",      ["Fossil_Brown_coal/Lignite"]),
    ("oil",          ["Fossil_Oil", "Fossil_Oil_shale", "Fossil_Peat"]),
    ("other",        ["Other", "Other_renewable", "Geothermal", "Marine", "Waste",
                      "Energy_storage"]),
]
TECH_NAMES = [g for g, _ in TECH_GROUPS]


def tech_of(carrier: str) -> str | None:
    """Return our tech-group name for a carrier name; None if unrecognised."""
    for group, members in TECH_GROUPS:
        if carrier in members:
            return group
    return None


# ---------------------------------------------------------------------------
# Build (country, year) per-tech share panel from generation-mix CSV
# ---------------------------------------------------------------------------
def build_state_panel() -> pd.DataFrame:
    print("Loading generation mix CSV (large)...")
    df = pd.read_csv(GEN_FILE, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = df["timestamp"].dt.year

    simple_cols = [c for c in df.columns
                   if c not in ("timestamp", "year")
                   and "(" not in c
                   and re.match(r"^[A-Z]{2}_", c)]

    rows = []
    skipped = set()
    for col in simple_cols:
        country = col[:2]
        if country not in PAPER_COUNTRIES:
            continue
        carrier = col[3:]
        tech = tech_of(carrier)
        if tech is None:
            skipped.add(carrier)
            continue
        annual = (df.groupby("year")[col].sum(min_count=1) / 1000.0)  # MWh -> GWh
        for year, gwh in annual.items():
            rows.append({"country": country, "year": int(year),
                         "tech": tech, "gwh": float(gwh)})
    if skipped:
        print(f"  carriers ignored (not in TECH_GROUPS): {sorted(skipped)}")

    long_panel = pd.DataFrame(rows).dropna(subset=["gwh"])

    wide = (long_panel.groupby(["country", "year", "tech"])["gwh"].sum()
            .unstack(fill_value=0.0).reset_index())
    for t in TECH_NAMES:
        if t not in wide.columns:
            wide[t] = 0.0
    wide["total_gen_gwh"] = wide[TECH_NAMES].sum(axis=1)

    # Filter implausibly small coverage years
    wide = wide[wide["total_gen_gwh"] > 1000].copy()

    for t in TECH_NAMES:
        wide[f"share_{t}"] = wide[t] / wide["total_gen_gwh"]

    panel = wide[["country", "year", "total_gen_gwh"]
                 + [f"share_{t}" for t in TECH_NAMES]].copy()
    return panel


# ---------------------------------------------------------------------------
# Optional: fuel-price merge
# ---------------------------------------------------------------------------
def load_fuel_prices() -> pd.DataFrame | None:
    if not FUEL_FILE.exists():
        print(f"  (no fuel prices at {FUEL_FILE} - skipping fuel-price predictors)")
        return None
    fp = pd.read_csv(FUEL_FILE)
    expected = {"year", "ttf_eur_per_mwh", "eua_eur_per_t", "api2_eur_per_t"}
    missing = expected - set(fp.columns)
    if missing:
        print(f"  WARNING: fuel_prices.csv missing columns {missing}; "
              f"continuing with what's there")
    return fp


# ---------------------------------------------------------------------------
# Build pairwise panel
# ---------------------------------------------------------------------------
def build_pairwise(state: pd.DataFrame, fuel: pd.DataFrame | None) -> pd.DataFrame:
    rho = pd.read_csv(PEARSON_FILE)
    rho = rho[(rho["dimension"] == "hour") & (rho["year2"] > rho["year1"])].copy()
    rho["lag"] = rho["year2"] - rho["year1"]

    # Country-mean shares over all available years (time-invariant per country)
    cmean = (state.groupby("country")[[f"share_{t}" for t in TECH_NAMES]]
             .mean()
             .add_prefix("cmean_")
             .reset_index())
    cmean.columns = ["country"] + [c.replace("cmean_share_", "cmean_")
                                    for c in cmean.columns[1:]]

    s1 = state.add_suffix("_y1").rename(
        columns={"country_y1": "country", "year_y1": "year1"})
    s2 = state.add_suffix("_y2").rename(
        columns={"country_y2": "country", "year_y2": "year2"})
    df = (rho
          .merge(s1, on=["country", "year1"], how="inner")
          .merge(s2, on=["country", "year2"], how="inner")
          .merge(cmean, on="country", how="left"))

    # Per-tech difference (signed and absolute) and local two-year level
    for t in TECH_NAMES:
        c1, c2 = f"share_{t}_y1", f"share_{t}_y2"
        df[f"d_{t}"]     = df[c2] - df[c1]
        df[f"abs_d_{t}"] = df[f"d_{t}"].abs()
        df[f"level_{t}"] = (df[c1] + df[c2]) / 2.0

    # Demand-level proxies
    df["d_total"]     = df["total_gen_gwh_y2"] - df["total_gen_gwh_y1"]
    df["abs_d_total"] = df["d_total"].abs()
    df["level_total"] = (df["total_gen_gwh_y1"] + df["total_gen_gwh_y2"]) / 2.0

    # Optional fuel-price merge
    if fuel is not None:
        for col in [c for c in fuel.columns if c != "year"]:
            f1 = fuel.set_index("year")[col].to_dict()
            df[f"{col}_y1"] = df["year1"].map(f1)
            df[f"{col}_y2"] = df["year2"].map(f1)
            df[f"d_{col}"]     = df[f"{col}_y2"] - df[f"{col}_y1"]
            df[f"abs_d_{col}"] = df[f"d_{col}"].abs()
            df[f"level_{col}"] = (df[f"{col}_y1"] + df[f"{col}_y2"]) / 2.0

    return df


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_ols(df: pd.DataFrame, predictors: list[str], country_fe: bool) -> tuple[dict, pd.Series, pd.Series]:
    """Fit y = X*beta. Returns summary dict + coef Series + p-value Series."""
    sub = df.dropna(subset=["rho"] + predictors)
    y = sub["rho"].values
    X = sub[predictors].copy()
    if country_fe:
        dummies = pd.get_dummies(sub["country"], prefix="c", drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X.astype(float))
    res = sm.OLS(y, X).fit()
    summary = {
        "n": int(res.nobs),
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }
    coefs = res.params.drop("const", errors="ignore")
    pvals = res.pvalues.drop("const", errors="ignore")
    return summary, coefs, pvals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    state = build_state_panel()
    state.to_csv(RESULTS / "system_state_panel_per_tech.csv", index=False)
    print(f"\nState panel: {len(state)} country-years; "
          f"{state['country'].nunique()} countries, "
          f"years {state['year'].min()}..{state['year'].max()}")

    fuel = load_fuel_prices()

    pair = build_pairwise(state, fuel)
    print(f"Pairwise HOD rho panel: {len(pair)} rows")

    # ---- Specifications -----------------------------------------------------
    # Drop one tech as the omitted category (shares sum to 1; otherwise
    # collinearity with the constant). Use 'other' as omitted.
    diff_techs  = [t for t in TECH_NAMES if t != "other"]
    abs_diff_predictors = [f"abs_d_{t}" for t in diff_techs]
    sgn_diff_predictors = [f"d_{t}"     for t in diff_techs]
    cmean_predictors    = [f"cmean_{t}" for t in diff_techs]   # country structure
    level_predictors    = [f"level_{t}" for t in diff_techs]   # local two-year level

    fuel_diff_predictors = []
    fuel_level_predictors = []
    if fuel is not None:
        for col in [c for c in fuel.columns if c != "year"]:
            fuel_diff_predictors.append(f"abs_d_{col}")
            fuel_level_predictors.append(f"level_{col}")

    specs: list[tuple[str, list[str], bool]] = [
        ("M0_lag_FE",                          ["lag"], True),
        ("M1_per_tech_d",                      abs_diff_predictors, False),
        ("M2_per_tech_d_+_lag",                abs_diff_predictors + ["lag"], False),
        ("M3_per_tech_d_+_cmean",              abs_diff_predictors + cmean_predictors, False),
        ("M4_per_tech_d_+_cmean_+_lag",        abs_diff_predictors + cmean_predictors + ["lag"], False),
        ("M5_per_tech_d_+_FE",                 abs_diff_predictors, True),
        ("M6_per_tech_d_+_cmean_+_lag_+_FE",   abs_diff_predictors + cmean_predictors + ["lag"], True),
    ]
    if fuel_diff_predictors:
        specs += [
            ("M7_+_fuel_diffs",
             abs_diff_predictors + cmean_predictors + ["lag"] + fuel_diff_predictors, False),
            ("M8_+_fuel_diffs_and_levels",
             abs_diff_predictors + cmean_predictors + ["lag"]
             + fuel_diff_predictors + fuel_level_predictors, False),
        ]

    # ---- Run all specs -----------------------------------------------------
    rows = []
    coefs_long = []
    for name, preds, fe in specs:
        s, coefs, pvals = fit_ols(pair, preds, fe)
        rows.append({"spec": name, "predictors": ", ".join(preds), "FE": fe, **s})
        for k, v in coefs.items():
            if str(k).startswith("c_"):
                continue
            coefs_long.append({
                "spec": name,
                "predictor": k,
                "coef": float(v),
                "p_value": float(pvals.get(k, np.nan)),
            })
    res_df = pd.DataFrame(rows).sort_values("r2_adj", ascending=False)
    coefs_df = pd.DataFrame(coefs_long)
    res_df.to_csv(RESULTS / "predict_pearson_shift_v2_models.csv", index=False)
    coefs_df.to_csv(RESULTS / "predict_pearson_shift_v2_coefs.csv", index=False)

    # ---- Human-readable summary --------------------------------------------
    out = []
    out.append("=" * 78)
    out.append("Predicting cross-year HOD rho — per-technology + country-mean predictors")
    out.append("=" * 78)
    out.append(f"\nN = {len(pair)} (country, y1, y2) observations, "
               f"{pair['country'].nunique()} countries, "
               f"{pair['year1'].min()}..{pair['year2'].max()}")
    out.append(f"Mean rho = {pair['rho'].mean():.3f}\n")

    out.append(f"{'Specification':<40} {'n':>5} {'k':>4} {'R^2':>7} {'R^2_adj':>9} {'AIC':>10}")
    out.append("-" * 78)
    for _, r in res_df.iterrows():
        k = len(str(r["predictors"]).split(", "))
        if r["FE"]:
            k += pair["country"].nunique() - 1
        out.append(f"{r['spec']:<40} {int(r['n']):>5} {k:>4} {r['r2']:>7.3f} "
                   f"{r['r2_adj']:>9.3f} {r['aic']:>10.0f}")

    # Spec with best R^2_adj — show top-ranked predictors
    best_spec_name = res_df.iloc[0]["spec"]
    best_coefs = (coefs_df[coefs_df["spec"] == best_spec_name]
                  .assign(absc=lambda d: d["coef"].abs())
                  .sort_values("absc", ascending=False)
                  .drop(columns="absc")
                  .head(15))
    out.append("")
    out.append(f"Top-15 |coefficients| in best spec ({best_spec_name}):")
    out.append(f"{'predictor':<28} {'coef':>11} {'p-value':>11}")
    out.append("-" * 78)
    for _, r in best_coefs.iterrows():
        out.append(f"{r['predictor']:<28} {r['coef']:>+11.4f} {r['p_value']:>11.3g}")

    # Did country-mean (cmean_*) successfully replace FE?
    no_fe = res_df[~res_df["FE"]]
    if not no_fe.empty:
        best_no_fe = no_fe.sort_values("r2_adj", ascending=False).iloc[0]
        out.append("")
        out.append(f"Best non-FE spec: {best_no_fe['spec']} "
                   f"R^2_adj = {best_no_fe['r2_adj']:.3f}")
    fe_models = res_df[res_df["FE"]]
    if not fe_models.empty:
        best_fe = fe_models.sort_values("r2_adj", ascending=False).iloc[0]
        out.append(f"Best FE spec:     {best_fe['spec']} "
                   f"R^2_adj = {best_fe['r2_adj']:.3f}")
        if not no_fe.empty:
            gap = best_fe["r2_adj"] - best_no_fe["r2_adj"]
            out.append(f"Gap (FE advantage over best non-FE):  R^2_adj +{gap:.3f}")
            if gap < 0.02:
                out.append("  -> country-mean predictors essentially replace FE.")
            else:
                out.append("  -> FE still adds; cmean predictors are not yet sufficient.")

    if fuel is None:
        out.append("\nNOTE: no fuel-price file at Analysis/data/fuel_prices.csv.")
        out.append("      Drop a CSV with columns year, ttf_eur_per_mwh, "
                   "eua_eur_per_t, api2_eur_per_t and re-run for richer specs.")

    text = "\n".join(out)
    print(text)
    (RESULTS / "predict_pearson_shift_v2_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
