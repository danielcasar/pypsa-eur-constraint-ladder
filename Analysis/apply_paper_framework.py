"""
Apply the paper-version framework (from Code/paper/scripts/) to the new
PyPSA-Eur DE hourly shadow prices for a chosen planning year.

This script is a THIN DRIVER. It does NOT re-implement the methodology; it
imports the published functions directly:

    from stage1_kruskal_wallis      import kruskal_wallis, _add_time_dims, ...
    from stage2_t_tests             import stage2_ttests
    from stage3_pearson_correlations import _corr
    + deviation_range from compute_compression_factors.py (loaded via AST to
      bypass that module's unconditional file-I/O at import time)

The original paper folder is READ-ONLY. We only:
  - sys.path.insert into Code/paper/scripts/  (importing modules, not modifying)
  - read   Code/data/ENTSOE_day_ahead_prices.csv
All outputs land in this script's results/<planning_year>/ folder.

Configurable via env vars:
    PLANNING_YEAR  (default 2050)  -- which PyPSA-Eur planning year to analyse
    REFERENCE_YEAR (default 2025)  -- ENTSO-E year for the compression-factor
                                      denominator (paper uses 2025)
"""

from __future__ import annotations

import ast
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------------
PLANNING_YEAR = int(os.environ.get("PLANNING_YEAR", 2050))
REFERENCE_YEAR = int(os.environ.get("REFERENCE_YEAR", 2025))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / str(PLANNING_YEAR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Original paper (READ-ONLY) -------------------------------------------
PAPER1_ROOT = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)
PAPER1_SCRIPTS = PAPER1_ROOT / "Code" / "paper" / "scripts"
PAPER1_ENTSOE = PAPER1_ROOT / "Code" / "data" / "ENTSOE_day_ahead_prices.csv"
assert PAPER1_SCRIPTS.exists(), f"Original paper scripts not found: {PAPER1_SCRIPTS}"
assert PAPER1_ENTSOE.exists(), f"Original paper ENTSO-E data not found: {PAPER1_ENTSOE}"

# ---- This paper / new data ------------------------------------------------
PYPSA_FILE = (
    HERE.parent / "pypsa-eur" / "results"
    / f"de-{PLANNING_YEAR}" / "de_hourly_shadow_price.csv"
)

# ---------------------------------------------------------------------------
# Import published functions (no methodology duplication here)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PAPER1_SCRIPTS))

from stage1_kruskal_wallis import (  # type: ignore  # noqa: E402
    kruskal_wallis,
    _add_time_dims,
    _sig_stars,
    TIME_DIMENSIONS,
    MONTH_TO_SEASON,
    DIMENSION_DF,
    CI_ALPHA,
)
from stage2_t_tests import stage2_ttests  # type: ignore  # noqa: E402
from stage3_pearson_correlations import _corr, MIN_GROUPS  # type: ignore  # noqa: E402


def _import_function_via_ast(filepath: Path, function_name: str):
    """
    Extract one function definition from a Python source file and return it
    as a callable, WITHOUT executing the rest of the module.

    Needed because compute_compression_factors.py runs unconditional file
    I/O at module-load time (loads paper-specific 2025 data and asserts
    8760 rows), which we cannot trigger here.
    """
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(filepath))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"np": np, "pd": pd}
            exec(compile(mod, str(filepath), "exec"), ns)
            return ns[function_name]
    raise ImportError(f"{function_name} not defined in {filepath}")


deviation_range = _import_function_via_ast(
    PAPER1_SCRIPTS / "compute_compression_factors.py", "deviation_range"
)

# ---------------------------------------------------------------------------
# Loaders (this paper's data only)
# ---------------------------------------------------------------------------
PYPSA_SOURCE = "PyPSA-Eur"
PYPSA_SCENARIO = f"Default-{PLANNING_YEAR}"
COUNTRY = "DE"
ENTSOE_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def load_pypsa() -> pd.DataFrame:
    df = pd.read_csv(PYPSA_FILE)
    ts_col = next(c for c in df.columns if c.lower() in ("snapshot", "timestamp"))
    price_col = next(c for c in df.columns if "price" in c.lower())
    df = (
        df.rename(columns={ts_col: "timestamp", price_col: "price"})
        .loc[:, ["timestamp", "price"]]
        .dropna()
    )
    df = _add_time_dims(df)         # paper function — guarantees same dims
    df["year"] = PLANNING_YEAR      # mark as the planning year
    return df


def load_entsoe_year(year: int) -> pd.DataFrame | None:
    df = pd.read_csv(PAPER1_ENTSOE, index_col=0, parse_dates=True)
    if COUNTRY not in df.columns:
        return None
    s = df[[COUNTRY]].rename(columns={COUNTRY: "price"}).dropna()
    s = s[s.index.year == year]
    if len(s) < 30:
        return None
    return _add_time_dims(s)


# ---------------------------------------------------------------------------
# Driver — composes the imported functions
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print(f"Paper-framework analysis: PyPSA-Eur DE {PLANNING_YEAR}")
    print(f"  ENTSO-E pooled years for Stage 3 profile: {ENTSOE_YEARS}")
    print(f"  ENTSO-E year for compression-factor denominator: {REFERENCE_YEAR}")
    print("=" * 72)

    if not PYPSA_FILE.exists():
        raise SystemExit(
            f"PyPSA-Eur prices not found: {PYPSA_FILE}\n"
            f"Has the snakemake run for de-{PLANNING_YEAR} finished?"
        )

    pypsa_df = load_pypsa()
    print(
        f"\nPyPSA-Eur DE {PLANNING_YEAR}: {len(pypsa_df):,} hours, "
        f"mean={pypsa_df['price'].mean():.2f} EUR/MWh, "
        f"median={pypsa_df['price'].median():.2f}, "
        f"std={pypsa_df['price'].std():.2f}"
    )

    entsoe_per_year: dict[int, pd.DataFrame] = {}
    for y in ENTSOE_YEARS:
        d = load_entsoe_year(y)
        if d is not None:
            entsoe_per_year[y] = d
    print(f"ENTSO-E DE: {len(entsoe_per_year)} years available "
          f"({sorted(entsoe_per_year)})")

    # ---- Stage 1: Kruskal-Wallis  (calls paper's kruskal_wallis) ----------
    s1_rows = []
    for dim in TIME_DIMENSIONS:
        r = kruskal_wallis(pypsa_df, dim)
        if r is None:
            continue
        s1_rows.append({
            "source": PYPSA_SOURCE, "scenario": PYPSA_SCENARIO,
            "country": COUNTRY, "year": PLANNING_YEAR, "dimension": dim,
            **r, "significance": _sig_stars(r["p_value"]),
        })
    for y, d in entsoe_per_year.items():
        for dim in TIME_DIMENSIONS:
            r = kruskal_wallis(d, dim)
            if r is None:
                continue
            s1_rows.append({
                "source": "ENTSOE", "scenario": "Historical",
                "country": COUNTRY, "year": y, "dimension": dim,
                **r, "significance": _sig_stars(r["p_value"]),
            })
    s1 = pd.DataFrame(s1_rows)
    s1.to_csv(RESULTS_DIR / "stage1_kw_pooled.csv", index=False)
    print(f"\n[Stage 1] -> stage1_kw_pooled.csv  ({len(s1)} rows)")

    # ---- Stage 2: one-sample t-tests  (calls paper's stage2_ttests) -------
    s2_rows = []
    s2_rows.extend(
        stage2_ttests(pypsa_df, PYPSA_SOURCE, PYPSA_SCENARIO, COUNTRY, PLANNING_YEAR)
    )
    for y, d in entsoe_per_year.items():
        s2_rows.extend(stage2_ttests(d, "ENTSOE", "Historical", COUNTRY, y))
    s2 = pd.DataFrame(s2_rows)
    s2.to_csv(RESULTS_DIR / "stage2_ttest_groups_detail.csv", index=False)
    print(f"[Stage 2] -> stage2_ttest_groups_detail.csv  ({len(s2)} rows)")

    # ---- Stage 3: profile correlations  (calls paper's _corr) -------------
    entsoe_pooled = (
        s2[s2["source"] == "ENTSOE"]
        .groupby(["country", "dimension", "group"])["mean_group"]
        .mean()
        .reset_index()
        .rename(columns={"mean_group": "entsoe_mean"})
    )
    model_profile = (
        s2[s2["source"] == PYPSA_SOURCE]
        .groupby(["country", "scenario", "dimension", "group"])["mean_group"]
        .mean()
        .reset_index()
        .rename(columns={"mean_group": "model_mean"})
    )
    s3_rows = []
    for (country, scenario, dim), grp in model_profile.groupby(
        ["country", "scenario", "dimension"]
    ):
        ref = entsoe_pooled[
            (entsoe_pooled["country"] == country)
            & (entsoe_pooled["dimension"] == dim)
        ]
        merged = (
            pd.merge(
                ref[["group", "entsoe_mean"]],
                grp[["group", "model_mean"]],
                on="group",
            )
            .sort_values("group")
            .dropna()
        )
        if len(merged) < MIN_GROUPS:
            continue
        c = _corr(merged["entsoe_mean"], merged["model_mean"])
        s3_rows.append({
            "country": country, "scenario": scenario, "dimension": dim, **c,
        })
    s3 = pd.DataFrame(s3_rows)
    s3.to_csv(RESULTS_DIR / "stage3_pearson_correlations.csv", index=False)
    print(f"[Stage 3] -> stage3_pearson_correlations.csv  ({len(s3)} rows)")

    # ---- Compression factors  (calls paper's deviation_range) -------------
    cf = None
    if REFERENCE_YEAR in entsoe_per_year:
        ref_df = entsoe_per_year[REFERENCE_YEAR]
        rows = []
        for dim in TIME_DIMENSIONS:
            e_range = deviation_range(ref_df["price"].values, ref_df[dim].values)
            p_range = deviation_range(pypsa_df["price"].values, pypsa_df[dim].values)
            xi = e_range / p_range if p_range > 0 else np.nan
            rows.append({
                "country": COUNTRY,
                "dimension": dim,
                "entsoe_year": REFERENCE_YEAR,
                "entsoe_range": round(e_range, 3),
                "model_year": PLANNING_YEAR,
                "model_range": round(p_range, 3),
                "xi": round(xi, 2),
            })
        cf = pd.DataFrame(rows)
        cf_path = RESULTS_DIR / (
            f"compression_factors_DEonly_model{PLANNING_YEAR}"
            f"_vs_entsoe{REFERENCE_YEAR}.csv"
        )
        cf.to_csv(cf_path, index=False)
        print(f"[Compression] -> {cf_path.name}  ({len(cf)} rows)")
    else:
        print(f"[Compression] SKIPPED — ENTSO-E {REFERENCE_YEAR} not available")

    # ---- Human-readable summary -------------------------------------------
    out = []
    out.append("\n" + "=" * 72)
    out.append(f"SUMMARY  (PyPSA-Eur DE {PLANNING_YEAR}  vs  ENTSO-E DE)")
    out.append("=" * 72)

    out.append("\n[Stage 1] Calendar-effect strength (eps^2)")
    for src in s1["source"].unique():
        sub = s1[s1["source"] == src]
        parts = [f"  {src:10s}"]
        for dim in TIME_DIMENSIONS:
            d = sub[sub["dimension"] == dim]
            if len(d) == 0:
                continue
            mean_eps = d["effect_size"].mean()
            sig_pct = (d["p_value"] < 0.05).mean() * 100
            parts.append(f"{dim}: eps^2={mean_eps:.3f} ({sig_pct:.0f}% sig)")
        out.append("  | ".join(parts))

    out.append("\n[Stage 3] Profile correlations (PyPSA-Eur vs ENTSO-E pooled)")
    for _, r in s3.iterrows():
        out.append(
            f"  {r['dimension']:8s}: r = {r['pearson_r']:+.3f} (p={r['pearson_p']:.3g}), "
            f"rho = {r['spearman_r']:+.3f} (p={r['spearman_p']:.3g}), k={r['n_groups']}"
        )

    if cf is not None:
        out.append(f"\n[Compression] xi = ENTSOE-{REFERENCE_YEAR}_range / PyPSA-Eur-{PLANNING_YEAR}_range")
        for _, r in cf.iterrows():
            out.append(
                f"  {r['dimension']:8s}: xi = {r['xi']:5.2f}   "
                f"(ENTSOE range {r['entsoe_range']:.1f} EUR/MWh, "
                f"model range {r['model_range']:.1f} EUR/MWh)"
            )
        out.append(
            "\n  Interpretation: xi > 1 means ENTSO-E swings are LARGER than the model;"
        )
        out.append(
            "                  xi < 1 means model swings exceed observed market swings."
        )

    text = "\n".join(out)
    print(text)
    (RESULTS_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote: {RESULTS_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
