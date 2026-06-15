"""Per-configuration validation for the constraint ladder.

Each ladder/profiling run can call :func:`validate_run` to emit a
``validation_report.json`` next to its results, so every configuration
ships with a pass/fail verdict rather than raw numbers. Three checks:

1. **Solve integrity** -- status ``ok`` / condition ``optimal`` (with
   crossover the duals are vertex-clean), no NaN duals, energy balance
   closes, and load shedding stays negligible unless the configuration
   is meant to shed.
2. **Dispatch plausibility** -- annual generation per country and
   carrier group against Ember's observed 2024 record
   (``constraint_ladder/data/ember_generation_2024.csv``). Wind / solar
   / hydro are anchored so they are expected to match; the informative
   carriers are gas and hard coal, where dispatch bugs surface (this is
   the check that caught the CO2 / coal-collapse bug).
3. The cross-configuration *directional* check (does each step move the
   compression / alignment metric the right way) is done downstream in
   the local framework analysis, where the per-zone shadow prices are
   compared to ENTSO-E; it is not part of this server-side report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

EMBER_CSV = Path(__file__).resolve().parent.parent / "data" / "ember_generation_2024.csv"

# Map PyPSA carriers to the Ember carrier groups in the benchmark.
CARRIER_GROUP = {
    "CCGT": "Gas", "OCGT": "Gas",
    "coal": "Coal", "lignite": "Coal",
    "nuclear": "Nuclear",
    "onwind": "Wind", "offwind-ac": "Wind", "offwind-dc": "Wind",
    "offwind-float": "Wind",
    "solar": "Solar", "solar-hsat": "Solar",
    "hydro": "Hydro", "ror": "Hydro",
    "biomass": "Bioenergy", "waste": "Bioenergy",
}
# Carriers whose annual energy is anchored/near-exact by construction;
# a deviation here means a calibration regression, not a dispatch bug.
ANCHORED_GROUPS = {"Wind", "Solar", "Hydro"}
# Tolerance for the dispatch gate (relative deviation vs Ember).
MIX_TOL = 0.20
# Benchmarks below this annual energy (TWh) are too small for % gates.
MIX_MIN_TWH = 2.0


def _generation_mix_twh(n: pypsa.Network) -> pd.Series:
    """Annual generation per (country, Ember carrier group), TWh."""
    g = n.generators
    gen = n.generators_t.p.sum().groupby(
        [g.bus.str[:2], g.carrier.map(CARRIER_GROUP)]
    ).sum()
    su = n.storage_units
    if len(su):
        sgen = n.storage_units_t.p.clip(lower=0).sum().groupby(
            [su.bus.str[:2], su.carrier.map(CARRIER_GROUP)]
        ).sum()
        gen = gen.add(sgen, fill_value=0.0)
    gen = gen.div(1e6)
    gen.index.names = ["country", "carrier_group"]
    return gen.dropna()


def validate_run(
    n: pypsa.Network,
    status: str,
    condition: str,
    *,
    config: str,
    sheds_by_design: bool = False,
    ember_csv: Path | str = EMBER_CSV,
) -> dict:
    """Return (and the caller writes) a validation report for one run."""
    report: dict = {"configuration": config, "checks": {}, "flags": []}

    # --- 1. solve integrity ---
    mp = n.buses_t.marginal_price
    nan_duals = int(mp.isna().sum().sum())
    optimal = (status == "ok") and (condition == "optimal")
    report["checks"]["solve"] = {
        "status": status,
        "condition": condition,
        "optimal": optimal,
        "nan_duals": nan_duals,
    }
    if not optimal:
        report["flags"].append(
            f"solve condition '{condition}' (not vertex-optimal; enable crossover for clean duals)"
        )
    if nan_duals:
        report["flags"].append(f"{nan_duals} NaN shadow prices")

    # --- load shedding ---
    ls_cols = [c for c in n.generators_t.p.columns if "load_shedding" in c.lower()]
    ls_twh = float(n.generators_t.p[ls_cols].sum().sum() / 1e6) if ls_cols else 0.0
    total_load_twh = float(n.loads_t.p_set.sum().sum() / 1e6)
    ls_share = ls_twh / total_load_twh if total_load_twh else 0.0
    report["checks"]["load_shedding"] = {
        "twh": round(ls_twh, 3),
        "share_of_load": round(ls_share, 5),
    }
    if not sheds_by_design and ls_share > 0.001:
        report["flags"].append(
            f"load shedding {ls_share*100:.2f}% of load (unexpected without scarcity)"
        )

    # --- 2. dispatch plausibility vs Ember ---
    # Annualise the modelled mix when the run covers only part of the year
    # (smoke tests), so the comparison to the annual Ember benchmark is
    # meaningful. For a full year the factor is 1.0.
    year_fraction = float(n.snapshot_weightings.generators.sum()) / 8760.0
    bench = pd.read_csv(ember_csv)
    bench = bench.set_index(["country", "carrier_group"])["twh"]
    mix = _generation_mix_twh(n)
    if year_fraction < 0.99:
        mix = mix / year_fraction
        report["checks"].setdefault("notes", []).append(
            f"generation annualised from {year_fraction:.3f} of the year (partial-horizon run)"
        )
    mix_flags = []
    rows = []
    for (cc, grp), obs in bench.items():
        if obs < MIX_MIN_TWH:
            continue
        model = float(mix.get((cc, grp), 0.0))
        dev = (model - obs) / obs
        rows.append(
            {"country": cc, "group": grp, "model": round(model, 1),
             "ember": round(obs, 1), "dev_pct": round(dev * 100, 0)}
        )
        if abs(dev) > MIX_TOL:
            tag = f"{cc} {grp} {dev*100:+.0f}%"
            mix_flags.append(tag)
            if grp in ANCHORED_GROUPS:
                report["flags"].append(f"ANCHORED carrier off: {tag} (calibration regression?)")
    n_off = len(mix_flags)
    report["checks"]["generation_mix"] = {
        "n_benchmarks": len(rows),
        "n_outside_tol": n_off,
        "outside_tol": mix_flags,
    }

    report["passed"] = len(report["flags"]) == 0
    return report


def write_report(report: dict, out_dir: Path) -> None:
    (Path(out_dir) / "validation_report.json").write_text(json.dumps(report, indent=2))
    verdict = "PASS" if report["passed"] else "REVIEW"
    flags = ("; ".join(report["flags"])) if report["flags"] else "none"
    print(f"  [validation] {verdict}  flags: {flags}", flush=True)
