"""Cumulative constraint ladder on the calibrated 2024 network.

Unlike ``profiling.py`` (each constraint in isolation, for compute
cost), this runner builds the *cumulative* sequence: each configuration
keeps every constraint of the previous one and adds the next. The
per-zone shadow prices it produces are the paper's headline result, so
it solves with **crossover enabled** for vertex-clean duals, and writes
a ``validation_report.json`` next to every configuration.

Cumulative order (each step adds to all previous):

  0  baseline      textbook energy-only LP
  1  +voll         flat-cost load shedding at VOLL
  2  +elastic      PWL price-elastic demand (QP)
  3  +ramping      per-generator ramp limits
  4  +reserves     energy-and-reserve co-clearing (extra_functionality)
  5  +rolling      switch to 24 h rolling horizon (realised series)
  6  +forecast     windows replayed on D-1 forecast information
  7  +uc           unit commitment with start-up costs (windowed MIP)

Steps 0-4 solve annually; 5-7 solve in 24 h windows (storage SOC pinned
to the annual perfect-foresight trajectory). Reserves, once added at
step 4, stay active through the windowed steps via extra_functionality.

The German EEG subsidy floor is intentionally excluded: it is a
zone-specific (DE) mechanism, run separately via runs/profiling.py
'subsidy_de' as a Germany case study rather than pooled here.

Usage (from the directory containing ``pypsa-eur/``):

    PYTHONPATH=ladder python ladder/runs/ladder.py [--weeks N] [step ...]

``--weeks N`` truncates to the first N*168 snapshots for smoke testing;
positional step names (e.g. ``00_baseline 01_voll``) restrict the run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pypsa

from constraint_ladder.constraints import apply_price_elastic_demand
from constraint_ladder.helpers import (
    add_operational_reserve_margin,
    anchor_vre_generation,
    apply_ramp_limits,
    apply_unit_commitment_csv,
    apply_voll_load_shedding,
    build_forecast_ratios,
    record_solve_metrics,
    reset_to_lp_baseline,
    solve_rolling_horizon,
    validate_run,
    write_report,
)

NETWORK_PATH = Path("pypsa-eur/resources/eu_2024_dispatch/networks/base_s_adm_elec_.nc")
BASELINE_SOLVED = Path("results/baseline/solved.nc")
FORECASTS_DIR = Path("calibration_data/ENTSOE_forecasts_2024")
GENERATION_DIR = Path("calibration_data/ENTSOE_generation_2024")
ZONE_PRICE_CSV = Path("ladder/constraint_ladder/data/zone_mean_price_2024.csv")
UC_CSV = "pypsa-eur/data/unit_commitment.csv"
RESULTS_ROOT = Path("results/ladder")

SOLVER_NAME = "gurobi"
# Crossover ENABLED (no "Crossover: 0") -> vertex-clean duals for the
# headline shadow prices, unlike the profiling runner.
LP_OPTIONS = {
    "Method": 2,
    "BarConvTol": 1e-6,
    "FeasibilityTol": 1e-5,
    "OptimalityTol": 1e-5,
    "Threads": 20,
    "Seed": 123,
}
MIP_OPTIONS = {"MIPGap": 0.01, "Threads": 20, "Seed": 123, "TimeLimit": 3600}
RESERVE_KWARGS = dict(epsilon_load=0.02, epsilon_vres=0.02, contingency_mw=4000.0)
VOLL_EUR_PER_MWH = 8000.0

# Cumulative step indices. The German EEG subsidy floor is NOT part of
# the pooled continental ladder -- its negative-price bidding is a
# largely German legacy-EEG phenomenon, so it is run separately as a DE
# case study (see runs/profiling.py 'subsidy_de'), not pooled across the
# 41 validated zones.
STEPS = [
    "00_baseline", "01_voll", "02_elastic", "03_ramping", "04_reserves",
    "05_rolling", "06_forecast", "07_uc",
]
IDX = {name: i for i, name in enumerate(STEPS)}
ROLLING_FROM = IDX["05_rolling"]
RESERVES_FROM = IDX["04_reserves"]
FORECAST_FROM = IDX["06_forecast"]
UC_FROM = IDX["07_uc"]


def zone_calibration(n):
    prices = pd.read_csv(ZONE_PRICE_CSV).set_index("zone")["mean_price_2024"]
    zone_mean_price = {b: float(prices.get(b, prices.mean())) for b in n.buses.index}
    zone_mean_load = {
        b: float(n.loads_t.p_set[b].mean())
        for b in n.buses.index
        if b in n.loads_t.p_set.columns
    }
    return zone_mean_price, zone_mean_load


def build_cumulative(weeks: int | None, target: int) -> pypsa.Network:
    """Network with every *static* constraint up to and including ``target``."""
    n = pypsa.Network(NETWORK_PATH)
    if weeks:
        n.snapshots = n.snapshots[: weeks * 168]
    reset_to_lp_baseline(n)
    anchor_vre_generation(n)
    if target >= IDX["01_voll"]:
        apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
    if target >= IDX["02_elastic"]:
        prices, loads = zone_calibration(n)
        apply_price_elastic_demand(n, prices, loads)
    if target >= IDX["03_ramping"]:
        apply_ramp_limits(n, UC_CSV)
    if target >= UC_FROM:
        n_comm = apply_unit_commitment_csv(n, UC_CSV)
        print(f"  committable generators: {n_comm}", flush=True)
    # reserves (step 4) is added at solve time via extra_functionality.
    return n


def solve_step(name: str, weeks: int | None) -> None:
    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    target = IDX[name]
    t0 = time.perf_counter()
    n = build_cumulative(weeks, target)

    reserve_fn = None
    if target >= RESERVES_FROM:
        def reserve_fn(m, sns):  # PyPSA extra_functionality signature
            add_operational_reserve_margin(m, sns, **RESERVE_KWARGS)

    options = MIP_OPTIONS if target >= UC_FROM else LP_OPTIONS
    is_windowed = target >= ROLLING_FROM
    sheds = target >= IDX["01_voll"]

    if not is_windowed:
        kwargs = dict(solver_name=SOLVER_NAME, solver_options=options)
        if reserve_fn is not None:
            kwargs["extra_functionality"] = reserve_fn
        status, condition = n.optimize(**kwargs)
        problem_class = "QP" if target == IDX["02_elastic"] else "LP"
        record_solve_metrics(
            n, output_dir=out, configuration=f"ladder/{name}",
            problem_class=problem_class, solver=SOLVER_NAME, solver_options=options,
            status=status, condition=condition,
            wallclock_s=time.perf_counter() - t0, network_path=NETWORK_PATH,
        )
    else:
        soc = pypsa.Network(BASELINE_SOLVED).storage_units_t.state_of_charge
        ratios = (
            build_forecast_ratios(n, FORECASTS_DIR, GENERATION_DIR)
            if target >= FORECAST_FROM else None
        )
        res = solve_rolling_horizon(
            n, horizon=24, ratios=ratios, soc_target=soc,
            extra_functionality=reserve_fn,
            solver_name=SOLVER_NAME, solver_options=options,
        )
        counts = res.condition.value_counts().to_dict()
        status = "ok"
        condition = "optimal" if counts.get("optimal", 0) == len(res) else "mixed"
        import json
        (out / "solve_metrics.json").write_text(json.dumps({
            "configuration": f"ladder/{name}",
            "problem_class": ("MIP (windowed)" if target >= UC_FROM else "LP (windowed)"),
            "n_windows": int(len(res)),
            "window_conditions": {str(k): int(v) for k, v in counts.items()},
            "wallclock_s": round(time.perf_counter() - t0, 1),
        }, indent=2))

    n.buses_t.marginal_price.to_csv(out / "shadow_price.csv")
    report = validate_run(n, status, condition, config=f"ladder/{name}", sheds_by_design=sheds)
    write_report(report, out)
    print(f"[{name}] {status}/{condition} in {time.perf_counter()-t0:.0f}s", flush=True)


def main() -> None:
    args = sys.argv[1:]
    weeks = None
    if "--weeks" in args:
        i = args.index("--weeks")
        weeks = int(args[i + 1])
        del args[i : i + 2]
    only = set(args)
    for name in STEPS:
        if only and name not in only:
            continue
        try:
            solve_step(name, weeks)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[{name}] FAILED -- continuing", flush=True)


if __name__ == "__main__":
    main()
