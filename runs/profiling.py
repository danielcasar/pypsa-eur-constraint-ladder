"""One-at-a-time constraint profiling on the calibrated 2024 network.

Solves the baseline plus each ladder constraint *in isolation* and
records wall-clock, problem size, and peak memory per configuration.
This is the profiling pass of the paper's method section: the
cumulative activation order is set empirically by per-constraint
compute cost, and the compute / scientific-value table of the results
section is populated from these runs.

Configurations (each starts from the freshly reset + anchored network):

* ``voll``       -- flat-cost load shedding at every AC bus
* ``elastic``    -- VOLL + PWL price-elastic demand (QP)
* ``ramping``    -- per-generator ramp limits from unit_commitment.csv
* ``reserves``   -- energy-and-reserve co-clearing
* ``subsidy_de`` -- DE must-take + EEG subsidy-floor bids
* ``rolling``    -- 24 h rolling horizon on realised series
* ``forecast``   -- 24 h rolling horizon on D-1 forecast information
* ``uc``         -- unit commitment with start-up costs (annual MIP)

Run order is cheap-to-expensive so partial results are usable early.
Each configuration writes ``results/profiling/<name>/`` with
``solve_metrics.json``, ``shadow_price.csv``, ``generation_mix.csv``.

Usage (from the directory containing ``pypsa-eur/``):

    PYTHONPATH=ladder python ladder/runs/profiling.py [config ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pypsa

from constraint_ladder.constraints import (
    apply_de_must_take,
    apply_de_subsidy_floor,
    apply_price_elastic_demand,
)
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
)

NETWORK_PATH = Path("pypsa-eur/resources/eu_2024_dispatch/networks/base_s_adm_elec_.nc")
BASELINE_SOLVED = Path("results/baseline/solved.nc")
FORECASTS_DIR = Path("calibration_data/ENTSOE_forecasts_2024")
GENERATION_DIR = Path("calibration_data/ENTSOE_generation_2024")
ZONE_PRICE_CSV = Path("ladder/constraint_ladder/data/zone_mean_price_2024.csv")
RESULTS_ROOT = Path("results/profiling")

SOLVER_NAME = "gurobi"
LP_OPTIONS = {
    "Method": 2,
    "Crossover": 0,
    "BarConvTol": 1e-6,
    "FeasibilityTol": 1e-5,
    "OptimalityTol": 1e-5,
    "Threads": 20,
    "Seed": 123,
}
MIP_OPTIONS = {
    "MIPGap": 0.01,
    "Threads": 20,
    "Seed": 123,
    "TimeLimit": 21600,  # 6 h cap for the annual UC MIP
}
# PyPSA-Eur config.default.yaml electricity.operational_reserve values.
RESERVE_KWARGS = dict(epsilon_load=0.02, epsilon_vres=0.02, contingency_mw=4000.0)
VOLL_EUR_PER_MWH = 8000.0


def fresh_network() -> pypsa.Network:
    n = pypsa.Network(NETWORK_PATH)
    reset_to_lp_baseline(n)
    anchor_vre_generation(n)
    return n


def zone_calibration(n: pypsa.Network):
    prices = pd.read_csv(ZONE_PRICE_CSV).set_index("zone")["mean_price_2024"]
    zone_mean_price = {b: float(prices.get(b, prices.mean())) for b in n.buses.index}
    zone_mean_load = {
        b: float(n.loads_t.p_set[b].mean())
        for b in n.buses.index
        if b in n.loads_t.p_set.columns
    }
    return zone_mean_price, zone_mean_load


def save_outputs(n: pypsa.Network, out: Path) -> None:
    n.buses_t.marginal_price.to_csv(out / "shadow_price.csv")
    gen = (
        n.generators_t.p.sum()
        .groupby([n.generators.bus.str[:2], n.generators.carrier])
        .sum()
        .div(1e6)
        .rename("twh")
    )
    su = (
        n.storage_units_t.p.clip(lower=0).sum()
        .groupby([n.storage_units.bus.str[:2], n.storage_units.carrier])
        .sum()
        .div(1e6)
        .rename("twh")
    )
    mix = pd.concat([gen, su])
    mix.index.names = ["country", "carrier"]
    mix.to_csv(out / "generation_mix.csv")


def run_annual(name: str, setup, problem_class: str, options: dict) -> None:
    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = fresh_network()
    setup(n)
    status, condition = n.optimize(
        solver_name=SOLVER_NAME, solver_options=options
    )
    wallclock = time.perf_counter() - t0
    record_solve_metrics(
        n,
        output_dir=out,
        configuration=f"profiling/{name}",
        problem_class=problem_class,
        solver=SOLVER_NAME,
        solver_options=options,
        status=status,
        condition=condition,
        wallclock_s=wallclock,
        network_path=NETWORK_PATH,
    )
    save_outputs(n, out)
    print(f"[{name}] {status}/{condition} in {wallclock:.0f}s", flush=True)


def run_reserves() -> None:
    out = RESULTS_ROOT / "reserves"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = fresh_network()
    apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
    n.optimize.create_model()
    add_operational_reserve_margin(n, n.snapshots, **RESERVE_KWARGS)
    status, condition = n.optimize.solve_model(
        solver_name=SOLVER_NAME, solver_options=LP_OPTIONS
    )
    wallclock = time.perf_counter() - t0
    record_solve_metrics(
        n,
        output_dir=out,
        configuration="profiling/reserves",
        problem_class="LP",
        solver=SOLVER_NAME,
        solver_options=LP_OPTIONS,
        status=status,
        condition=condition,
        wallclock_s=wallclock,
        network_path=NETWORK_PATH,
    )
    save_outputs(n, out)
    print(f"[reserves] {status}/{condition} in {wallclock:.0f}s", flush=True)


def run_rolling(name: str, with_forecast: bool) -> None:
    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = fresh_network()
    apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
    soc = pypsa.Network(BASELINE_SOLVED).storage_units_t.state_of_charge
    ratios = (
        build_forecast_ratios(n, FORECASTS_DIR, GENERATION_DIR)
        if with_forecast
        else None
    )
    res = solve_rolling_horizon(
        n,
        horizon=24,
        ratios=ratios,
        soc_target=soc,
        solver_name=SOLVER_NAME,
        solver_options=LP_OPTIONS,
    )
    wallclock = time.perf_counter() - t0
    counts = res.condition.value_counts().to_dict()
    metrics = {
        "configuration": f"profiling/{name}",
        "problem_class": "LP (rolling, 24h windows)",
        "n_windows": int(len(res)),
        "window_conditions": {str(k): int(v) for k, v in counts.items()},
        "wallclock_s": round(wallclock, 1),
        "voll_eur_per_mwh": VOLL_EUR_PER_MWH,
        "forecast_information_set": bool(with_forecast),
    }
    (out / "solve_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_outputs(n, out)
    print(f"[{name}] {counts} in {wallclock:.0f}s", flush=True)


def main() -> None:
    import traceback

    only = set(sys.argv[1:])

    def wants(name: str) -> bool:
        return not only or name in only

    def guard(name: str, fn) -> None:
        """Run one config, log and continue on failure (per-config isolation)."""
        if not wants(name):
            return
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"[{name}] FAILED -- continuing with remaining configs", flush=True)

    def setup_elastic(n):
        apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
        prices, loads = zone_calibration(n)
        apply_price_elastic_demand(n, prices, loads)

    def setup_subsidy(n):
        apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
        apply_de_must_take(n)
        apply_de_subsidy_floor(n)

    def setup_uc(n):
        apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
        n_comm = apply_unit_commitment_csv(n, "pypsa-eur/data/unit_commitment.csv")
        print(f"  committable generators: {n_comm}", flush=True)

    # VOLL load shedding is the feasibility floor: without it the windowed
    # and reserve/UC steps go infeasible at scarcity hours. So every config
    # is baseline + VOLL + one constraint, measured on the same base; each
    # constraint's isolated cost is its config time minus the VOLL config.
    def setup_ramping(n):
        apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH)
        apply_ramp_limits(n, "pypsa-eur/data/unit_commitment.csv")

    guard("voll", lambda: run_annual(
        "voll",
        lambda n: apply_voll_load_shedding(n, voll_eur_per_mwh=VOLL_EUR_PER_MWH),
        "LP", LP_OPTIONS))
    guard("elastic", lambda: run_annual("elastic", setup_elastic, "QP", LP_OPTIONS))
    guard("ramping", lambda: run_annual("ramping", setup_ramping, "LP", LP_OPTIONS))
    guard("reserves", run_reserves)
    guard("subsidy_de", lambda: run_annual("subsidy_de", setup_subsidy, "LP", LP_OPTIONS))
    guard("rolling", lambda: run_rolling("rolling", with_forecast=False))
    guard("forecast", lambda: run_rolling("forecast", with_forecast=True))
    guard("uc", lambda: run_annual("uc", setup_uc, "MIP", MIP_OPTIONS))


if __name__ == "__main__":
    main()
