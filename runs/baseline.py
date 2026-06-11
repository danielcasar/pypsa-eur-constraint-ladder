"""Baseline LP dispatch on the continental 2024 prepared network.

Solves the default PyPSA-Eur dispatch with no ladder constraints active
(ramping is kept as part of the default baseline; UC and start-up costs
are stripped via ``reset_to_lp_baseline``). Writes year-long per-zone
shadow prices to ``results/baseline/shadow_price.csv``.

Wallclock on Marvin (Threadripper 3960X, 20 threads, Gurobi 13 barrier
no-crossover): ~13 min for a 8760 h x 50 zone x 507 generator LP.

Run with the mamba env active:

    mamba activate pypsa-eur-ladder
    python runs/baseline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pypsa

from constraint_ladder.helpers import reset_to_lp_baseline


NETWORK_PATH = Path("pypsa-eur/resources/eu_2024_dispatch/networks/base_s_adm_elec_.nc")
RESULTS_DIR = Path("results/baseline")
SOLVER_OPTIONS = {
    "Method": 2,        # barrier
    "Crossover": 0,     # skip crossover; barrier point already has duals
    "BarConvTol": 1e-6,
    "FeasibilityTol": 1e-5,
    "OptimalityTol": 1e-5,
    "Threads": 20,
    "Seed": 123,
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n = pypsa.Network(NETWORK_PATH)
    print(
        f"loaded: {len(n.snapshots)} snapshots x {len(n.buses)} buses "
        f"x {len(n.generators)} generators",
        flush=True,
    )

    reset_to_lp_baseline(n)
    print("reset done; solving with barrier no-crossover...", flush=True)

    status, condition = n.optimize(solver_name="gurobi", solver_options=SOLVER_OPTIONS)
    elapsed = time.perf_counter() - t0
    print(
        f"status={status} condition={condition} elapsed={elapsed:.1f}s "
        f"objective={n.objective:.3e}",
        flush=True,
    )

    n.buses_t.marginal_price.to_csv(RESULTS_DIR / "shadow_price.csv")
    print(f"shadow prices written to {RESULTS_DIR / 'shadow_price.csv'}", flush=True)

    print("per-zone year-average shadow price (top 10):")
    print(n.buses_t.marginal_price.mean().sort_values(ascending=False).head(10))
    print("per-zone year-average (bottom 10):")
    print(n.buses_t.marginal_price.mean().sort_values().head(10))


if __name__ == "__main__":
    main()
