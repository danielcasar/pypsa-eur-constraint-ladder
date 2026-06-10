"""Copy-paste walkthrough for manual constraint activation.

Run this from a Python session (or step through it in a notebook). Each
section solves one configuration and dumps shadow prices + objective to
``RESULTS_DIR / <config_name>/``. Skip or reorder as you like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypsa

# Make the sibling packages importable as top-level names without
# turning ``code/`` itself into a Python package (which would shadow the
# stdlib ``code`` module).
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from constraints import (  # noqa: E402
    DE_BID_BEHAVIOUR_2025,
    apply_de_must_take,
    apply_de_subsidy_floor,
    apply_price_elastic_demand,
)
from driver import (  # noqa: E402
    add_operational_reserve_margin,
    apply_voll_load_shedding,
    build_calibration,
    enable_unit_commitment,
    reset_to_lp_baseline,
    restore_ramping,
)


NETWORK_PATH = Path("pypsa-eur/results/eu_2025/networks/base_s_50_elec_.nc")
RESULTS_DIR = Path("results/eu_2025")
SOLVER = "gurobi"


def dump(n: pypsa.Network, name: str) -> None:
    """Write shadow prices and objective for a solved network."""
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    n.buses_t.marginal_price.to_csv(out / "shadow_price.csv")
    (out / "objective.txt").write_text(f"{n.objective:.6f}\n")


# ----------------------------------------------------------------------
# 0. Baseline LP -- default PyPSA-Eur dispatch.
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
baseline_snapshot = reset_to_lp_baseline(n)
n.optimize(solver_name=SOLVER)
dump(n, "baseline")


# ----------------------------------------------------------------------
# 1. + VOLL load shedding (LP, pins the dual at VOLL during scarcity).
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
n.optimize(solver_name=SOLVER)
dump(n, "voll")


# ----------------------------------------------------------------------
# 2. + price-elastic demand (LP -> QP).
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
prices, loads = build_calibration(n)  # add entsoe_price_csv=... if needed
apply_price_elastic_demand(
    n, zone_mean_price=prices, zone_mean_load=loads, elasticity_at_op=-0.05,
)
n.optimize(solver_name=SOLVER)
dump(n, "elastic_demand")


# ----------------------------------------------------------------------
# 3. + operating reserves (LP, requires create_model -> custom -> solve).
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
prices, loads = build_calibration(n)
apply_price_elastic_demand(
    n, zone_mean_price=prices, zone_mean_load=loads, elasticity_at_op=-0.05,
)
n.optimize.create_model()
add_operational_reserve_margin(
    n, n.snapshots,
    epsilon_load=0.03, epsilon_vres=0.05, contingency_mw=1500.0,
)
n.optimize.solve_model(solver_name=SOLVER)
dump(n, "reserves")


# ----------------------------------------------------------------------
# 4. + rolling horizon (LP, 24 h window with SoC carryover).
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
prices, loads = build_calibration(n)
apply_price_elastic_demand(
    n, zone_mean_price=prices, zone_mean_load=loads, elasticity_at_op=-0.05,
)
# Reserves intentionally skipped here -- combining reserves with
# rolling-horizon needs an extra_functionality callback, see notes.
n.optimize.optimize_with_rolling_horizon(
    horizon=24, overlap=0, solver_name=SOLVER,
)
dump(n, "rolling_horizon")


# ----------------------------------------------------------------------
# 5. + unit commitment (LP -> MIP).
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
baseline_snapshot = reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
prices, loads = build_calibration(n)
apply_price_elastic_demand(
    n, zone_mean_price=prices, zone_mean_load=loads, elasticity_at_op=-0.05,
)
enable_unit_commitment(n, baseline_snapshot)
n.optimize(solver_name=SOLVER, solver_options={"MIPGap": 0.005, "TimeLimit": 86400})
dump(n, "unit_commitment")


# ----------------------------------------------------------------------
# 6. + DE subsidy floor and must-take.
# ----------------------------------------------------------------------
n = pypsa.Network(NETWORK_PATH)
baseline_snapshot = reset_to_lp_baseline(n)
apply_voll_load_shedding(n, voll_eur_per_mwh=8000.0)
prices, loads = build_calibration(n)
apply_price_elastic_demand(
    n, zone_mean_price=prices, zone_mean_load=loads, elasticity_at_op=-0.05,
)
enable_unit_commitment(n, baseline_snapshot)
apply_de_must_take(n, params=DE_BID_BEHAVIOUR_2025)
apply_de_subsidy_floor(n, params=DE_BID_BEHAVIOUR_2025)
n.optimize(solver_name=SOLVER, solver_options={"MIPGap": 0.005, "TimeLimit": 86400})
dump(n, "subsidy_floor_de")
