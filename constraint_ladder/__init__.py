"""Constraint ladder for the PyPSA-Eur model-market shadow-price paper.

Top-level package re-exporting the two functional groups:

- ``constraint_ladder.constraints`` -- per-deviation activator functions
  (``apply_de_subsidy_floor``, ``apply_price_elastic_demand``, ...) that
  the paper's manual ladder turns on one at a time.
- ``constraint_ladder.helpers`` -- native PyPSA / PyPSA-Eur helpers
  (``reset_to_lp_baseline``, ``apply_voll_load_shedding``,
  ``enable_unit_commitment``, ``add_operational_reserve_margin``,
  ``build_calibration``).

Typical run script (see ``runs/baseline.py`` for a working example):

    import pypsa
    from constraint_ladder.helpers import reset_to_lp_baseline

    n = pypsa.Network(NETWORK_PATH)
    reset_to_lp_baseline(n)
    n.optimize(solver_name="gurobi", solver_options=...)
"""

from . import constraints, helpers

__all__ = ["constraints", "helpers"]
