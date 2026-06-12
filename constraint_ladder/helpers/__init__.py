"""Building blocks for manual step-by-step constraint activation on a
PyPSA-Eur network.

Use these helpers in an interactive Python session or notebook to apply
one constraint at a time and inspect the resulting shadow prices before
deciding whether to activate the next one. See ``runs/baseline.py``
in the repo root for a self-contained example.
"""

from .calibration import build_calibration
from .metrics import record_solve_metrics
from .native_constraints import (
    DEFAULT_THERMAL_CARRIERS,
    add_operational_reserve_margin,
    apply_ramp_limits,
    apply_unit_commitment_csv,
    apply_voll_load_shedding,
    enable_unit_commitment,
    reset_to_lp_baseline,
    restore_ramping,
)
from .rolling_horizon import build_forecast_ratios, solve_rolling_horizon
from .vre_anchoring import anchor_vre_generation

__all__ = [
    "DEFAULT_THERMAL_CARRIERS",
    "add_operational_reserve_margin",
    "anchor_vre_generation",
    "apply_ramp_limits",
    "apply_unit_commitment_csv",
    "apply_voll_load_shedding",
    "build_calibration",
    "build_forecast_ratios",
    "enable_unit_commitment",
    "record_solve_metrics",
    "reset_to_lp_baseline",
    "restore_ramping",
    "solve_rolling_horizon",
]
