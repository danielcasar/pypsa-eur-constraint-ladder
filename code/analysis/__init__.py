"""Comparison pipeline: per-configuration shadow prices -> Delta kappa_d closure shares.

Wraps Paper 1's three-stage framework (Caesar 2026) so it can be
called once per configuration in the ladder and once per modelled
bidding zone, producing the closure-share matrix that is the
headline result of this paper.
"""

from .closure_shares import (
    LADDER_CONFIGURATIONS,
    closure_shares_per_zone,
    compute_kappa_per_config,
    load_per_config_shadow_prices,
)

__all__ = [
    "LADDER_CONFIGURATIONS",
    "closure_shares_per_zone",
    "compute_kappa_per_config",
    "load_per_config_shadow_prices",
]
