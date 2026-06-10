"""Constraint implementations for the PyPSA-Eur constraint-ladder paper.

Each module exposes a single ``apply_*`` function that mutates a PyPSA
``Network`` in place. They are designed to be called after
``add_electricity.py`` has produced the prepared network and before
``solve_network.py`` (or its driver equivalent) is invoked.
"""

from .price_elastic_demand import (
    BROWN_NORM_A,
    BROWN_NORM_B,
    BROWN_NORM_D,
    BROWN_NORM_ELASTICITY,
    apply_price_elastic_demand,
)
from .subsidy_floor import apply_subsidy_floor
from .subsidy_floor_de import (
    CohortParameters,
    DE_BID_BEHAVIOUR_2025,
    apply_de_must_take,
    apply_de_subsidy_floor,
)
__all__ = [
    "apply_price_elastic_demand",
    "BROWN_NORM_A",
    "BROWN_NORM_B",
    "BROWN_NORM_D",
    "BROWN_NORM_ELASTICITY",
    "apply_subsidy_floor",
    "apply_de_must_take",
    "apply_de_subsidy_floor",
    "CohortParameters",
    "DE_BID_BEHAVIOUR_2025",
]
