"""Germany-specific EEG bid-behaviour implementation (percentage-based).

Splits each DE VRE aggregate generator into three behavioural cohorts
governed by three published-aggregate parameters per technology:

- ``must_take_share``: fraction of installed capacity below the §6 EEG
  2014 100 kW remote-curtailability threshold. These plants
  physically feed in whenever the resource is available regardless
  of the spot price; encoded by a sister generator with
  ``p_min_pu = p_max_pu``. Applied by :func:`apply_de_must_take`.

- ``subsidy_floor_share``: fraction of installed capacity above the
  curtailability threshold, within the 20-year EEG support window,
  and not subject to the §24/§51 EEG negative-price suspension rule.
  These plants curtail when the spot price falls below the negative
  of their fixed tariff; encoded by a sister generator with
  ``marginal_cost = -subsidy_floor_avg_tariff``. Applied by
  :func:`apply_de_subsidy_floor`.

- ``subsidy_floor_avg_tariff_eur_per_mwh``: the capacity-weighted
  average EEG tariff (EUR/MWh) across the subsidy-floor cohort's
  vintages, derived from aggregate EEG-payment statistics rather
  than per-plant disaggregation.

The remainder of each generator's capacity falls in the price-aware
cohort (post-subsidy, suspension-eligible, or unsubsidised) and is
left at PyPSA-Eur's default zero-marginal-cost behaviour.

The empirical bid-floor mechanism is documented by Nicolosi (2010,
Energy Policy 38(11), 7257-7268). The closest open-source
implementation lives in AMIRIS (Frey et al. 2020,
Energies 13(20), 5350) under the variable market-premium scheme.

Default parameter values for Germany 2025 are drawn from published
aggregate statistics (BNetzA EEG Monitoring, BMWK annual EEG data,
Fraunhofer ISE Energy Charts, Fraunhofer IEE Windmonitor); see
Appendix table ``tab:de_bid_behaviour_params`` of the paper for
exact citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pypsa


@dataclass(frozen=True)
class CohortParameters:
    """Per-technology bid-behaviour parameters for the percentage-based model."""

    must_take_share: float
    subsidy_floor_share: float
    subsidy_floor_avg_tariff_eur_per_mwh: float

    def __post_init__(self) -> None:
        for name in ("must_take_share", "subsidy_floor_share"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value}")
        if self.subsidy_floor_avg_tariff_eur_per_mwh < 0:
            raise ValueError(
                "subsidy_floor_avg_tariff_eur_per_mwh must be non-negative; "
                f"got {self.subsidy_floor_avg_tariff_eur_per_mwh}"
            )
        if self.must_take_share + self.subsidy_floor_share > 1.0 + 1e-9:
            raise ValueError(
                "must_take_share + subsidy_floor_share must not exceed 1.0; "
                f"got {self.must_take_share + self.subsidy_floor_share}"
            )


DE_BID_BEHAVIOUR_2025: dict[str, CohortParameters] = {
    "solar": CohortParameters(
        must_take_share=0.65,
        subsidy_floor_share=0.20,
        subsidy_floor_avg_tariff_eur_per_mwh=280.0,
    ),
    "onwind": CohortParameters(
        must_take_share=0.02,
        subsidy_floor_share=0.55,
        subsidy_floor_avg_tariff_eur_per_mwh=55.0,
    ),
    "offwind": CohortParameters(
        must_take_share=0.00,
        subsidy_floor_share=0.45,
        subsidy_floor_avg_tariff_eur_per_mwh=37.0,
    ),
    "biomass": CohortParameters(
        must_take_share=0.15,
        subsidy_floor_share=0.60,
        subsidy_floor_avg_tariff_eur_per_mwh=150.0,
    ),
}


def _generator_p_max_pu(n: pypsa.Network, gen_name: str) -> pd.Series:
    """Return per-snapshot ``p_max_pu`` for a generator (time-varying or scalar)."""
    if gen_name in n.generators_t.p_max_pu.columns:
        return n.generators_t.p_max_pu[gen_name].copy()
    scalar = n.generators.at[gen_name, "p_max_pu"]
    return pd.Series(scalar, index=n.snapshots, name=gen_name)


def _add_sister_generator(
    n: pypsa.Network,
    parent: str,
    sister: str,
    sister_p_nom_mw: float,
    *,
    marginal_cost: float = 0.0,
    must_take: bool = False,
) -> None:
    """Add a sister generator and reduce the parent's ``p_nom`` accordingly."""
    parent_p_nom = n.generators.at[parent, "p_nom"]
    if sister_p_nom_mw > parent_p_nom + 1e-6:
        raise ValueError(
            f"sister capacity {sister_p_nom_mw:.1f} MW exceeds parent "
            f"{parent!r} p_nom {parent_p_nom:.1f} MW"
        )

    n.generators.loc[sister] = n.generators.loc[parent].copy()
    n.generators.at[sister, "p_nom"] = sister_p_nom_mw
    n.generators.at[sister, "marginal_cost"] = marginal_cost

    profile = _generator_p_max_pu(n, parent)
    n.generators_t.p_max_pu[sister] = profile
    if must_take:
        n.generators_t.p_min_pu[sister] = profile

    n.generators.at[parent, "p_nom"] = parent_p_nom - sister_p_nom_mw


def apply_de_must_take(
    n: pypsa.Network,
    params: dict[str, CohortParameters] | None = None,
    *,
    zone: str = "DE",
) -> pd.DataFrame:
    """Force dispatch on the must-take share of each DE VRE aggregate generator.

    For each technology in ``params``, splits the matching PyPSA-Eur
    generator ``f"{zone} {technology}"`` into the original (residual)
    and a sister generator ``f"{zone} {technology} must_take"`` with
    capacity ``must_take_share × P_g^nom`` and
    ``p_min_pu = p_max_pu`` (forced dispatch). Generators not present
    in ``n.generators`` are skipped.

    Returns
    -------
    DataFrame indexed by technology with the per-technology
    ``must_take_mw`` actually written into the network.
    """
    params = params or DE_BID_BEHAVIOUR_2025
    applied: dict[str, float] = {}
    for tech, p in params.items():
        parent = f"{zone} {tech}"
        if parent not in n.generators.index or p.must_take_share <= 0:
            continue
        sister_mw = p.must_take_share * n.generators.at[parent, "p_nom"]
        _add_sister_generator(
            n,
            parent,
            f"{parent} must_take",
            sister_mw,
            marginal_cost=0.0,
            must_take=True,
        )
        applied[tech] = sister_mw
    return pd.DataFrame({"must_take_mw": applied})


def apply_de_subsidy_floor(
    n: pypsa.Network,
    params: dict[str, CohortParameters] | None = None,
    *,
    zone: str = "DE",
) -> pd.DataFrame:
    """Apply the EEG bid floor to the curtailable subsidy-floor cohort.

    For each technology in ``params``, splits the matching PyPSA-Eur
    generator into the original (residual) and a sister generator
    ``f"{zone} {technology} floor"`` with capacity
    ``subsidy_floor_share × P_g^nom`` and
    ``marginal_cost = -subsidy_floor_avg_tariff``. Generators not
    present in ``n.generators`` are skipped.

    Returns
    -------
    DataFrame indexed by technology with the per-technology
    ``subsidy_floor_mw`` and ``floor_eur_per_mwh`` actually written
    into the network.
    """
    params = params or DE_BID_BEHAVIOUR_2025
    applied = []
    for tech, p in params.items():
        parent = f"{zone} {tech}"
        if parent not in n.generators.index:
            continue
        if p.subsidy_floor_share <= 0 or p.subsidy_floor_avg_tariff_eur_per_mwh <= 0:
            continue
        sister_mw = p.subsidy_floor_share * n.generators.at[parent, "p_nom"]
        _add_sister_generator(
            n,
            parent,
            f"{parent} floor",
            sister_mw,
            marginal_cost=-p.subsidy_floor_avg_tariff_eur_per_mwh,
            must_take=False,
        )
        applied.append(
            {
                "technology": tech,
                "subsidy_floor_mw": sister_mw,
                "floor_eur_per_mwh": -p.subsidy_floor_avg_tariff_eur_per_mwh,
            }
        )
    return (
        pd.DataFrame(applied).set_index("technology")
        if applied
        else pd.DataFrame()
    )
