"""Price-elastic demand via PWL log-log approximation.

Implements the demand-side elasticity step of the constraint ladder
following Brown, Neumann & Riepin (2025, Energy Economics 147, 108483),
Section 3.2, and the official PyPSA ``demand-elasticity`` example
(https://docs.pypsa.org/latest/examples/demand-elasticity/).

For each AC bus z, three load-shedding generators are added,
approximating a constant-elasticity log-log demand curve in
piecewise-linear form. The shape is calibrated so the curve passes
through the empirical zone-mean operating point (zone-mean wholesale
price ``p_z``, zone-mean hourly load ``d_z``) with a local elasticity
of ``elasticity_at_op`` at that point.

PyPSA encoding: per segment k, ``marginal_cost = a_k - b_k * D_k``
and ``marginal_cost_quadratic = b_k / 2``, with ``sign = 1`` so the
generator acts as load shedding. The flat-cost VOLL load-shedding
component from the preceding ladder step (added by PyPSA-Eur's
``solving.options.load_shedding`` mechanism) is retained and sits
above the elastic segments as the inelastic top of the demand stack.

The aggregate three-segment demand curve has an inelastic bulk (the
first segment covers ~95 percent of zone load with shallow slope), a
moderately elastic middle (the next ~5 percent), and a more elastic
tail (the last ~10 percent capacity), reproducing the qualitative
shape used by Brown et al. for their German PyPSA experiments.
"""

from __future__ import annotations

import pandas as pd
import pypsa


# Brown 2025 Sec. 3.2 default segment shape, normalized to a
# (d_op = 100 MW, p_op = 100 EUR/MWh) operating point.
BROWN_NORM_A = (8000.0, 400.0, 200.0)   # intercepts a_k (EUR/MWh)
BROWN_NORM_B = (80.0, 40.0, 20.0)       # slopes b_k (EUR / MW^2 h)
BROWN_NORM_D = (95.5, 5.0, 10.0)        # segment widths D_k (MW)
BROWN_NORM_P_OP = 100.0                 # reference price (EUR/MWh)
BROWN_NORM_D_OP = 100.0                 # reference load (MW)
BROWN_NORM_ELASTICITY = -0.05           # -5% at op point


def apply_price_elastic_demand(
    n: pypsa.Network,
    zone_mean_price: dict[str, float],
    zone_mean_load: dict[str, float],
    *,
    elasticity_at_op: float = BROWN_NORM_ELASTICITY,
    bus_carrier: str = "AC",
    carrier_name: str = "load_elastic",
) -> pd.DataFrame:
    """Add three PWL load-shedding generators per AC bus.

    The three generators together approximate a log-log demand curve
    with the given local elasticity at the zone-mean operating point.

    Parameters
    ----------
    n
        PyPSA Network. Mutated in place.
    zone_mean_price
        Dict mapping bus name to the empirical 2025 mean wholesale
        price (EUR/MWh). Typically the mean of the ENTSO-E day-ahead
        time series for the bus's bidding zone.
    zone_mean_load
        Dict mapping bus name to the empirical 2025 mean hourly load
        (MW). Typically ``n.loads_t.p_set[bus].mean()``.
    elasticity_at_op
        Target local elasticity at the zone operating point. Default
        -5 percent per Hirth, Khanna & Ruhnau (2024) and Arnold
        (2023) for Germany. The same value is applied across zones
        in the absence of zone-specific elasticity measurements; this
        is a documented limitation of the implementation.
    bus_carrier
        Bus carrier to apply elasticity to. Default ``"AC"``, the
        PyPSA-Eur electricity bus carrier.
    carrier_name
        Carrier name for the added generators. Default
        ``"load_elastic"``. The carrier is created if not present.

    Returns
    -------
    DataFrame indexed by ``(bus, segment)`` with the per-segment
    parameters actually written into the network: ``p_nom_mw``,
    ``marginal_cost_eur_per_mwh``, ``marginal_cost_quadratic``, and
    the resulting ``local_elasticity_at_op`` for the segment.
    """
    if elasticity_at_op >= 0:
        raise ValueError(
            f"elasticity_at_op must be negative; got {elasticity_at_op}"
        )

    if carrier_name not in n.carriers.index:
        n.add("Carrier", carrier_name)

    # Slope rescaling factor: preserves Brown's segment shape but
    # adjusts slopes so the local elasticity at the zone operating
    # point matches the requested target.
    elasticity_ratio = elasticity_at_op / BROWN_NORM_ELASTICITY

    applied: list[dict] = []
    for bus_name in n.buses.index:
        if n.buses.at[bus_name, "carrier"] != bus_carrier:
            continue
        if bus_name not in zone_mean_price or bus_name not in zone_mean_load:
            continue

        p_z = float(zone_mean_price[bus_name])
        d_z = float(zone_mean_load[bus_name])
        if p_z <= 0 or d_z <= 0:
            continue

        scale_a = p_z / BROWN_NORM_P_OP
        scale_D = d_z / BROWN_NORM_D_OP
        # b_z = b_norm * (p_z / d_z) / (P_OP / D_OP) / elasticity_ratio
        scale_b = (
            (p_z / d_z)
            / (BROWN_NORM_P_OP / BROWN_NORM_D_OP)
            / elasticity_ratio
        )

        for k, (a_norm, b_norm, D_norm) in enumerate(
            zip(BROWN_NORM_A, BROWN_NORM_B, BROWN_NORM_D), start=1
        ):
            a_k = a_norm * scale_a
            b_k = b_norm * scale_b
            D_k = D_norm * scale_D
            c_k = a_k - b_k * D_k          # linear coefficient (EUR/MWh)
            mcq_k = b_k / 2.0              # PyPSA marginal_cost_quadratic

            gen_name = f"{bus_name} {carrier_name} {k}"
            n.add(
                "Generator",
                gen_name,
                bus=bus_name,
                carrier=carrier_name,
                p_nom=D_k,
                marginal_cost=c_k,
                marginal_cost_quadratic=mcq_k,
                sign=1.0,
            )
            applied.append(
                {
                    "bus": bus_name,
                    "segment": k,
                    "p_nom_mw": D_k,
                    "marginal_cost_eur_per_mwh": c_k,
                    "marginal_cost_quadratic": mcq_k,
                    "local_elasticity_at_op": -(1.0 / b_k) * (p_z / d_z),
                }
            )

    if not applied:
        return pd.DataFrame()
    return pd.DataFrame(applied).set_index(["bus", "segment"])
