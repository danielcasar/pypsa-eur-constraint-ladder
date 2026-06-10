"""Activation helpers for PyPSA-Eur native constraint features.

Each function mutates the ``pypsa.Network`` in place and is designed
to be safe to call independently and idempotently within a single
ladder run. The reserves helper is the only one that requires a
linopy model to already exist (``n.optimize.create_model()``); the
others operate on the static ``Network`` object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import xarray as xr


# Default thermal carriers eligible for unit commitment. Matches the
# PyPSA-Eur convention.
DEFAULT_THERMAL_CARRIERS: tuple[str, ...] = (
    "coal",
    "lignite",
    "CCGT",
    "OCGT",
    "biomass",
    "oil",
    "nuclear",
)


def reset_to_lp_baseline(n: pypsa.Network) -> dict[str, pd.Series]:
    """Strip non-default LP / MIP features so the network solves as the baseline.

    The paper's baseline equates to "PyPSA-Eur with its default
    workflow output": ramping limits are kept (default-on), unit
    commitment and start-up costs are stripped (default-off and
    re-applied only at the UC ladder step). Stores the original
    commitment flags + start-up costs so :func:`enable_unit_commitment`
    can restore them at the relevant step.
    """
    snapshot: dict[str, pd.Series] = {
        "ramp_limit_up": n.generators["ramp_limit_up"].copy(),
        "ramp_limit_down": n.generators["ramp_limit_down"].copy(),
        "committable": n.generators["committable"].copy(),
        "start_up_cost": n.generators.get(
            "start_up_cost", pd.Series(0.0, index=n.generators.index)
        ).copy(),
        "min_up_time": n.generators.get(
            "min_up_time", pd.Series(0, index=n.generators.index)
        ).copy(),
        "min_down_time": n.generators.get(
            "min_down_time", pd.Series(0, index=n.generators.index)
        ).copy(),
    }
    # Ramping limits are part of the baseline (default-on across 6 of 7
    # surveyed ESOMs and active in PyPSA-Eur's default workflow); only
    # UC-related attributes are stripped.
    n.generators["committable"] = False
    if "start_up_cost" in n.generators.columns:
        n.generators["start_up_cost"] = 0.0
    return snapshot


def apply_voll_load_shedding(
    n: pypsa.Network,
    *,
    voll_eur_per_mwh: float = 8000.0,
    bus_carrier: str = "AC",
    carrier_name: str = "load_shedding_voll",
    p_nom_mw: float = 1.0e7,
) -> int:
    """Attach a flat-cost load-shedding generator at every AC bus.

    Marginal cost ``voll_eur_per_mwh`` and effectively unbounded
    capacity ``p_nom_mw``. The energy-balance dual is pinned to VOLL
    during scarcity events.

    Returns
    -------
    int
        Number of load-shedding generators added.
    """
    if carrier_name not in n.carriers.index:
        n.add("Carrier", carrier_name)
    ac_buses = n.buses[n.buses.carrier == bus_carrier].index
    added = 0
    for bus in ac_buses:
        gen_name = f"{bus} {carrier_name}"
        if gen_name in n.generators.index:
            continue
        n.add(
            "Generator",
            gen_name,
            bus=bus,
            carrier=carrier_name,
            p_nom=p_nom_mw,
            marginal_cost=voll_eur_per_mwh,
            sign=1.0,
        )
        added += 1
    return added


def restore_ramping(
    n: pypsa.Network, snapshot: dict[str, pd.Series]
) -> None:
    """Re-apply the ramp limits saved by :func:`reset_to_lp_baseline`."""
    n.generators["ramp_limit_up"] = snapshot["ramp_limit_up"]
    n.generators["ramp_limit_down"] = snapshot["ramp_limit_down"]


def enable_unit_commitment(
    n: pypsa.Network,
    snapshot: dict[str, pd.Series],
    *,
    committable_carriers: tuple[str, ...] = DEFAULT_THERMAL_CARRIERS,
    default_min_up_time_h: int = 1,
    default_min_down_time_h: int = 1,
    default_start_up_cost_eur_per_mw: float = 50.0,
) -> int:
    """Mark dispatchable thermal generators as committable.

    Restores any start-up costs and min-up/down times that were
    snapshotted by :func:`reset_to_lp_baseline`. Where the snapshotted
    value is zero, falls back to a small default so the MIP is
    meaningful.

    Returns
    -------
    int
        Number of generators marked committable.
    """
    sel = n.generators.carrier.isin(committable_carriers)
    n.generators.loc[sel, "committable"] = True

    # Restore start-up cost; default if snapshot had zero
    snap_su = snapshot["start_up_cost"].reindex(n.generators.index).fillna(0.0)
    needs_default = sel & (snap_su <= 0.0)
    n.generators.loc[sel & ~needs_default, "start_up_cost"] = snap_su[sel & ~needs_default]
    n.generators.loc[needs_default, "start_up_cost"] = (
        default_start_up_cost_eur_per_mw
        * n.generators.loc[needs_default, "p_nom"]
    )

    # Restore min-up/down; default if snapshot had zero
    for attr, default in (
        ("min_up_time", default_min_up_time_h),
        ("min_down_time", default_min_down_time_h),
    ):
        if attr not in n.generators.columns:
            n.generators[attr] = 0
        snap = snapshot[attr].reindex(n.generators.index).fillna(0)
        needs_default = sel & (snap <= 0)
        n.generators.loc[sel & ~needs_default, attr] = snap[sel & ~needs_default]
        n.generators.loc[needs_default, attr] = default

    return int(sel.sum())


def add_operational_reserve_margin(
    n: pypsa.Network,
    sns: pd.DatetimeIndex,
    *,
    epsilon_load: float = 0.03,
    epsilon_vres: float = 0.05,
    contingency_mw: float = 1500.0,
) -> None:
    """Add system-wide operating-reserve constraints to the active linopy model.

    Mirrors PyPSA-Eur's ``add_operational_reserve_margin`` (in
    ``scripts/solve_network.py``) with the constant-capacity case
    only (no extendable VRES). Must be called after
    ``n.optimize.create_model()`` and before ``n.optimize.solve_model()``.
    """
    if not hasattr(n, "model") or n.model is None:
        raise RuntimeError(
            "n.model is not set. Call n.optimize.create_model() before "
            "applying the reserve margin."
        )

    n.model.add_variables(
        0.0,
        np.inf,
        coords=[sns, n.generators.index],
        name="Generator-r",
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Total demand per snapshot
    demand = n.loads_t.p_set.reindex(sns).sum(axis=1)

    # VRES potential of (non-extendable) renewables
    vres_i = n.generators_t.p_max_pu.columns
    capacity_factor = n.generators_t.p_max_pu[vres_i].reindex(sns)
    p_nom = n.generators.p_nom[vres_i]
    potential = (capacity_factor * p_nom).sum(axis=1)

    rhs = epsilon_load * demand + epsilon_vres * potential + contingency_mw
    n.model.add_constraints(
        summed_reserve >= xr.DataArray(rhs.to_xarray()), name="reserve_margin"
    )

    # Joint capacity: p_g,t + r_g,t <= p_max_pu(t) * p_nom
    gen_i = n.generators.index
    p_nom_full = n.generators.p_nom.reindex(gen_i)
    p_max_pu = n.generators_t.p_max_pu.reindex(columns=gen_i).fillna(1.0)
    p_max_pu_full = p_max_pu.reindex(sns).fillna(method="ffill").fillna(1.0)
    rhs_capacity = xr.DataArray(
        (p_max_pu_full * p_nom_full).to_numpy(),
        coords={"snapshot": sns, "Generator": gen_i},
    )
    p_var = n.model["Generator-p"]
    n.model.add_constraints(
        p_var + reserve <= rhs_capacity, name="reserve_capacity"
    )
