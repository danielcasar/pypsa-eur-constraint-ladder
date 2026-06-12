"""Annual-energy anchoring of wind and solar profiles to observed 2024.

The raw cutout profiles carry the documented reanalysis wind bias
(Staffell & Pfenninger 2016; Jourdier 2020): over-prediction over flat
northern Europe and offshore, under-prediction in complex southern
terrain, plus missing wake / availability / curtailment losses. Solar
profiles carry a smaller level bias entangled with the AC/DC capacity
convention.

``anchor_vre_generation`` rescales the ``p_max_pu`` time series of each
anchoring group (one or more bidding zones sharing one observed total)
by a single factor so that annual available energy matches the observed
2024 generation, keeping the hourly shape from the 2024 meteorology.
Factors above one are clipped at ``p_max_pu = 1`` and re-converged
iteratively so the annual target is still met.

Targets come from ``constraint_ladder/data/vre_zone_targets_2024.csv``,
built by ``Analysis/build_vre_targets_2024.py`` from ENTSO-E actual
generation per bidding zone (hourly-resampled), with Ember 2024 country
totals substituting where the Transparency Platform under-reports
(distributed solar) or lacks coverage entirely (GB, western Balkans).
The basis used for every group is recorded in the CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypsa

TARGETS_CSV = Path(__file__).resolve().parent.parent / "data" / "vre_zone_targets_2024.csv"

CARRIER_GROUPS: dict[str, tuple[str, ...]] = {
    "wind": ("onwind", "offwind-ac", "offwind-dc", "offwind-float"),
    "solar": ("solar", "solar-hsat"),
}


def anchor_vre_generation(
    n: pypsa.Network,
    targets_csv: Path | str = TARGETS_CSV,
    *,
    tol: float = 0.005,
    max_iter: int = 25,
) -> pd.DataFrame:
    """Scale wind and solar ``p_max_pu`` per anchoring group to observed energy.

    Returns a DataFrame indexed by group with columns ``carrier_group``,
    ``potential_twh`` (before), ``target_twh``, ``achieved_twh``,
    ``factor``, ``basis``. Groups without matching generators are
    skipped.
    """
    targets = pd.read_csv(targets_csv)
    gens = n.generators
    weights = n.snapshot_weightings.generators

    rows = []
    for _, t in targets.iterrows():
        zones = t["zones"].split(";")
        carriers = CARRIER_GROUPS[t["carrier_group"]]
        target = float(t["target_twh"])
        sel = gens[gens.carrier.isin(carriers) & gens.bus.isin(zones)]
        idx = sel.index.intersection(n.generators_t.p_max_pu.columns)
        if not len(idx) or target <= 0:
            continue
        orig = n.generators_t.p_max_pu[idx].copy()
        p_nom = gens.p_nom[idx]
        potential = (orig.mul(weights, axis=0) * p_nom).sum().sum() / 1e6
        if potential <= 0:
            continue
        factor = target / potential
        scaled = orig
        achieved = potential
        for _ in range(max_iter):
            scaled = (orig * factor).clip(upper=1.0)
            achieved = (scaled.mul(weights, axis=0) * p_nom).sum().sum() / 1e6
            if abs(achieved - target) / target < tol:
                break
            factor *= target / achieved
        n.generators_t.p_max_pu.loc[:, idx] = scaled
        rows.append(
            {
                "group": t["group"],
                "carrier_group": t["carrier_group"],
                "potential_twh": round(potential, 2),
                "target_twh": round(target, 2),
                "achieved_twh": round(achieved, 2),
                "factor": round(factor, 4),
                "basis": t["basis"],
            }
        )

    return pd.DataFrame(rows).set_index("group")
