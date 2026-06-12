"""Annual-energy anchoring of wind profiles to observed 2024 generation.

The 2024 SARAH3+ERA5 cutout shows the documented ERA5 regional wind
bias: over-prediction in flat / offshore northern Europe (GB +52 %,
NL +29 %, DE +22 % against Ember 2024) and under-prediction in complex
Mediterranean terrain (IT -43 %, ES -29 %). Wake, availability, and
curtailment losses are also absent from atlite's raw capacity factors.

``anchor_wind_generation`` rescales each country's wind ``p_max_pu``
time series by a single factor so that the annual *available* wind
energy matches Ember's observed 2024 generation, keeping the hourly
shape from the 2024 meteorology. Factors above one are clipped at
``p_max_pu = 1`` and the factor is re-converged iteratively so the
annual target is still met. The same factor applies to all bidding
zones and all wind carriers of a country (Ember reports country-level
wind without an onshore/offshore split).

Observed-generation anchoring slightly understates *available* energy
in countries with material curtailment (observed = available minus
curtailed), which is conservative for the price analysis.

Source of the targets: Ember, Yearly electricity data (full release),
retrieved 2026-06-12; category "Electricity generation", variable
"Wind", year 2024, TWh.
"""

from __future__ import annotations

import pandas as pd
import pypsa

WIND_CARRIERS: tuple[str, ...] = (
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
)

# Ember 2024 wind generation, TWh, per country (ISO2).
EMBER_WIND_2024_TWH: dict[str, float] = {
    "AT": 9.1,
    "BA": 0.4,
    "BE": 14.1,
    "BG": 1.5,
    "CH": 0.2,
    "CZ": 0.7,
    "DE": 141.6,
    "DK": 20.4,
    "EE": 1.0,
    "ES": 62.2,
    "FI": 20.5,
    "FR": 45.4,
    "GB": 83.3,
    "GR": 12.6,
    "HR": 2.7,
    "HU": 0.7,
    "IE": 11.6,
    "IT": 22.2,
    "LT": 3.4,
    "LU": 0.5,
    "LV": 0.3,
    "ME": 0.3,
    "MK": 0.2,
    "NL": 33.5,
    "NO": 14.6,
    "PL": 25.9,
    "PT": 14.3,
    "RO": 6.4,
    "RS": 1.2,
    "SE": 40.4,
    "SI": 0.0,
    "SK": 0.0,
    "XK": 0.4,
}


def anchor_wind_generation(
    n: pypsa.Network,
    targets_twh: dict[str, float] | None = None,
    *,
    carriers: tuple[str, ...] = WIND_CARRIERS,
    tol: float = 0.005,
    max_iter: int = 25,
) -> pd.DataFrame:
    """Scale wind ``p_max_pu`` per country to match observed annual energy.

    Returns a DataFrame indexed by country with columns
    ``potential_twh`` (before), ``target_twh``, ``achieved_twh``,
    ``factor``. Countries without a target or without wind generators
    are skipped (profile left untouched).
    """
    if targets_twh is None:
        targets_twh = EMBER_WIND_2024_TWH

    gens = n.generators
    wind = gens[gens.carrier.isin(carriers)]
    weights = n.snapshot_weightings.generators

    rows = []
    for country, target in sorted(targets_twh.items()):
        idx = wind[wind.bus.str[:2] == country].index
        idx = idx.intersection(n.generators_t.p_max_pu.columns)
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
                "country": country,
                "potential_twh": round(potential, 2),
                "target_twh": target,
                "achieved_twh": round(achieved, 2),
                "factor": round(factor, 4),
            }
        )

    return pd.DataFrame(rows).set_index("country")
