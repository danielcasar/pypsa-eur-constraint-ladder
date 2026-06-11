"""Subsidy-floor constraint for fixed-tariff-supported variable renewables.

For each subsidised VRE generator the per-MWh fixed-tariff support level
is encoded as a marginal-cost override: ``marginal_cost = -tariff``.
This makes the generator's effective bid floor equal to the negative
of its tariff, reproducing the empirical EEG-driven bid behaviour
documented by Nicolosi (2010, Energy Policy 38(11), 7257-7268). The
closest existing implementation lives in the AMIRIS agent-based model
(Frey et al. 2020, Energies 13(20):5350), where the analogous
behaviour is encoded as a bidding-strategy property of the
``RenewableTrader`` rather than as an LP cost coefficient.
"""

from __future__ import annotations

import pandas as pd
import pypsa


def apply_subsidy_floor(n: pypsa.Network, tariffs: pd.Series) -> None:
    """Mutate ``n`` so that each subsidised VRE generator bids at ``-tariff``.

    Parameters
    ----------
    n
        PyPSA Network. Mutated in place.
    tariffs
        Series indexed by generator name (a subset of
        ``n.generators.index``) giving the per-MWh fixed-tariff support
        level in EUR/MWh. Generators absent from the index are left at
        their existing ``marginal_cost`` (typically 0 for must-take VRE).

    Raises
    ------
    KeyError
        If ``tariffs`` references generator names that do not exist in
        ``n.generators``.
    ValueError
        If any tariff is negative.
    """
    if (tariffs < 0).any():
        bad = tariffs[tariffs < 0].index.tolist()
        raise ValueError(f"Tariffs must be non-negative; got negatives for {bad}")

    missing = tariffs.index.difference(n.generators.index)
    if len(missing) > 0:
        raise KeyError(f"Tariff entries without matching generators: {list(missing)}")

    n.generators.loc[tariffs.index, "marginal_cost"] = -tariffs.values
