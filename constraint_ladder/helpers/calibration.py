"""Per-zone calibration data loader for the elasticity step.

The elasticity constraint takes per-bus mean wholesale price and
mean hourly load as anchor points. For the continental run these
should come from the ENTSO-E day-ahead price record and load time
series. For the test/de-1week network we fall back to the network's
own mean load and a fixed reference price.

The loader returns two dicts keyed by ``pypsa.Network`` bus name.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pypsa


# Fallback used when no ENTSO-E series is available for the bus's
# bidding zone. Set to the long-run German wholesale mean for 2025.
FALLBACK_MEAN_PRICE_EUR_PER_MWH = 80.0


def zone_mean_load_from_network(n: pypsa.Network) -> dict[str, float]:
    """Mean hourly load per AC bus, computed directly from ``n.loads_t.p_set``.

    The PyPSA-Eur convention is one Load component per AC bus, named
    identically to the bus. Where a bus has no load, the bus is
    omitted.
    """
    out: dict[str, float] = {}
    ac_buses = n.buses[n.buses.carrier == "AC"].index
    p_set = n.loads_t.p_set
    for bus in ac_buses:
        if bus in p_set.columns:
            out[bus] = float(p_set[bus].mean())
        else:
            # Some PyPSA-Eur conventions store the load name as the bus
            # name plus a suffix; try the bus prefix.
            candidates = [c for c in p_set.columns if c.startswith(bus)]
            if candidates:
                out[bus] = float(p_set[candidates].sum(axis=1).mean())
    return out


def zone_mean_price_from_entsoe(
    n: pypsa.Network,
    *,
    entsoe_price_csv: Path | str | None = None,
    fallback_eur_per_mwh: float = FALLBACK_MEAN_PRICE_EUR_PER_MWH,
    zone_for_bus: dict[str, str] | None = None,
) -> dict[str, float]:
    """Per-AC-bus mean wholesale price.

    If an ENTSO-E day-ahead price CSV is supplied (columns are
    bidding-zone codes, rows are hourly timestamps), the loader maps
    each bus to its bidding zone via ``zone_for_bus`` and returns
    the mean of the matching column. Without a CSV, every bus gets
    ``fallback_eur_per_mwh``.

    The ``zone_for_bus`` mapping defaults to ``bus_name[:2]`` (e.g.
    ``"DE0 0" -> "DE"``), matching PyPSA-Eur's continental
    convention.
    """
    out: dict[str, float] = {}
    ac_buses = n.buses[n.buses.carrier == "AC"].index

    if entsoe_price_csv is None or not Path(entsoe_price_csv).exists():
        for bus in ac_buses:
            out[bus] = fallback_eur_per_mwh
        return out

    df = pd.read_csv(entsoe_price_csv, index_col=0, parse_dates=True)
    zone_for_bus = zone_for_bus or {bus: bus[:2] for bus in ac_buses}

    for bus in ac_buses:
        zone = zone_for_bus.get(bus, bus[:2])
        if zone in df.columns:
            series = df[zone].dropna()
            out[bus] = float(series.mean()) if len(series) else fallback_eur_per_mwh
        else:
            out[bus] = fallback_eur_per_mwh
    return out


def build_calibration(
    n: pypsa.Network,
    *,
    entsoe_price_csv: Path | str | None = None,
    fallback_price_eur_per_mwh: float = FALLBACK_MEAN_PRICE_EUR_PER_MWH,
    zone_for_bus: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convenience wrapper that returns ``(zone_mean_price, zone_mean_load)``
    in the shape expected by :func:`apply_price_elastic_demand`.
    """
    prices = zone_mean_price_from_entsoe(
        n,
        entsoe_price_csv=entsoe_price_csv,
        fallback_eur_per_mwh=fallback_price_eur_per_mwh,
        zone_for_bus=zone_for_bus,
    )
    loads = zone_mean_load_from_network(n)
    return prices, loads
