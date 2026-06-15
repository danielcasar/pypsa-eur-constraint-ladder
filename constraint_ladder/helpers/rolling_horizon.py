"""Rolling-horizon dispatch: foresight truncation and forecast error.

Two ladder steps share this module:

* **Step (i) -- foresight truncation.** ``solve_rolling_horizon`` with
  ``ratios=None`` solves the year in sequential 24 h windows on the
  *realised* series, carrying storage state of charge between windows.
  This isolates the effect of losing the annual arbitrage horizon while
  keeping perfect within-window information (the deterministic
  rolling-horizon paradigm PyPSA ships natively; re-implemented here so
  both steps share one code path).

* **Step (ii) -- forecast error.** The same loop with ``ratios`` from
  :func:`build_forecast_ratios`: within each window the wind / solar
  ``p_max_pu`` and the load ``p_set`` are multiplied by the historical
  D-1 forecast / realised ratio from the ENTSO-E Transparency Platform,
  so the dispatch LP clears on the same information set EUPHEMIA saw at
  gate closure. Duals from these windows are day-ahead prices formed on
  forecasts. Storage state is chained on the forecast-cleared schedule
  (the standard day-ahead simplification; balancing is out of scope).

Ratios are built from the same platform on both sides
(forecast / realised), so level conventions cancel and the anchored
calibration of the profiles is preserved -- only the forecast error is
injected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# ENTSO-E zone -> PyPSA zone prefixes (same mapping as the VRE targets).
ZONE_MAP: dict[str, list[str]] = {
    "AT": ["AT00"], "BE": ["BE00"], "BG": ["BG00"], "CH": ["CH00"],
    "CZ": ["CZ00"], "DE_LU": ["DE00", "LUG1"],
    "DK_1": ["DKW1"], "DK_2": ["DKE1"], "EE": ["EE00"], "ES": ["ES00"],
    "FI": ["FI00"], "FR": ["FR00"], "GR": ["GR00"], "HR": ["HR00"],
    "HU": ["HU00"], "IE_SEM": ["IE00", "GBNI"],
    "IT_NORD": ["ITN1"], "IT_CNOR": ["ITCN"], "IT_CSUD": ["ITCS"],
    "IT_SUD": ["ITS1"], "IT_CALA": ["ITCA"], "IT_SICI": ["ITSI"],
    "IT_SARD": ["ITSA"], "LT": ["LT00"], "LV": ["LV00"], "NL": ["NL00"],
    "NO_1": ["NOS1"], "NO_2": ["NOS2"], "NO_3": ["NOM1"],
    "NO_4": ["NON1"], "NO_5": ["NOS5"], "PL": ["PL00"], "PT": ["PT00"],
    "RO": ["RO00"], "SE_1": ["SE01"], "SE_2": ["SE02"], "SE_3": ["SE03"],
    "SE_4": ["SE04"], "SI": ["SI00"], "SK": ["SK00"], "RS": ["RS00"],
}

WIND_CARRIERS = ("onwind", "offwind-ac", "offwind-dc", "offwind-float")
SOLAR_CARRIERS = ("solar", "solar-hsat")

WIND_COLS = ("Wind Onshore", "Wind Offshore")
SOLAR_COLS = ("Solar",)

# Ratio sanity clips: VRE forecast errors beyond x3 are data artefacts
# (realised near zero); load forecasts are accurate to a few percent.
VRE_RATIO_CLIP = (0.0, 3.0)
LOAD_RATIO_CLIP = (0.7, 1.3)


def _hourly_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Resample mixed-resolution ENTSO-E frames to hourly, tz-naive UTC."""
    df = df.resample("1h").mean()
    df.index = df.index.tz_localize(None)
    return df


def build_forecast_ratios(
    n: pypsa.Network,
    forecasts_dir: Path | str,
    generation_dir: Path | str,
) -> dict[str, pd.DataFrame]:
    """Per-zone hourly forecast/realised ratios for load, wind, solar.

    Returns ``{"load": DataFrame, "wind": DataFrame, "solar": DataFrame}``
    indexed by the network snapshots, one column per PyPSA zone. Zones
    without forecast data (GB, western Balkans, ESMA, FR15) get ratio
    one (perfect-information fallback, documented).
    """
    forecasts_dir = Path(forecasts_dir)
    generation_dir = Path(generation_dir)
    sns = n.snapshots

    ratios = {
        k: pd.DataFrame(1.0, index=sns, columns=sorted(n.buses.index))
        for k in ("load", "wind", "solar")
    }

    for ez, zones in ZONE_MAP.items():
        # --- load: D-1 forecast / realised, both from the platform ---
        f_load = forecasts_dir / f"{ez}_load_forecast.csv"
        a_load = forecasts_dir / f"{ez}_load_actual_forecast.csv"
        if f_load.exists() and a_load.exists():
            fc = _hourly_naive_utc(
                pd.read_csv(f_load, index_col=0, parse_dates=True)
            ).iloc[:, 0]
            ac = _hourly_naive_utc(
                pd.read_csv(a_load, index_col=0, parse_dates=True)
            ).iloc[:, 0]
            r = (fc / ac).reindex(sns)
            r = r.where(ac.reindex(sns) > 0).clip(*LOAD_RATIO_CLIP).fillna(1.0)
            for z in zones:
                ratios["load"][z] = r

        # --- VRE: D-1 forecast / realised generation ---
        f_vre = forecasts_dir / f"{ez}_vre_forecast.csv"
        a_gen = generation_dir / f"{ez}.csv"
        if not (f_vre.exists() and a_gen.exists()):
            continue
        fc = _hourly_naive_utc(pd.read_csv(f_vre, index_col=0, parse_dates=True))
        ac = _hourly_naive_utc(pd.read_csv(a_gen, index_col=0, parse_dates=True))
        for group, cols in (("wind", WIND_COLS), ("solar", SOLAR_COLS)):
            f_cols = [c for c in cols if c in fc.columns]
            a_cols = [c for c in cols if c in ac.columns]
            if not f_cols or not a_cols:
                continue
            f_sum = fc[f_cols].sum(axis=1).reindex(sns)
            a_sum = ac[a_cols].sum(axis=1).reindex(sns)
            # Threshold: below 2 % of the zone's annual peak the ratio is
            # numerically meaningless (night solar, calm wind).
            thresh = 0.02 * max(a_sum.max(), 1.0)
            r = (f_sum / a_sum).where(a_sum > thresh).clip(*VRE_RATIO_CLIP)
            r = r.fillna(1.0)
            for z in zones:
                ratios[group][z] = r

    return ratios


def solve_rolling_horizon(
    n: pypsa.Network,
    *,
    horizon: int = 24,
    ratios: dict[str, pd.DataFrame] | None = None,
    soc_target: pd.DataFrame | None = None,
    extra_functionality=None,
    solver_name: str = "gurobi",
    solver_options: dict | None = None,
    log_every: int = 50,
) -> pd.DataFrame:
    """Solve the full year in sequential windows of ``horizon`` snapshots.

    With ``ratios`` set, each window's wind/solar ``p_max_pu`` and load
    ``p_set`` are scaled by the corresponding forecast/realised ratio
    before the solve (forecast-error step); without, the realised series
    are used (foresight-truncation step). Storage state of charge is
    carried between windows; cyclic constraints are disabled.

    ``soc_target`` (index: snapshots, columns: storage units; typically
    ``n.storage_units_t.state_of_charge`` from the annual
    perfect-foresight solve) pins each window's terminal state of charge
    to the long-term trajectory via ``state_of_charge_set``. Without it,
    a myopic window assigns storage no future value and dumps the
    reservoirs --- the known degenerate behaviour of naive rolling
    horizons. Pinning reproduces the real market structure where
    day-ahead dispatch sits on top of longer-term hydro scheduling.

    Returns a per-window status DataFrame. Duals accumulate in
    ``n.buses_t.marginal_price`` window by window.
    """
    sns = n.snapshots
    windows = [sns[i : i + horizon] for i in range(0, len(sns), horizon)]

    su = n.storage_units
    orig_cyclic = su.cyclic_state_of_charge.copy()
    orig_soc_init = su.state_of_charge_initial.copy()
    orig_soc_set = n.storage_units_t.state_of_charge_set.copy()
    su["cyclic_state_of_charge"] = False
    if soc_target is not None:
        # Pin only the last snapshot of every window.
        window_ends = [w[-1] for w in windows]
        pin = pd.DataFrame(
            np.nan, index=sns, columns=soc_target.columns
        )
        pin.loc[window_ends] = soc_target.loc[window_ends].values
        n.storage_units_t.state_of_charge_set = pin
        # The annual reference solve is cyclic, so its first-window
        # target is only reachable from the year-end state. Start the
        # chain there instead of the static default (which is ~empty
        # for the large reservoirs and makes window 0 infeasible).
        su["state_of_charge_initial"] = soc_target.iloc[-1].reindex(su.index).fillna(
            orig_soc_init
        )

    gens = n.generators
    vre_zone = {}
    if ratios is not None:
        for carriers, group in ((WIND_CARRIERS, "wind"), (SOLAR_CARRIERS, "solar")):
            for g in gens.index[gens.carrier.isin(carriers)]:
                vre_zone[g] = (group, gens.at[g, "bus"])
        orig_p_max_pu = n.generators_t.p_max_pu.copy()
        orig_p_set = n.loads_t.p_set.copy()
        load_zone = {l: n.loads.at[l, "bus"] for l in n.loads.index}

    records = []
    for w_i, w in enumerate(windows):
        if ratios is not None:
            for g, (group, zone) in vre_zone.items():
                if g in orig_p_max_pu.columns and zone in ratios[group].columns:
                    n.generators_t.p_max_pu.loc[w, g] = (
                        orig_p_max_pu.loc[w, g] * ratios[group].loc[w, zone]
                    ).clip(0.0, 1.0)
            for l, zone in load_zone.items():
                if l in orig_p_set.columns and zone in ratios["load"].columns:
                    n.loads_t.p_set.loc[w, l] = (
                        orig_p_set.loc[w, l] * ratios["load"].loc[w, zone]
                    )

        opt_kwargs = dict(
            snapshots=w,
            solver_name=solver_name,
            solver_options=solver_options or {},
        )
        if extra_functionality is not None:
            opt_kwargs["extra_functionality"] = extra_functionality
        status, condition = n.optimize(**opt_kwargs)
        if condition != "optimal":
            # Barrier without crossover often returns "suboptimal" on the
            # small window LPs; dual simplex is exact and fast at this
            # size. Retry once. (For MIP windows this only changes the
            # root-LP method; integrality is preserved.)
            retry = dict(solver_options or {})
            retry.pop("Crossover", None)
            retry["Method"] = 1
            opt_kwargs["solver_options"] = retry
            status, condition = n.optimize(**opt_kwargs)
        records.append(
            {"window": w_i, "start": w[0], "status": status, "condition": condition}
        )
        if condition != "optimal":
            print(f"  window {w_i} ({w[0]}): {status}/{condition}", flush=True)
        elif w_i % log_every == 0:
            print(f"  window {w_i}/{len(windows)} ok", flush=True)

        # Chain storage state into the next window.
        last_soc = n.storage_units_t.state_of_charge.loc[w[-1]]
        n.storage_units["state_of_charge_initial"] = last_soc.reindex(su.index).fillna(
            su.state_of_charge_initial
        )

    # Restore mutated statics and (in forecast mode) the realised inputs.
    su["cyclic_state_of_charge"] = orig_cyclic
    su["state_of_charge_initial"] = orig_soc_init
    n.storage_units_t.state_of_charge_set = orig_soc_set
    if ratios is not None:
        n.generators_t.p_max_pu = orig_p_max_pu
        n.loads_t.p_set = orig_p_set

    return pd.DataFrame(records).set_index("window")
