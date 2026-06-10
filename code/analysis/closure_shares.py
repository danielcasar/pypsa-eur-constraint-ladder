"""Closure-share computation per configuration per zone per calendar dimension.

Pipeline:
  1. Load per-configuration shadow-price CSVs produced by ``driver/run_ladder.py``
     (one CSV per configuration; rows = snapshots, columns = bus names).
  2. For each AC bus (i.e. modelled bidding zone), compare the modelled
     shadow-price time series against the matched ENTSO-E day-ahead
     time series for the same zone and same calendar period.
  3. Compute the compression factor kappa_d = sigma_d^model / sigma_d^obs per
     calendar dimension d in {hour, weekday, month, season} via the
     Paper-1 ``deviation_range`` routine.
  4. Compute the closure share

        Delta kappa_d^(k) = (|kappa_d^(k-1) - 1| - |kappa_d^(k) - 1|) /
                            |kappa_d^(0) - 1|

     per configuration k, per zone, per dimension, per the §2.3.2 definition.

Output: a tidy DataFrame keyed by (configuration, zone, dimension)
with columns kappa_d, delta_kappa_d, n_snapshots_used, etc.

The module is designed to be importable both as
``code.analysis.closure_shares`` (when the ``code`` directory is on
``sys.path``) and from the top-level package once ``code.analysis``
is exposed.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Mirror of code.driver.LADDER_CONFIGURATIONS to keep this module
# importable without triggering the driver's solver imports.
LADDER_CONFIGURATIONS: tuple[str, ...] = (
    "baseline",
    "voll",
    "elastic_demand",
    "reserves",
    "unit_commitment",
    "subsidy_floor_de",
)


PAPER1_ROOT_DEFAULT = Path(
    "C:/Users/Daniel/OneDrive - TU Wien/Publications/"
    "Can shadow prices from energy system optimization models "
    "serve as reliable electricity price indicators"
)


def _import_function_via_ast(filepath: Path, function_name: str):
    """Extract one function definition from a Python source file.

    Lifted from ``Analysis/apply_paper_framework.py`` so we don't
    depend on that script's process-time configuration.
    """
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(filepath))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"np": np, "pd": pd}
            exec(compile(mod, str(filepath), "exec"), ns)  # noqa: S102
            return ns[function_name]
    raise ImportError(f"{function_name} not defined in {filepath}")


def _load_paper1_functions(paper1_root: Path | None = None) -> dict:
    """Return a dict with the Paper-1 callables we need.

    Keys: ``deviation_range``, ``_add_time_dims``, ``TIME_DIMENSIONS``.
    """
    root = paper1_root or PAPER1_ROOT_DEFAULT
    scripts = root / "Code" / "paper" / "scripts"
    if not scripts.exists():
        raise FileNotFoundError(f"Paper 1 scripts not found at {scripts}")

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    # Import the calendar-dimension utilities directly (their module
    # is import-safe).
    from stage1_kruskal_wallis import _add_time_dims, TIME_DIMENSIONS  # type: ignore

    # Import deviation_range via AST (module has unconditional file I/O).
    deviation_range = _import_function_via_ast(
        scripts / "compute_compression_factors.py", "deviation_range"
    )
    return {
        "deviation_range": deviation_range,
        "_add_time_dims": _add_time_dims,
        "TIME_DIMENSIONS": TIME_DIMENSIONS,
    }


def load_per_config_shadow_prices(
    results_dir: Path | str,
    configurations: tuple[str, ...] = LADDER_CONFIGURATIONS,
) -> dict[str, pd.DataFrame]:
    """Load each configuration's shadow_price.csv into a DataFrame.

    Configurations whose CSV is missing are silently skipped (useful
    when only a subset of the ladder has been run).
    """
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for cfg in configurations:
        csv = results_dir / cfg / "shadow_price.csv"
        if not csv.exists():
            logger.info("Skipping %s: %s not found", cfg, csv)
            continue
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        out[cfg] = df
    return out


def _ac_bus_columns(df: pd.DataFrame) -> list[str]:
    """Return AC-bus column names (excluding H2, battery, etc. suffixes).

    Matches the de-1week test network convention where non-AC buses
    have suffix names like " H2", " battery".
    """
    return [
        c for c in df.columns
        if not (" H2" in c or " battery" in c or " load" in c)
    ]


def compute_kappa_per_config(
    per_config_shadow: dict[str, pd.DataFrame],
    entsoe_series: pd.Series,
    paper1_fns: dict | None = None,
) -> pd.DataFrame:
    """Compute the compression factor kappa_d per configuration per dimension.

    Parameters
    ----------
    per_config_shadow
        Output of :func:`load_per_config_shadow_prices`. Each DataFrame
        is treated as a per-bus shadow-price time series; the mean
        across all AC buses is used as the per-configuration price.
    entsoe_series
        ENTSO-E day-ahead price for the matched zone and time window,
        as a Series indexed by hourly timestamps.
    paper1_fns
        Dict from :func:`_load_paper1_functions`. Loaded on demand if
        not supplied.

    Returns
    -------
    DataFrame with columns ``configuration``, ``dimension``,
    ``kappa_d``, ``n_snapshots``.
    """
    if paper1_fns is None:
        paper1_fns = _load_paper1_functions()
    add_dims = paper1_fns["_add_time_dims"]
    deviation_range = paper1_fns["deviation_range"]
    time_dims: tuple[str, ...] = paper1_fns["TIME_DIMENSIONS"]

    # ENTSO-E series may span many calendar years; the smoke / continental
    # comparison windows are decided per configuration so we keep it
    # indexed by timestamp here and window before calling ``add_dims``
    # (which drops the timestamp column itself). Drop timezone so the
    # ENTSO-E series and the (tz-naive) shadow-price index compare cleanly.
    entsoe_indexed = entsoe_series.dropna()
    if entsoe_indexed.index.tz is not None:
        entsoe_indexed = entsoe_indexed.tz_convert("UTC").tz_localize(None)

    rows: list[dict] = []
    for cfg, df in per_config_shadow.items():
        ac = _ac_bus_columns(df)
        if not ac:
            logger.warning("Configuration %s has no AC buses; skipping", cfg)
            continue
        # Mean across modelled buses to obtain a single zone price.
        model_series = df[ac].mean(axis=1).dropna()
        if model_series.empty:
            continue

        # Window ENTSO-E to the model time range BEFORE losing the timestamp.
        window_mask = (entsoe_indexed.index >= model_series.index.min()) & (
            entsoe_indexed.index <= model_series.index.max()
        )
        entsoe_window = entsoe_indexed[window_mask]
        if entsoe_window.empty:
            logger.warning(
                "No overlapping ENTSO-E hours for configuration %s window [%s, %s]; "
                "comparing against the full ENTSO-E series instead.",
                cfg, model_series.index.min(), model_series.index.max(),
            )
            entsoe_window = entsoe_indexed

        model_df = add_dims(
            pd.DataFrame({"price": model_series.values}, index=model_series.index)
        )
        obs_df = add_dims(
            pd.DataFrame({"price": entsoe_window.values}, index=entsoe_window.index)
        )

        for dim in time_dims:
            if obs_df[dim].nunique() < 2 or model_df[dim].nunique() < 2:
                continue
            obs_range = deviation_range(
                obs_df["price"].values, obs_df[dim].values
            )
            mod_range = deviation_range(
                model_df["price"].values, model_df[dim].values
            )
            kappa = mod_range / obs_range if obs_range > 0 else np.nan
            rows.append(
                {
                    "configuration": cfg,
                    "dimension": dim,
                    "kappa_d": kappa,
                    "obs_range_eur_per_mwh": obs_range,
                    "model_range_eur_per_mwh": mod_range,
                    "n_model_snapshots": int(len(model_df)),
                    "n_obs_snapshots": int(len(obs_df)),
                }
            )
    return pd.DataFrame(rows)


def closure_shares_per_zone(
    per_config_shadow: dict[str, pd.DataFrame],
    entsoe_by_zone: dict[str, pd.Series],
    bus_to_zone: dict[str, str] | None = None,
    paper1_fns: dict | None = None,
) -> pd.DataFrame:
    """Per-zone, per-configuration kappa_d and Delta kappa_d^(k).

    For the continental run, ``bus_to_zone`` maps each AC bus name to
    its ENTSO-E bidding-zone code; the comparison is done independently
    per zone. For a single-zone (e.g. DE-cluster) network, supply a
    mapping that points every bus to the same zone.

    Returns
    -------
    DataFrame keyed by ``(configuration, zone, dimension)`` with
    columns ``kappa_d``, ``delta_kappa_d`` and the supporting counts.
    """
    if paper1_fns is None:
        paper1_fns = _load_paper1_functions()
    add_dims = paper1_fns["_add_time_dims"]
    deviation_range = paper1_fns["deviation_range"]
    time_dims: tuple[str, ...] = paper1_fns["TIME_DIMENSIONS"]

    if bus_to_zone is None:
        # Default: every AC bus name maps to its first 2 characters as
        # the bidding-zone code (matches PyPSA-Eur's DE0 0 -> DE).
        sample_df = next(iter(per_config_shadow.values()))
        bus_to_zone = {b: b[:2] for b in _ac_bus_columns(sample_df)}

    rows: list[dict] = []
    for zone in sorted(set(bus_to_zone.values())):
        zone_buses = [b for b, z in bus_to_zone.items() if z == zone]
        if zone not in entsoe_by_zone:
            logger.info("No ENTSO-E series for zone %s; skipping", zone)
            continue

        # Build a per-config zone price as the mean of zone buses.
        per_config_zone: dict[str, pd.DataFrame] = {}
        for cfg, df in per_config_shadow.items():
            cols = [c for c in zone_buses if c in df.columns]
            if not cols:
                continue
            series = df[cols].mean(axis=1)
            per_config_zone[cfg] = pd.DataFrame(
                {"timestamp": series.index, "price": series.values}
            )

        # Use compute_kappa_per_config repurposed: feed it a single
        # "configuration" at a time so each row keeps its zone tag.
        for cfg, df in per_config_zone.items():
            zone_kappa = compute_kappa_per_config(
                {cfg: df.set_index("timestamp")[["price"]].rename(
                    columns={"price": zone_buses[0]}
                )},
                entsoe_by_zone[zone],
                paper1_fns=paper1_fns,
            )
            for _, r in zone_kappa.iterrows():
                rows.append({"zone": zone, **r.to_dict()})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute Delta kappa per (zone, dimension) by ordering configurations
    # as in LADDER_CONFIGURATIONS and applying the §2.3.2 formula.
    order = {c: i for i, c in enumerate(LADDER_CONFIGURATIONS)}
    df["__order"] = df["configuration"].map(order)
    df = df.sort_values(["zone", "dimension", "__order"]).reset_index(drop=True)

    df["delta_kappa_d"] = np.nan
    for (zone, dim), grp in df.groupby(["zone", "dimension"]):
        grp_sorted = grp.sort_values("__order")
        kappa = grp_sorted["kappa_d"].to_numpy()
        if len(kappa) == 0 or np.isnan(kappa[0]):
            continue
        baseline_gap = abs(kappa[0] - 1.0)
        if baseline_gap <= 0:
            continue
        # Δκ_d^(k) per the §2.3.2 definition
        deltas = np.full_like(kappa, np.nan, dtype=float)
        for i in range(1, len(kappa)):
            if np.isnan(kappa[i]) or np.isnan(kappa[i - 1]):
                continue
            deltas[i] = (abs(kappa[i - 1] - 1.0) - abs(kappa[i] - 1.0)) / baseline_gap
        df.loc[grp_sorted.index, "delta_kappa_d"] = deltas

    return df.drop(columns="__order")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-configuration subfolders with shadow_price.csv",
    )
    p.add_argument(
        "--entsoe-csv",
        required=True,
        help="ENTSO-E day-ahead price CSV (columns = zone codes, rows = hourly timestamps)",
    )
    p.add_argument(
        "--zone",
        default="DE",
        help="Single zone code to compare against (default DE). Use --all-zones for the per-zone matrix.",
    )
    p.add_argument(
        "--all-zones",
        action="store_true",
        help="Run per-zone closure shares using bus[:2] -> zone mapping",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV for the closure-share table (default: results-dir/closure_shares.csv)",
    )
    p.add_argument(
        "--paper1-root",
        default=None,
        help="Override Paper 1 root path (default: hardcoded TU Wien path)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    paper1_root = Path(args.paper1_root) if args.paper1_root else None
    paper1_fns = _load_paper1_functions(paper1_root)

    per_config = load_per_config_shadow_prices(args.results_dir)
    if not per_config:
        raise SystemExit(f"No per-configuration shadow prices found under {args.results_dir}")

    entsoe = pd.read_csv(args.entsoe_csv, index_col=0, parse_dates=True)

    if args.all_zones:
        entsoe_by_zone = {z: entsoe[z].dropna() for z in entsoe.columns}
        out = closure_shares_per_zone(per_config, entsoe_by_zone, paper1_fns=paper1_fns)
    else:
        if args.zone not in entsoe.columns:
            raise SystemExit(f"Zone {args.zone!r} not in ENTSO-E columns {list(entsoe.columns)}")
        out = compute_kappa_per_config(
            per_config, entsoe[args.zone].dropna(), paper1_fns=paper1_fns
        )
        # Add Delta kappa per dimension for single-zone case.
        order = {c: i for i, c in enumerate(LADDER_CONFIGURATIONS)}
        out["__order"] = out["configuration"].map(order)
        out = out.sort_values(["dimension", "__order"]).reset_index(drop=True)
        out["delta_kappa_d"] = np.nan
        for dim, grp in out.groupby("dimension"):
            grp_sorted = grp.sort_values("__order")
            kappa = grp_sorted["kappa_d"].to_numpy()
            if len(kappa) == 0 or np.isnan(kappa[0]):
                continue
            baseline_gap = abs(kappa[0] - 1.0)
            if baseline_gap <= 0:
                continue
            deltas = np.full_like(kappa, np.nan, dtype=float)
            for i in range(1, len(kappa)):
                if np.isnan(kappa[i]) or np.isnan(kappa[i - 1]):
                    continue
                deltas[i] = (
                    abs(kappa[i - 1] - 1.0) - abs(kappa[i] - 1.0)
                ) / baseline_gap
            out.loc[grp_sorted.index, "delta_kappa_d"] = deltas
        out = out.drop(columns="__order")

    output = Path(args.output) if args.output else Path(args.results_dir) / "closure_shares.csv"
    out.to_csv(output, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
