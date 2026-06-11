"""Capture per-configuration solver metrics.

Each ladder run writes a ``solve_metrics.json`` next to its
``shadow_price.csv`` describing the LP / QP / MIP that was solved, so
the paper's compute / scientific-value trade-off section can be
populated mechanically rather than re-typed by hand.

Fields written
--------------
configuration         : str    user-supplied identifier (e.g. 'baseline')
problem_class         : str    'LP' | 'QP' | 'MIP'
status                : str    PyPSA optimize status ('ok', 'warning', ...)
condition             : str    solver termination condition ('optimal', ...)
solver                : str    'gurobi' | 'highs' | ...
solver_options        : dict   options passed to ``n.optimize``
wallclock_s           : float  total Python-side wallclock including model build + solve
gurobi_runtime_s      : float  Gurobi's own RunTime attribute, if available
objective             : float  optimal objective value (None if not solved)
n_snapshots           : int
n_buses               : int
n_generators          : int
num_variables         : int    LP/MIP variable count after presolve
num_constraints       : int    LP/MIP constraint count after presolve
num_nonzeros          : int    matrix non-zero count after presolve
mip_gap               : float  final relative MIP gap, only for MIP runs
peak_memory_mb        : int    resident-set peak (Linux only, via /proc/self/status)
solver_version        : str
network_path          : str    absolute path of input .nc
shadow_price_csv      : str    output csv path
timestamp_utc         : str    ISO-8601 wall-clock at write

Use
---
    from time import perf_counter
    from constraint_ladder.helpers.metrics import record_solve_metrics

    t0 = perf_counter()
    n = pypsa.Network(NETWORK_PATH)
    ...
    status, condition = n.optimize(solver_name='gurobi', solver_options=opts)
    wallclock = perf_counter() - t0

    record_solve_metrics(
        n,
        output_dir=RESULTS_DIR,
        configuration='baseline',
        problem_class='LP',
        solver='gurobi',
        solver_options=opts,
        status=status, condition=condition,
        wallclock_s=wallclock,
        network_path=NETWORK_PATH,
    )
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pypsa


def _peak_memory_mb() -> int | None:
    """Resident-set peak for the current process (Linux only)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except OSError:
        return None
    return None


def _gurobi_problem_size(n: pypsa.Network) -> tuple[int | None, int | None, int | None, float | None]:
    """Return (n_vars, n_constr, n_nonzeros, gurobi_runtime_s) if available."""
    try:
        model = n.model.solver_model  # linopy backend handle to gurobipy.Model
    except (AttributeError, RuntimeError):
        return None, None, None, None
    try:
        n_vars = int(model.NumVars)
        n_cons = int(model.NumConstrs)
        n_nzs = int(model.NumNZs)
        runtime = float(getattr(model, "Runtime", 0.0)) or None
        return n_vars, n_cons, n_nzs, runtime
    except (AttributeError, Exception):  # noqa: BLE001
        return None, None, None, None


def _mip_gap(n: pypsa.Network) -> float | None:
    try:
        model = n.model.solver_model
        return float(getattr(model, "MIPGap", float("nan")))
    except (AttributeError, RuntimeError, ValueError):
        return None


def _solver_version(solver: str) -> str | None:
    if solver == "gurobi":
        try:
            import gurobipy

            ver = gurobipy.gurobi.version()
            return ".".join(str(x) for x in ver)
        except Exception:  # noqa: BLE001
            return None
    if solver == "highs":
        try:
            import highspy

            return getattr(highspy, "__version__", None)
        except Exception:  # noqa: BLE001
            return None
    return None


def record_solve_metrics(
    n: pypsa.Network,
    *,
    output_dir: Path,
    configuration: str,
    problem_class: str,
    solver: str,
    solver_options: Mapping[str, Any] | None,
    status: str,
    condition: str,
    wallclock_s: float,
    network_path: Path | str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the metrics dict, write ``solve_metrics.json`` to ``output_dir``,
    and return the dict so the caller can also use it inline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_vars, n_cons, n_nzs, gurobi_runtime = _gurobi_problem_size(n)

    metrics: dict[str, Any] = {
        "configuration": configuration,
        "problem_class": problem_class,
        "status": str(status),
        "condition": str(condition),
        "solver": solver,
        "solver_options": dict(solver_options or {}),
        "solver_version": _solver_version(solver),
        "wallclock_s": round(float(wallclock_s), 3),
        "gurobi_runtime_s": round(gurobi_runtime, 3) if gurobi_runtime else None,
        "objective": float(n.objective) if status == "ok" else None,
        "n_snapshots": int(len(n.snapshots)),
        "n_buses": int(len(n.buses)),
        "n_generators": int(len(n.generators)),
        "num_variables": n_vars,
        "num_constraints": n_cons,
        "num_nonzeros": n_nzs,
        "mip_gap": _mip_gap(n) if problem_class == "MIP" else None,
        "peak_memory_mb": _peak_memory_mb(),
        "network_path": str(network_path),
        "shadow_price_csv": str(output_dir / "shadow_price.csv"),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "python_version": platform.python_version(),
    }
    if extra:
        metrics.update(dict(extra))

    (output_dir / "solve_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    return metrics
