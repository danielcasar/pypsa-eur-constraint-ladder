# GENeSYS-MOD.jl — Local Setup Notes

## Environment

- **Julia**: 1.12.6 (installed via juliaup, `winget install Julialang.Juliaup`)
- **Repo**: `GENeSYS_MOD.jl/` cloned from https://github.com/GENeSYS-MOD/GENeSYS_MOD.jl (package version 4.0.0)
- **Default solver in tests**: HiGHS (open-source). Replace with Gurobi for full runs.

## Activate the environment

```powershell
cd "GENeSYS_MOD.jl"
julia --project=.
```

Inside the REPL: `using GENeSYSMOD` (note: package is `GENeSYSMOD`, not `GENeSYS_MOD`).

## Adding Gurobi

After installing Gurobi locally and setting `GUROBI_HOME`:

```julia
using Pkg
Pkg.add("Gurobi")
Pkg.build("Gurobi")
```

Then in run scripts, swap the solver argument:

```julia
using Gurobi
solver = Gurobi.Optimizer
```

## Entry point

The model is invoked as:

```julia
model, data = genesysmod(; solver=solver, DNLPsolver=Ipopt.Optimizer, ...)
```

It returns the JuMP `model` directly — duals are accessible via JuMP's `dual(...)` API.

A working minimal example with all required kwargs lives in `GENeSYS_MOD.jl/test/test.jl`.
The smallest fully working invocation uses:
- `model_region = "europe"`
- `data_file = "RegularParameters_testdata"`
- `hourly_data_file = "Timeseries_testdata"`
- inputs in `test/TestData/Inputs/`

To smoke-test the full pipeline:
```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

## Shadow prices — important findings

### Where they come from
- The relevant constraint is **`EB2_EnergyBalanceEachTS`** — energy balance per timestep, indexed by `(year, timeslice, fuel, region)`.
- Defined in `src/genesysmod_equ.jl:376`.
- Annual aggregate is `EB3_EnergyBalanceEachYear` (not what we want for hourly analysis).

### How they are written to disk
Two paths in `src/genesysmod_results_raw.jl`:

1. **`genesysmod_getdualsbyname(model, Switch, extr_str, "EB2_EnergyBalanceEachTS")`** —
   writes raw per-timestep duals to
   `EB2_EnergyBalanceEachTS_<region>_<pathway>_<scenario>_<extr_str>.csv`.
   Called automatically when `switch_processed_results = 1` (see `src/genesysmod_results.jl:21`).

2. **Aggregated `resourcecosts`** — `genesysmod_results.jl:28` averages duals across timesteps
   per `(region, fuel, year)`. **Throws away temporal resolution** — do NOT use this for our analysis.

### Quirk to be aware of (relevant for our paper)

Both dual-writing functions skip zero-valued duals:

```julia
# genesysmod_results_raw.jl:89, :111
if dual(con) != 0
    push!(df, ...)
end
```

This means hours with non-binding energy balance (e.g. surplus / curtailment periods)
are **missing rows**, not zero rows. For a full timeseries you must either:
- post-process: reconstruct the full `(region, fuel, year, timestep)` grid and fill missing with 0, or
- patch this filter out locally (one-line removal in both functions).

### Time index
The `timestep` written is GENeSYS-MOD's `l` (timeslice) label, not a calendar timestamp.
Mapping back to hours uses the timeseries reduction settings (`elmod_daystep`, `elmod_hourstep`).
The default test config uses 80 days × 4-hour blocks → 480 representative timesteps/year.

## Test config (minimal Europe run)

`test/test.jl` already runs a tractable Europe instance with:
- `elmod_daystep = 80`, `elmod_hourstep = 4` (reduced timeseries)
- `switch_intertemporal = 0` (myopic, single-year)
- `switch_ccs = 1`, `switch_ramping = 0`
- HiGHS solver

This is the right starting point — solves in minutes, exposes the full output pipeline.

## Open questions (next session)

- Decide whether to patch the zero-dual filter or post-process.
- Pick scenarios + years for the paper's runs (probably needs editing
  `genesysmod_scenariodata_europe.jl`).

## Gurobi setup (2026-05-06)

- Gurobi Optimizer 13.0.2 installed at `C:\gurobi1302\`.
- License activated to `C:\Users\Daniel\gurobi.lic` (academic, expires 2027-05-06).
  - Note: requires TU Wien VPN/network for the initial `grbgetkey` activation.
  - Beware: `grbgetkey` interactively prompts for the save dir. If you feed it
    an empty stdin from PowerShell, a BOM character ends up as the dir name.
    Either type the path interactively or pre-set `GRB_LICENSE_FILE`.
- `Gurobi.jl` v1.9.2 added to the project. Uses `GUROBI_HOME=C:\gurobi1302\win64`
  (set this env var before launching Julia, or in a startup file).

## Smoke-test result (2026-05-06)

`Pkg.test()` outcome on HiGHS, Julia 1.12.6:

- **Investment Run**: PASS (10m48s) — full Europe test instance, optimal solution.
- **Dispatch Runs** (Simple, OneNodeStorage, TwoNodes): all 3 PASS (11m30s).
- **Fetch Input Data**: FAIL — but this is a separate `fetch_data_release` /
  `update_and_process_data` utility that pulls scenario templates from a remote
  server. Not on the path for our analysis. Safe to ignore.

Verified outputs landed in `test/TestData/Results/`:
- `EB2_EnergyBalanceEachTS_europe_MinimalExample_globalLimit_inv_run.csv` — the
  per-timestep shadow prices we want. Format: `name,value` with name parseable
  as `EB2_EnergyBalanceEachTS|<year>|<timeslice>|<fuel>|<region>`.
- `Selected_Duals_*.csv` — same data via the alternate writer.
- 80+ CSVs of primal results (production, capacity, trade, emissions, etc.).

Sample row from the dual file:
`EB2_EnergyBalanceEachTS|2018|1924|Biofuel|DE,4964.81`

## Gurobi smoke-test result (2026-05-06)

Same Europe test instance, run via `run_gurobi_smoke.jl`:

- **Status**: OPTIMAL
- **Elapsed**: 283.2s (~4m43s) — vs. ~10m48s on HiGHS (≈2.3× speedup on this small instance; gap will grow on full-resolution runs).
- Outputs in `GENeSYS_MOD.jl/GurobiSmokeResults/`, including
  `EB2_EnergyBalanceEachTS_*.csv` and `Selected_Duals_*.csv`.

To run from a fresh PowerShell session:

```powershell
$env:Path += ";$env:USERPROFILE\.julia\juliaup\bin"
$env:GUROBI_HOME = "C:\gurobi1302\win64"
julia --project="GENeSYS_MOD.jl" "GENeSYS_MOD.jl\run_gurobi_smoke.jl"
```

Ready for scenario customization.
