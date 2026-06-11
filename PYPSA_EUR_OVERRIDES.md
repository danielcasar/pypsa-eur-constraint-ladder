# Deviations from vanilla PyPSA-Eur v2026.02.0

This document lists every modification we apply on top of the upstream
`PyPSA/pypsa-eur` repository at tag `v2026.02.0` (the pin recorded in
`SETUP_NOTES.md`). Each entry says **what** changed, **why** it changed,
**where** the change lives, and **how** it is invoked.

Vanilla PyPSA-Eur sources themselves are **never** edited. Every
deviation is either a config override consumed via Snakemake's
`--configfile`, or a one-off data transformation applied via a Python
script in `tools/`. The upstream clone at `pypsa-eur/` is kept on the
`v2026.02.0` tag verbatim.

---

## 1. Configuration overrides

All overrides live in `pypsa_setup/configs/eu_2024_dispatch.yaml`. Snakemake reads
this file on top of `pypsa-eur/config/config.default.yaml` so only the
listed keys deviate from upstream.

### 1.1 Run identifier
```yaml
run:
  name: "eu_2024_dispatch"
  shared_resources: { policy: false }
```
Isolates this paper's prepared resources under
`resources/eu_2024_dispatch/` so a vanilla run sharing the same PyPSA-Eur
clone is unaffected.

### 1.2 Scenario / horizon
```yaml
foresight: overnight
scenario:
  clusters: [50]
  opts: ['']
  sector_opts: ['']
  planning_horizons: [2025]
```
Single-period overnight optimisation. 50 zones is the continental
discretisation target. `planning_horizons: [2025]` is decoupled from the
snapshot calendar (which is 2024); it drives the cost-data file
retrieval (`costs_2025.csv`) because PyPSA-Eur publishes cost CSVs only
at quinquennial horizons.

### 1.3 Calendar
```yaml
snapshots:
  start: "2024-01-01"
  end: "2025-01-01"
  inclusive: "left"
```
Full 2024, hourly. 2024 is a leap year, so the network carries
8784 snapshots. **Why 2024 and not 2025**: PyPSA-Eur v2026.02.0 was
tagged before 2025 historical data was assembled; multiple year-keyed
upstream files (synthetic load, nuclear availability) stop at 2024 and
would need hand-patching for a 2025 calendar. 2024 is the latest year
the upstream supports without patching.

### 1.4 Dispatch-only switch
```yaml
electricity:
  extendable_carriers:
    Generator: []
    StorageUnit: []
    Store: []
    Link: []

sector:
  extendable_carriers:
    Generator: []
    StorageUnit: []
    Store: []
    Link: []
```
All extendable carrier lists empty, so neither the electricity LP nor
the sector LP introduces capacity-expansion variables. The model is
pure dispatch on the 2024-pinned fleet.

### 1.5 Weather cutout
```yaml
atlite:
  default_cutout: "europe-2024-sarah3-era5"
```
Replaces the upstream default (`europe-2013-sarah3-era5`) with the 2024
SARAH3+ERA5 hybrid cutout, pre-published at data.pypsa.org. Renewable +
hydro profiles are then computed on 2024 meteorology aligned with the
2024 snapshot calendar.

### 1.6 Cost data — base file + 2024-actual overrides
```yaml
costs:
  year: 2025
  overwrites:
    fuel:
      gas: 32
      coal: 11
      lignite: 4
      oil: 50
      uranium: 1.7
      biomass: 25
  emission_prices:
    enable: true
    co2: 65
```

**Base file: `costs_2025.csv`.** PyPSA-Eur publishes cost CSVs only at
quinquennial horizons (2020, 2025, 2030, ..., 2050). `costs_2024.csv`
does not exist (verified HTTP 404). 2025 is the nearest published file
and is structurally close to actual 2024 prices.

**Per-attribute fuel-cost overrides** (all in EUR/MWh thermal):

| Carrier | Override | Source |
|---|---|---|
| gas | 32 | TTF day-ahead 2024 calendar-year average (peaks Q1 ~55, lows Q3 ~25) |
| coal | 11 | CIF NWE API2 2024 average, converted to MWh-thermal |
| lignite | 4 | Mine-mouth, stable across years |
| oil | 50 | Brent 2024 average ~$80/bbl in MWh-thermal equivalent |
| uranium | 1.7 | Nuclear fuel cycle cost, stable |
| biomass | 25 | Northwest Europe wholesale 2024 |

OCGT and CCGT fuel cost inherits from `gas` automatically in
`scripts/process_cost_data.py` (line ~178), so overriding `gas` propagates
to both gas-turbine technologies.

**CO₂ price override**: `emission_prices.enable: true, co2: 65` — EUA
front-year contract 2024 calendar-year average ~EUR 65/tCO2.

**Deprecation note**: PyPSA-Eur emits a `DeprecationWarning` recommending
overrides via an external `data/custom_costs.csv`. For a single-paper run
the YAML form is fine; we accept the warning.

### 1.7 Solver
```yaml
solving:
  solver:
    name: gurobi
    options: gurobi-default
  solver_options:
    gurobi-default:
      threads: 20
      method: 2
      crossover: 0
      BarConvTol: 1.e-6
      FeasibilityTol: 1.e-5
      OptimalityTol: 1.e-5
      Seed: 123
```
Barrier without crossover, 20 threads (Marvin has 24 physical cores;
leaves ~4 for linopy / I/O).

---

## 2. Data extensions (one-off scripts in `tools/`)

### 2.1 Synthetic load extension
**Script**: `pypsa_setup/data_patches/extend_synthetic_load.py`.
**Touches**: `pypsa-eur/data/synthetic_electricity_demand/primary/v2/load_synthetic_raw.csv`.
**Operation**: Appends shape-shifted copies of the upstream synthetic
profile to extend coverage from native end-of-2023 to end-of-2025:
- 2020 (leap) → 2024 (leap)
- 2023 (non-leap) → 2025 (non-leap)
**Run-time**: idempotent. A second run is a no-op once the target rows
are present.
**Why needed**: PyPSA-Eur's `build_electricity_demand` rule selects the
full snapshot window from the synthetic file as a gap-fill source even
when real ENTSO-E / OPSD covers the year. Without extension, any
snapshot range past 2023-12-31 raises a `KeyError` on time-index
selection.
**Coverage effect**: only countries whose real series are absent for
2024 — i.e., the small markets (BA, ME, XK, MD, UA) — see the
shape-shifted data; major markets read from real series and ignore the
synthetic fallback.

---

## 3. Known data gaps accepted as-is

These are upstream limitations we do **not** patch around (because the
data is consumed correctly by PyPSA-Eur's fallback logic, or because
the impact is negligible for our use case).

| File | Native upper bound | Effect on our 2024 run |
|---|---|---|
| `data/nuclear_p_max_pu.csv` | 2024 (last column) | OK — script reads 2024 column directly |
| `data/cost/.../costs_*.csv` | Quinquennial only | Handled via `costs.year: 2025` + per-attr overrides above |
| Other year-keyed CSVs | Various | Read with script-side `df.iloc[:, -1]` fallback where applicable |

---

## 4. Verification + reproducibility

Vanilla PyPSA-Eur at `v2026.02.0` is reproduced from a clean clone:
```bash
git clone https://github.com/PyPSA/pypsa-eur.git
cd pypsa-eur && git checkout v2026.02.0
```

Our deviations on top are reproduced by:
1. Cloning this repo alongside (`git clone <this-repo> ladder`).
2. Running the one-off data extension once
   (`python ../ladder/pypsa_setup/data_patches/extend_synthetic_load.py`).
3. Pointing Snakemake at the override config
   (`--configfile ../ladder/pypsa_setup/configs/eu_2024_dispatch.yaml`).

No edits are made inside `pypsa-eur/` itself; the clone stays on the
pinned tag throughout.
