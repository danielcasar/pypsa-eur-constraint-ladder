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
  clusters: [adm]
  opts: ['']
  sector_opts: ['']
  planning_horizons: [2025]
clustering:
  mode: administrative
  administrative:
    level: bz
```
Single-period overnight optimisation. The `clusters: [adm]` wildcard
together with `clustering.mode: administrative` + `level: bz` drives
PyPSA-Eur's bidding-zone-aware clustering: one cluster per real ENTSO-E
bidding zone (Italy gets NORD/CSUD/SUD/SARD/SICI/CNORD; Sweden
SE1-SE4; Denmark DK1/DK2; etc.). Bus shapes come from
`build_bidding_zones.py` which fuses electricitymaps-contrib and
entsoe-py shape sources and adjusts to TYNDP 2024. `planning_horizons:
[2025]` is decoupled from the snapshot calendar (which is 2024); it
drives the cost-data file retrieval (`costs_2025.csv`) because
PyPSA-Eur publishes cost CSVs only at quinquennial horizons.

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
  estimate_renewable_capacities:
    from_powerplantmatching: false
    from_irenastat: true
    year: 2024

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

The renewable-capacity source is switched from
`from_powerplantmatching` (default; unit-level OPSD/GEM/JRC data that
lags actual deployment) to `from_irenastat` (IRENA country-aggregate
statistics, currently with 2024 totals). The default left
Germany/Italy/Netherlands/Belgium solar 50-90% under-counted because
those countries' rapid 2024 solar additions had not yet propagated
through the unit-level data sources at PyPSA-Eur v2026.02.0 tag time.
IRENA's aggregate numbers reflect the actual installed base.

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

**Full comparison: upstream values vs our overrides** (all fuel in EUR/MWh
thermal; verified against the raw archive file
`pypsa-eur/data/costs/archive/v0.14.0/costs_2025.csv` and the processed
output `resources/eu_2024_dispatch/costs_2025_processed.csv` on 2026-06-12):

| Carrier | Upstream | Upstream basis (money year) | Override | 2024-observed basis | Sources to double-check |
|---|---|---|---|---|---|
| gas | 42.90 | World Bank Commodity Markets + TYNDP 2024 scenario inputs (2020 EUR) | **32** | TTF day-ahead 2024 calendar avg ≈ €34 (Q1 ~27, Q4 ~45) | ICE Endex TTF; EEX spot indices; ACER wholesale market reports |
| coal | 7.82 | TYNDP 2024 scenario inputs, IEA 2022 APS (2021 EUR) | **11** | API2 CIF ARA 2024 avg ≈ $105–115/t → ≈ €13–15/MWh_th at 6.98 MWh/t, 0.92 EUR/USD | Argus/McCloskey API2; EEX coal futures; Trading Economics |
| lignite | 7.94 | TYNDP 2024, Booz&co Lignite G2 (2021 EUR) | **4** | Mine-mouth convention, €3–5 typical | EURACOAL Coal in Europe; Öko-Institut/Agora lignite studies |
| oil | 43.00 | World Bank + TYNDP, crude +5 % heavy-oil markup (2020 EUR) | **50** | Brent 2024 avg ~$80.5/bbl → ≈ €44–46/MWh incl. markup | EIA Brent series; OPEC MOMR |
| nuclear (fuel) | 7.45 | TYNDP 2024, EIA 2022 (2021 EUR) | **2.6** | Front-end fuel cycle at ESA 2024 multiannual contract price (142.26 EUR/kgU) + UxC conversion (~$55/kgU) / enrichment (~$160/SWU) / fabrication (~$350/kg); WNA arithmetic, 45 GWd/tU burnup | Euratom Supply Agency Market Observatory (official EU utility prices); UxC; WNA Economics of Nuclear Power |
| biomass | 9.35 | IEA 2011 via old PyPSA assumptions (2015 EUR) | **25** | NWE industrial wood-pellet wholesale 2024 ≈ €120–140/t ÷ 4.8 MWh/t | Bioenergy Europe Statistical Report; EUROSTAT |
| CO₂ | (off by default) | — | **65 EUR/tCO₂** | EEX EUA front-year 2024 calendar avg ≈ €65 | EEX EUA auctions; Ember carbon price tracker |

Key reading of the comparison: the upstream values are **TYNDP scenario
conventions in 2020/2021 money**, not observed prices. Upstream gas
(42.90) is ~25 % above the 2024 TTF average; upstream coal (7.82) is
roughly **half** the API2-implied 2024 value; upstream biomass (9.35) is
a 2011-vintage number. Our overrides move every carrier to the observed
2024 regime. Two of our values are themselves imperfect: coal 11 sits
slightly below the API2-implied €13–15 band, and oil 50 sits ~10 % above
the Brent-implied €44–46 (oil is almost never price-setting in the
modelled zones, so the effect is negligible).

**Resulting effective marginal costs on the prepared network** (EUR/MWh
electric, incl. VOM, efficiency, CO₂ at €65/t): CCGT 61.9, OCGT 85.0,
hard coal 35.0, lignite 16.2, nuclear 12.4, biomass 53.4, oil 150.9.

OCGT and CCGT fuel cost inherits from `gas` automatically in
`scripts/process_cost_data.py` (line ~178), so overriding `gas` propagates
to both gas-turbine technologies.

**⚠ Uranium-row override does not propagate — target `nuclear` directly.**
The `nuclear` technology row in `costs_2025.csv` carries its **own**
`fuel` attribute (7.4536 EUR/MWh_th, copied from the uranium row
upstream in technology-data, before any runtime overwrite applies). An
`overwrites.fuel.uranium` entry only modifies the standalone `uranium`
row, which no electricity-only generator reads — a silent no-op. We
therefore overwrite `nuclear: 2.6` directly (the `uranium: 2.6` entry is
kept purely for internal consistency). Resulting nuclear marginal cost:
4.46 (VOM) + 2.6/0.326 (fuel/η) = **12.4 EUR/MWh**, consistent with
published estimates of French nuclear bid levels (~10–20 EUR/MWh).
History: an earlier `uranium: 1.7` was silently ineffective, leaving
nuclear at the TYNDP-based 27.3 EUR/MWh through the first baseline runs;
fixed 2026-06-12 with the ESA-based value.

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
