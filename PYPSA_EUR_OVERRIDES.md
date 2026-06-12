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

```yaml
electricity:
  transmission_limit: v1.0
```
**Transmission grid pinned (added 2026-06-12).** The upstream default
`transmission_limit: vopt` marks **every AC line and DC link
extendable** (`set_transmission_limit` in `prepare_network.py` flips
`s_nom_extendable` / `p_nom_extendable` for factor `opt` or > 1.0), so
a nominally dispatch-only run silently co-optimises transmission
expansion (bounded at +20 GW/line and +30 GW/link), relaxing congestion
and flattening inter-zonal price spreads. `v1.0` keeps the grid at
today's volume with no expansion variables. Discovered in the
pre-ladder audit: all baseline solves before this date carried the
artefact and were re-run. Belt-and-braces: `reset_to_lp_baseline()` in
`constraint_ladder/helpers/native_constraints.py` also force-clears all
extendable flags at solve time.

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
      gas: 34
      coal: 14
      lignite: 4
      oil: 45
      nuclear: 2.6
      uranium: 2.6
      biomass: 25
    efficiency:
      coal: 0.40
      lignite: 0.42
  emission_prices:
    enable: true
    co2: 65.0   # float literal required -- see warning below
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
| gas | 42.90 | World Bank Commodity Markets + TYNDP 2024 scenario inputs (2020 EUR) | **34** | TTF day-ahead 2024 calendar avg ≈ €34 (Q1 ~27, Q4 ~45) | ICE Endex TTF; EEX spot indices; ACER wholesale market reports |
| coal | 7.82 | TYNDP 2024 scenario inputs, IEA 2022 APS (2021 EUR) | **14** | API2 CIF ARA 2024 avg ≈ $105–115/t → ≈ €13–15/MWh_th at 6.98 MWh/t, 0.92 EUR/USD | Argus/McCloskey API2; EEX coal futures; Trading Economics |
| lignite | 7.94 | TYNDP 2024, Booz&co Lignite G2 (2021 EUR) | **4** | Mine-mouth convention, €3–5 typical | EURACOAL Coal in Europe; Öko-Institut/Agora lignite studies |
| oil | 43.00 | World Bank + TYNDP, crude +5 % heavy-oil markup (2020 EUR) | **45** | Brent 2024 avg ~$80.5/bbl → ≈ €44–46/MWh incl. markup | EIA Brent series; OPEC MOMR |
| nuclear (fuel) | 7.45 | TYNDP 2024, EIA 2022 (2021 EUR) | **2.6** | Front-end fuel cycle at ESA 2024 multiannual contract price (142.26 EUR/kgU) + UxC conversion (~$55/kgU) / enrichment (~$160/SWU) / fabrication (~$350/kg); WNA arithmetic, 45 GWd/tU burnup | Euratom Supply Agency Market Observatory (official EU utility prices); UxC; WNA Economics of Nuclear Power |
| biomass | 9.35 | IEA 2011 via old PyPSA assumptions (2015 EUR) | **25** | NWE industrial wood-pellet wholesale 2024 ≈ €120–140/t ÷ 4.8 MWh/t | Bioenergy Europe Statistical Report; EUROSTAT |
| CO₂ | (off by default) | — | **65 EUR/tCO₂** | EEX EUA front-year 2024 calendar avg ≈ €65 | EEX EUA auctions; Ember carbon price tracker |

Key reading of the comparison: the upstream values are **TYNDP scenario
conventions in 2020/2021 money**, not observed prices. Upstream gas
(42.90) is ~25 % above the 2024 TTF average; upstream coal (7.82) is
roughly **half** the API2-implied 2024 value; upstream biomass (9.35) is
a 2011-vintage number. Our overrides move every carrier to the observed
2024 regime. All override values sit at the midpoint of their observed-2024 source
band (reconciled 2026-06-12; the initial draft used rough placeholders
of 32/11/50 for gas/coal/oil that did not match the cited bands).

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

**⚠ CO₂ price must be a YAML float literal (`65.0`, not `65`).**
`prepare_network.py` applies the flat emission price only under
`isinstance(co2, float)`; a YAML integer silently falls through with no
warning and **no CO₂ cost reaches any generator**. Caught by the
generation-mix validation gate on 2026-06-12: without CO₂ the merit
order inverts (lignite 16.2, coal 35.0 undercut CCGT 61.9 EUR/MWh) and
the baseline dispatched DE hard coal at +439 % of observed 2024 while
gas collapsed to ~zero continent-wide. With the float literal the
effective marginal costs become lignite ≈ 96, hard coal ≈ 96,
CCGT ≈ 84, OCGT ≈ 117 EUR/MWh — matching the observed 2024 gas-before-
coal order. All solves before this date carried the artefact.

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

### 2.1b Marginal-unit efficiency overrides (coal / lignite)
```yaml
costs:
  overwrites:
    efficiency:
      coal: 0.40
      lignite: 0.42
```
`costs_2025.csv` carries generic fleet-average efficiencies (lignite
0.33, hard coal 0.356) under which, at EUA 65 EUR/t, lignite prices at
96 EUR/MWh — *above* CCGT at 85 — and modelled coal generation
collapses to −82 % of observed 2024 (gate finding, 2026-06-12). The
2024 *operating* fleet is far more efficient after the phase-out
closures: German lignite is dominated by BoA-class units (Neurath F/G
43.6 %, Niederaussem K 43.2 %, Lippendorf 42.5 %; Öko-Institut 2017,
*Die deutsche Braunkohlenwirtschaft*, unit-level data), and the
remaining hard-coal fleet mixes modern German units (Datteln 4 ~45 %)
with the Polish fleet (new 45–46 %, 200-MW class ~37 %; Schröder et
al. 2013, DIW Data Documentation 68: existing ~39 %, new 46 %).
Resulting SRMC: lignite ≈ 77, CCGT ≈ 85, hard coal ≈ 86 EUR/MWh —
reproducing the observed 2024 order (lignite baseload, gas and coal
alternating at the margin). Efficiency also scales specific emissions
per MWh_el consistently.

### 2.1c Wind + solar annual-energy anchoring to observed 2024
**Module**: `constraint_ladder/helpers/vre_anchoring.py`, applied at
solve time by every run script (after `reset_to_lp_baseline`).
**Targets**: `constraint_ladder/data/vre_zone_targets_2024.csv`, built
by `pypsa_setup/data_patches/build_vre_targets_2024.py` from ENTSO-E
actual generation per bidding zone (fetched 2026-06-12; mixed 15/30/60-
minute publication resolution resampled to hourly means before
integrating).
**Operation**: per anchoring group (usually one bidding zone; DE00+LUG1
and IE00+GBNI share their ENTSO-E areas), scales the wind / solar
`p_max_pu` series by a single factor (clipped at 1.0, iteratively
re-converged) so annual available energy matches observed 2024
generation. Hourly shape stays 2024 meteorology.
**Consistency rule**: per country and carrier, if the ENTSO-E total is
within 10 % of Ember's yearly value → ENTSO-E per-zone targets used
directly (wind: 20 of 25 countries). Otherwise ENTSO-E zonal *shares*
are kept but the country total is rescaled to Ember, which corrects the
known Transparency-Platform coverage gaps (distributed solar: NL −98 %,
HR −70 %, SE −56 %, IT −23 %, DE −14 % vs Ember). GB and the western
Balkans (no ENTSO-E coverage) use Ember country totals. The basis per
group is recorded in the targets CSV.
**Why needed**: the raw cutout shows the documented reanalysis wind
bias (Staffell & Pfenninger 2016; Jourdier 2020) — GB +52 %, NL +29 %,
DE +22 % over; IT −43 %, ES −29 % under — plus missing
wake/availability/curtailment losses. A ±30–50 % wind-energy error
reshapes residual load and the zero-price / gas-setting hours,
contaminating the price-shape validation.
**Caveat**: observed generation embeds curtailment, slightly
understating available energy in high-VRE zones (conservative).
**Hydro**: deliberately NOT switched to ENTSO-E — its hydro totals
under-report against Ember in CH (−64 %), DE (−26 %), AT (−20 %)
due to small-plant coverage, so the Ember-based EIA patch (§2.2)
remains the hydro anchor.

### 2.2 EIA hydro statistics extension to 2024
**Script**: `pypsa_setup/data_patches/extend_eia_hydro_2024.py`.
**Touches**: `pypsa-eur/data/eia_hydro_annual_generation.csv`.
**Operation**: Appends a 2024 column with observed net hydro generation
per country from Ember's yearly electricity dataset (retrieved
2026-06-12).
**Why needed**: `build_hydro_profile` normalises weather-year inflow to
the EIA annual statistic of the matching calendar year; the bundled
file ends at 2023 and missing years fall back to the per-country
**1980–2023 median**. 2024 was a wet year (ES 34.4 vs ~25 TWh in 2023,
IT 53.1, AT 45.7, NO 139.6), so median normalisation under-delivered
hydro by 20–30 % in NO/AT/ES/IT/FR — caught by the generation-mix gate.
**Run-time**: idempotent; no-op once the 2024 column exists.

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
