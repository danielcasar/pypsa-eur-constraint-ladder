# Electricity Price Evolution — Task list

Last updated: 2026-05-13

(Working title; folder path still reads "Shadow prices 2.0" until folder rename.)

## Pillars

- **A. Cross-ESOM validation** — extend beyond GENeSYS-MOD + PyPSA-Eur
- **B. Temporal-pattern shift across model planning years** — how do calendar effects evolve with decarbonisation?
- **B′. Structural shift within ENTSO-E** — how do calendar effects evolve in the *observed* market over 2018–2025?
- **C. Identification of relevant constraints** — perturbation ladder; which model features close the gap?
- **D. Use-case taxonomy of future-electricity-price needs** — who actually needs hourly prices vs. daily averages vs. just price levels? Map applications to data requirements.
- **E. Unified framework: ESOM shadow prices ⊕ observed temporal patterns** — bridge the two strands developed so far into one coherent methodology.

---

## This week (no compute, no blockers)

- [ ] Email TU Wien sysadmin: request access to the institute compute server (RAM, cores, OS, Gurobi license model)
- [ ] Email ECMF secretariat: submission format, deadlines, abstract length

---

## Months 1–2 — Foundations (mostly local; some server-dependent)

### Pillar B′ — ENTSO-E own structural shift  *(can do entirely on laptop now)*
- [ ] Apply Stage 1/2/3 + compression factors year-by-year to ENTSO-E DE 2018–2025
- [ ] Quantify year-over-year change in ε² per calendar dimension
- [ ] Identify regime breaks (pre-2022, 2022 crisis, 2023+) and characterise the shift
- [ ] Repeat for AT, FR, ES, NL — does the shift pattern generalise across countries?
- [ ] Reuse: `Code/paper/scripts/compute_entsoe_cross_year_pearson.py` and friends

### Pillar B — Cross-year model comparison (PyPSA-Eur, GENeSYS-MOD)
- [x] PyPSA-Eur DE 2050 — solved, framework applied
- [ ] PyPSA-Eur DE 2030 — **stabilise** (currently OOM at 31 GB; in progress)
- [ ] PyPSA-Eur DE 2035 *(server)*
- [ ] PyPSA-Eur DE 2045 *(server)*
- [ ] GENeSYS-MOD investment-only runs for 4 EnVis scenarios at full hourly *(server)*
- [ ] Apply paper framework to the (model × year × scenario) grid

### Deliverable
- [ ] **ECMF abstract** — submit when format/dates known. Headline: "structural-shift trajectory across decarb pathways, mirrored & distorted in two structurally different ESOMs"

---

## Months 3–4 — Pillar C (server-dependent — this is the paper's novel contribution)

### Constraint perturbation ladder
- [ ] Baseline: PyPSA-Eur DE 2030 (existing thermal fleet → directly comparable to ENTSO-E history)
- [ ] +Unit commitment + startup costs (MIP — server only)
- [ ] +Ramping constraints
- [ ] +Reserve requirements / ancillary services
- [ ] +Scarcity pricing / VOLL layer
- [ ] +Higher temporal granularity (sub-hourly representation)
- [ ] After each step: recompute compression factors + Stage 3 correlations vs ENTSO-E
- [ ] Plot: gap-closure attribution per calendar dimension

---

## Months 4–6 — Pillar A (validation across other ESOMs)

### Pick ONE additional model — don't try three
- [ ] Decision: Calliope vs. oemof vs. harvested IAMC outputs *(decide month 4)*
- [ ] Set up the chosen model for DE
- [ ] Match scenario (high RES, equivalent decarb level — no exact equivalence needed)
- [ ] Apply paper framework
- [ ] Add to model-comparison table

### Cheap parallel track — published ESOM outputs
- [ ] Search Zenodo / IAMC databases for hourly electricity prices: REMIND, MESSAGE, IMAGE, TIMES-PanEU
- [ ] Compute compression factors from published outputs (no rerun needed)

---

## Notes / risks

- **Server access is critical path** for everything in months 3–6. Push hard if it slips past month 2.
- **Don't strict-translate scenarios** between models — the paper is about structural patterns, not absolute values. Each model runs its own native scenario; comparison happens on derived structural metrics (compression factor, calendar profile correlations).
- **Hourly resolution non-negotiable** for the framework — Stage 1/2/3 require 24-hour groups for the hour dimension. Resampling to 3h breaks the comparison.

---

## Pillar D — Use-case taxonomy of future-price needs *(literature + interviews, no compute)*

Goal: map who *actually* needs which kind of future electricity-price information. This justifies the structural-shape-forecast framing as a publishable contribution by showing it serves a real user need that is currently mis-served by hourly-point-forecasts.

### Categories to populate
- [ ] **Hourly prices, specific year** (e.g. day-ahead trading R&D, intraday strategy back-testing, storage operation algorithms) — needs full hourly time series
- [ ] **Average daily/weekly profiles** (e.g. heat-pump dispatch design, EV-charging tariff structures, demand-response programs) — needs typical-day shape, not specific dates
- [ ] **Annual averages** (e.g. PPA pricing, LCOE, capture-price forecasts, simple project IRR) — needs one number per year
- [ ] **Distributional summaries** (e.g. battery sizing, scarcity-event hedging, PV PPA volatility) — needs a few percentiles or a CV
- [ ] **Long-run trends** (e.g. transmission planning, capacity-expansion modelling, policy targets) — needs decade-long trajectories

### Aggregation tasks
- [ ] Survey 30-50 papers / industry reports across these categories to count which use which data form
- [ ] Build a 2x2 matrix: (data form needed) × (currently used data form) → identify mismatch
- [ ] Identify the categories where structural-shape forecasts are *the right answer* but currently get hourly point forecasts (= the framework's market)
- [ ] Reach out to 3-5 EEG colleagues / industry contacts across categories to validate the mapping

### Deliverable
- [ ] One-page taxonomy table for the paper's introduction motivating the structural-shape-forecast framing

---

## Pillar E — Unified framework: ESOM shadow prices ⊕ observed temporal patterns

The two strands developed so far:
- **Strand 1**: ESOM shadow prices (GENeSYS-MOD, PyPSA-Eur) — model-derived future prices, structurally rich but compressed/distorted versus reality
- **Strand 2**: Observed temporal patterns (ENTSO-E + capacity-share regression) — empirical, predictive of *shape* from energy-system structure, validated with bootstrap PIs

These two have been treated as separate analyses. The smooth framework would bridge them: use the empirical shape-prediction model as the *yardstick* against which ESOM-projected future shadow-price profiles are evaluated. The ESOM tells you WHAT future scenarios look like; the empirical model tells you HOW the shape SHOULD look given the future system state.

### Conceptual scaffolding
- [ ] Articulate the two strands as **complementary**, not competitive
  - ESOM: forward-looking, scenario-driven, internally consistent, structurally rich
  - Empirical: backward-validated, structure-only predictors, calibrated PIs
- [ ] Define the bridging operation: given an ESOM scenario's projected (capacity, demand, interconnection) for year y, the empirical model produces an *expected* shape under historical-style market dynamics. The ESOM produces its *own* shadow-price shape. The gap = model-induced distortion.
- [ ] The gap quantification IS the contribution: "given the same future system state, how does an ESOM's shadow-price shape differ from what observed-market dynamics would have produced under the same fundamentals?"

### Concrete validation experiments
- [ ] Take EnVis-Green 2030 system state (or PyPSA-Eur 2030 outputs)
- [ ] Run the empirical shape regression with that system state → predicted observed-style profile + PI
- [ ] Compare to the ESOM's own shadow-price profile for that same year/scenario
- [ ] Decompose the gap by calendar dimension (hour/weekday/month/season)
- [ ] Repeat for 2050 to show how the gap evolves with decarbonisation

### Two paper hypotheses
- [ ] **H1**: Gap closes for low-resolution dimensions (month, season) because annual averages are similar, opens for high-resolution (hour) because ESOMs miss flexibility/UC mechanisms
- [ ] **H2**: Gap is invariant to scenario (Green vs. NECP vs. Trinity) at the structural level — shape differences come from model architecture, not scenario assumptions

### Deliverable
- [ ] Framework diagram (one figure) showing the two strands feeding into a comparison
- [ ] The unified framework becomes the *spine* of the Shadow Prices 2.0 paper: ESOM gives future structure → empirical model gives expected market response → gap = model artefact catalogue
