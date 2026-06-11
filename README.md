# pypsa-eur-constraint-ladder

Reproducibility repo for "Narrowing the model-market gap in European
day-ahead prices: a constraint-based analysis in PyPSA-Eur".

## Layout

```
.
├── constraint_ladder/    Python package -- per-deviation activators + native helpers
│   ├── constraints/      (price_elastic_demand, subsidy_floor, subsidy_floor_de)
│   └── helpers/          (calibration loader, native_constraints)
├── pypsa_setup/          Scaffolding to take vanilla PyPSA-Eur v2026.02.0 to a 2024 baseline
│   ├── configs/          (Snakemake config overrides)
│   └── data_patches/     (one-off scripts that extend bundled data)
├── runs/                 Thin per-configuration solve scripts (baseline first; ladder steps to follow)
├── pyproject.toml        Editable-install metadata
├── SETUP_NOTES.md        End-to-end server install + workflow
├── PYPSA_EUR_OVERRIDES.md  Every deviation we apply on top of vanilla PyPSA-Eur
└── LICENSE               MIT
```

PyPSA-Eur itself is **not** vendored; it is cloned separately alongside this
repo on the server. See `SETUP_NOTES.md`.

## Quick start (server side)

```bash
cd ~/projects2/daniel/pypsa-eur-constraint-ladder
mamba activate pypsa-eur-ladder
python runs/baseline.py
```
