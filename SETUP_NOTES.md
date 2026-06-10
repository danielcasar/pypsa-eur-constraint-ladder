# PyPSA-Eur — Setup Notes (current paper)

## Pinned versions

- **PyPSA-Eur workflow**: [`v2026.02.0`](https://github.com/PyPSA/pypsa-eur/releases/tag/v2026.02.0)
  (released 2026-02-19). Frozen for this paper's life cycle; do not bump until
  after submission.
- **PyPSA Python library**: whatever `pypsa-eur/workflow/envs/environment.yaml`
  pins for the chosen PyPSA-Eur tag (currently the `pypsa>=1.0` series).
- **Gurobi**: 13.x on Marvin (academic licence already active on the server).

## How this repo couples to PyPSA-Eur

We do **not** vendor PyPSA-Eur. Coupling is at two layers:

1. **File-level.** PyPSA-Eur's Snakemake workflow produces a prepared network
   (`.nc` file) that encodes the 2025 fleet, load, NTC, and carrier set for
   the target zones. Our `code/driver/example.py` loads that file as input
   and applies the constraint-activation helpers on top.
2. **Library-level.** Both PyPSA-Eur and our code `import pypsa` directly.
   The `pypsa` library API (`n.optimize.create_model`, `n.add`,
   `n.buses_t.marginal_price`, etc.) is the actual stable surface. Pinning
   PyPSA-Eur to `v2026.02.0` transitively pins the `pypsa` library version
   that gets installed.

PyPSA-Eur lives in a **separate clone alongside our repo** on the server,
not inside it. The pinned tag in this file is the reproducibility anchor.

## Server-side install (Marvin)

```bash
# project root (parallel to the lng-model layout)
cd /projects2/daniel
mkdir -p pypsa-eur-constraint-ladder && cd pypsa-eur-constraint-ladder

# our repo
git clone https://github.com/danielcasar/pypsa-eur-constraint-ladder.git ladder

# upstream PyPSA-Eur at the pinned tag
git clone https://github.com/PyPSA/pypsa-eur.git pypsa-eur
cd pypsa-eur && git checkout v2026.02.0 && cd ..

# environment from the pinned PyPSA-Eur's environment.yaml
mamba env create -f pypsa-eur/envs/environment.yaml -n pypsa-eur-ladder
mamba activate pypsa-eur-ladder

# install our code on top so `from constraints import ...` etc. just works
pip install -e ladder
```

After this, the constraint modules and helpers are importable in any Python
session under the `pypsa-eur-ladder` env. See `code/driver/example.py` for
the manual step-by-step walkthrough.

## Continental 2025 prepared network

Run from inside `pypsa-eur/`:

```bash
snakemake -j 20 \
    --configfile ../configs/eu_2025_dispatch.yaml \
    -- resources/eu_2025_dispatch/networks/base_s_50_elec_.nc
```

The override config at `configs/eu_2025_dispatch.yaml` keeps every
upstream default except: run name (`eu_2025_dispatch`), 50 clusters,
2025 calendar, dispatch-only (empty `extendable_carriers` on both
`electricity` and `sector`), and Gurobi barrier as the LP method. The
resulting `.nc` is what the manual workflow in `code/driver/example.py`
loads (update `NETWORK_PATH` accordingly).
