# `runs/` -- thin scripts that solve one ladder configuration each

Each file in this directory loads the prepared continental network, applies a
specific set of constraints (the baseline plus whatever the ladder step adds
on top), solves with Gurobi, and writes the resulting shadow prices to disk.
Scripts are designed to be:

- **Independent**: each one re-loads the prepared `.nc` and applies all
  cumulative constraints up to and including its own ladder step. There is no
  shared state between scripts.
- **Self-contained**: ~30-50 lines, copy-paste-readable.
- **Editable**: change a constant at the top to use a different network or
  output dir; everything else stays the same.

Run from the project root on Marvin, with the `pypsa-eur-ladder` mamba env
active:

```bash
mamba activate pypsa-eur-ladder
python runs/baseline.py
```

| Script | Configuration | LP class |
|---|---|---|
| `baseline.py` | Default LP dispatch | LP |

Additional ladder-step scripts (`voll.py`, `elastic_demand.py`, `reserves.py`,
`rolling_horizon.py`, `unit_commitment.py`, `subsidy_floor.py`) will be added
as the constraint ladder is exercised step by step.
