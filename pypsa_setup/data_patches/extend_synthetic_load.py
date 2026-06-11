"""Extend PyPSA-Eur's bundled synthetic-load file beyond its native 2023 end.

The file ``data/synthetic_electricity_demand/primary/v2/load_synthetic_raw.csv``
shipped with PyPSA-Eur v2026.02.0 (and most prior tags) ends at
2023-12-31 23:00. The Snakemake rule ``build_electricity_demand`` selects the
full snapshot window from this file as a gap-fill source even for countries
whose real series fully covers the window, so any snapshot range that exceeds
2023 raises a KeyError.

For the constraint-ladder paper we calibrate against observed 2025 prices, so
we extend the synthetic file by appending re-dated copies of recent
non-leap-year profiles to 2024 and 2025. The extension is shape-only:

- 2024 (leap)     <- copy of 2020 (most recent leap year in the file)
- 2025 (non-leap) <- copy of 2023 (most recent non-leap year)

For countries where real ENTSO-E / OPSD data covers 2024-25, the gap-fill is
unused and this extension has no effect on the run; only the small-market
gap-fill (BA, ME, XK, MD, UA, ...) sees the shifted 2020 / 2023 shape.
Document this in the paper's limitations.

Run once from inside ``pypsa-eur/`` after the upstream data has been
retrieved by Snakemake (i.e., after the first ``retrieve_*`` rules
complete). Idempotent: running it again on an already-extended file is a
no-op (the post-2023 rows are detected and not re-appended).

    python ../ladder/tools/extend_synthetic_load.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SYNTHETIC_PATH = Path(
    "data/synthetic_electricity_demand/primary/v2/load_synthetic_raw.csv"
)

# (target_year, template_year) pairs. Template must be present in the file
# and match the target's leap/non-leap status for a one-to-one date map.
EXTENSIONS: tuple[tuple[int, int], ...] = (
    (2024, 2020),
    (2025, 2023),
)


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def main(path: Path = SYNTHETIC_PATH) -> int:
    if not path.exists():
        print(f"[error] synthetic load file not found at {path}", file=sys.stderr)
        return 2

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    last_native = df.index.max()
    print(f"loaded {len(df):,} rows, native range "
          f"{df.index.min()} -> {last_native}")

    appended: list[pd.DataFrame] = []
    for target_year, template_year in EXTENSIONS:
        if df.index.year.max() >= target_year:
            print(f"  skip {target_year}: already present (max year "
                  f"{df.index.year.max()})")
            continue
        if _is_leap(target_year) != _is_leap(template_year):
            print(f"[error] leap mismatch: target {target_year} vs "
                  f"template {template_year}", file=sys.stderr)
            return 3
        template = df.loc[str(template_year)]
        if len(template) == 0:
            print(f"[error] template year {template_year} has zero rows",
                  file=sys.stderr)
            return 4
        shifted = template.copy()
        shifted.index = shifted.index.map(
            lambda dt, ty=target_year: dt.replace(year=ty)
        )
        appended.append(shifted)
        print(f"  staged {target_year} <- {template_year} "
              f"({len(shifted)} rows)")

    if not appended:
        print("nothing to append, file already covers the requested years")
        return 0

    extended = pd.concat([df, *appended])
    extended = extended.sort_index()
    extended.to_csv(path)
    print(f"wrote {len(extended):,} rows to {path}, "
          f"new range {extended.index.min()} -> {extended.index.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
