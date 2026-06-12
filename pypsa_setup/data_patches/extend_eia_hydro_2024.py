"""Append observed 2024 hydro generation to the bundled EIA statistics.

PyPSA-Eur's ``build_hydro_profile`` normalises the weather-year inflow
so that annual hydro energy per country matches the EIA annual
generation statistic for that calendar year. The bundled
``data/eia_hydro_annual_generation.csv`` in PyPSA-Eur v2026.02.0 ends
at 2023, and the fallback for missing years is the per-country
**1980--2023 median** (``build_hydro_profile.py``, ``missing_years``
branch). 2024 was a wet year in Iberia, the Alps, and Scandinavia, so
median-normalised inflow under-delivers hydro by 20--30 % in the
affected countries (caught by the generation-mix validation gate,
2026-06-12).

This patch appends a ``2024`` column with observed 2024 net hydro
generation per country, taken from Ember's yearly electricity dataset
(Yearly Full Release, retrieved 2026-06-12; category "Electricity
generation", variable "Hydro", TWh). Ember's "Hydro" excludes pure
pumped storage, matching the EIA convention used by the file.

Idempotent: a second run is a no-op once the 2024 column exists.

Usage (from the directory containing ``pypsa-eur/``):

    python ladder/pypsa_setup/data_patches/extend_eia_hydro_2024.py
"""

from __future__ import annotations

from pathlib import Path

EIA_CSV = Path("pypsa-eur/data/eia_hydro_annual_generation.csv")

# Ember yearly dataset, 2024, "Electricity generation" / "Hydro" / TWh.
# Keys are the display names used in the EIA CSV.
EMBER_HYDRO_2024_TWH: dict[str, float] = {
    "Albania": 8.0,
    "Austria": 45.7,
    "Belgium": 0.6,
    "Bosnia and Herzegovina": 5.4,
    "Bulgaria": 2.8,
    "Croatia": 6.7,
    "Czechia": 2.7,
    "Denmark": 0.0,
    "Estonia": 0.0,
    "Finland": 14.3,
    "France": 71.1,
    "Germany": 23.8,
    "Greece": 3.6,
    "Hungary": 0.2,
    "Ireland": 0.8,
    "Italy": 53.1,
    "Kosovo": 0.2,
    "Latvia": 3.2,
    "Lithuania": 0.4,
    "Luxembourg": 0.1,
    "Montenegro": 1.8,
    "Netherlands": 0.1,
    "North Macedonia": 1.4,
    "Norway": 139.6,
    "Poland": 2.1,
    "Portugal": 14.9,
    "Romania": 14.2,
    "Serbia": 10.1,
    "Slovakia": 4.8,
    "Slovenia": 5.3,
    "Spain": 34.4,
    "Sweden": 64.6,
    "Switzerland": 44.9,
    "United Kingdom": 5.8,
}


def main() -> None:
    lines = EIA_CSV.read_text(encoding="utf-8-sig").splitlines()

    # Row 3 (index 2) is the header row: "API,,1980,1981,...,2023".
    header_idx = next(
        i for i, l in enumerate(lines) if l.startswith("API")
    )
    if lines[header_idx].rstrip(",").endswith("2024"):
        print("2024 column already present -- nothing to do.")
        return
    assert lines[header_idx].rstrip(",").endswith("2023"), (
        "unexpected last year in header: " + lines[header_idx][-30:]
    )
    lines[header_idx] = lines[header_idx].rstrip(",") + ",2024"

    n_filled = 0
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].strip():
            continue
        parts = lines[i].split(",")
        if len(parts) < 2:
            continue
        name = parts[1].strip().strip('"').strip()
        value = EMBER_HYDRO_2024_TWH.get(name)
        if value is None:
            lines[i] = lines[i] + ","
        else:
            lines[i] = lines[i] + f",{value}"
            n_filled += 1

    EIA_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"appended 2024 column: {n_filled} countries filled from Ember")


if __name__ == "__main__":
    main()
