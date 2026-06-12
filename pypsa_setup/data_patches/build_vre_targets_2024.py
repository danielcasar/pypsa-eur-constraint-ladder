"""Build per-zone VRE anchoring targets from the fetched ENTSO-E data.

Reads the per-zone generation-by-type CSVs in
``Analysis/data/ENTSOE_generation_2024/`` (mixed 15-min/30-min/hourly
resolution -- resampled to hourly means before integrating) and writes
the anchoring-target table consumed by
``constraint_ladder/helpers/vre_anchoring.py``.

Consistency rule (documented in the paper): per country and carrier,
if the ENTSO-E total is within 10 % of Ember's yearly value, the
ENTSO-E per-zone targets are used directly. Otherwise the ENTSO-E
zonal *shares* are kept but rescaled so the country total matches
Ember (Ember corrects known Transparency-Platform coverage gaps --
distributed solar in particular -- against national statistics).
Zones with no ENTSO-E coverage (GB, western Balkans) use Ember
country totals directly.

Output: ``constraint_ladder/data/vre_zone_targets_2024.csv`` with
columns ``group`` (anchoring group id), ``zones`` (semicolon-joined
PyPSA zone prefixes), ``carrier_group`` (wind / solar),
``target_twh``, ``basis``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "data" / "ENTSOE_generation_2024"
OUT = HERE.parent / "constraint_ladder" / "data" / "vre_zone_targets_2024.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ENTSO-E zone -> list of PyPSA zone prefixes it covers.
ZONE_MAP: dict[str, list[str]] = {
    "AT": ["AT00"], "BE": ["BE00"], "BG": ["BG00"], "CH": ["CH00"],
    "CZ": ["CZ00"], "DE_LU": ["DE00", "LUG1"],
    "DK_1": ["DKW1"], "DK_2": ["DKE1"], "EE": ["EE00"], "ES": ["ES00"],
    "FI": ["FI00"], "FR": ["FR00"], "GR": ["GR00"], "HR": ["HR00"],
    "HU": ["HU00"], "IE_SEM": ["IE00", "GBNI"],
    "IT_NORD": ["ITN1"], "IT_CNOR": ["ITCN"], "IT_CSUD": ["ITCS"],
    "IT_SUD": ["ITS1"], "IT_CALA": ["ITCA"], "IT_SICI": ["ITSI"],
    "IT_SARD": ["ITSA"], "LT": ["LT00"], "LV": ["LV00"], "NL": ["NL00"],
    "NO_1": ["NOS1"], "NO_2": ["NOS2"], "NO_3": ["NOM1"],
    "NO_4": ["NON1"], "NO_5": ["NOS5"], "PL": ["PL00"], "PT": ["PT00"],
    "RO": ["RO00"], "SE_1": ["SE01"], "SE_2": ["SE02"], "SE_3": ["SE03"],
    "SE_4": ["SE04"], "SI": ["SI00"], "SK": ["SK00"], "RS": ["RS00"],
}

# ENTSO-E zone -> ISO2 country (for the Ember consistency check).
ZONE_COUNTRY: dict[str, str] = {
    z: ("DE" if z == "DE_LU" else "IE" if z == "IE_SEM" else z.split("_")[0])
    for z in ZONE_MAP
}

# Ember yearly 2024, TWh (retrieved 2026-06-12). Country level.
EMBER_2024 = {
    "wind": {
        "AT": 9.1, "BE": 14.1, "BG": 1.5, "CH": 0.2, "CZ": 0.7,
        "DE": 141.6, "DK": 20.4, "EE": 1.0, "ES": 62.2, "FI": 20.5,
        "FR": 45.4, "GB": 83.3, "GR": 12.6, "HR": 2.7, "HU": 0.7,
        "IE": 11.6, "IT": 22.2, "LT": 3.4, "LV": 0.3, "NL": 33.5,
        "NO": 14.6, "PL": 25.9, "PT": 14.3, "RO": 6.4, "RS": 1.2,
        "SE": 40.4, "SI": 0.0, "SK": 0.0,
        "AL": 0.0, "BA": 0.4, "ME": 0.3, "MK": 0.2, "XK": 0.4,
    },
    "solar": {
        "AT": 8.1, "BE": 8.6, "BG": 5.5, "CH": 5.7, "CZ": 3.6,
        "DE": 74.1, "DK": 3.8, "EE": 1.1, "ES": 58.3, "FI": 0.9,
        "FR": 25.1, "GB": 14.8, "GR": 11.3, "HR": 0.8, "HU": 9.2,
        "IE": 1.1, "IT": 36.0, "LT": 1.4, "LV": 0.5, "NL": 21.8,
        "NO": 0.5, "PL": 17.7, "PT": 7.1, "RO": 3.4, "RS": 0.2,
        "SE": 4.2, "SI": 1.3, "SK": 0.7,
        "AL": 0.3, "BA": 0.4, "ME": 0.1, "MK": 0.9, "XK": 0.0,
    },
}

# Zones with no ENTSO-E coverage: Ember country totals, one group each.
EMBER_ONLY_GROUPS: dict[str, list[str]] = {
    "GB": ["GB00", "GBZE"],
    "AL": ["AL00"], "BA": ["BA00"], "ME": ["ME00"], "MK": ["MK00"],
    "XK": ["XK00"],
}

WIND_COLS = ("Wind Onshore", "Wind Offshore")
SOLAR_COLS = ("Solar",)


def zone_energy_twh(zone: str) -> dict[str, float]:
    df = pd.read_csv(GEN_DIR / f"{zone}.csv", index_col=0, parse_dates=True)
    hourly = df.resample("1h").mean()
    out = {}
    for grp, cols in (("wind", WIND_COLS), ("solar", SOLAR_COLS)):
        present = [c for c in cols if c in hourly.columns]
        out[grp] = float(hourly[present].sum().sum() / 1e6) if present else 0.0
    return out


def main() -> None:
    per_zone = {z: zone_energy_twh(z) for z in ZONE_MAP}

    rows = []
    print(f"{'country':8s} {'carrier':6s} {'entsoe':>8s} {'ember':>8s} {'dev':>6s}  rule")
    for grp in ("wind", "solar"):
        # group ENTSO-E zones by country for the consistency check
        by_country: dict[str, list[str]] = {}
        for z in ZONE_MAP:
            by_country.setdefault(ZONE_COUNTRY[z], []).append(z)
        for country, zones in sorted(by_country.items()):
            entsoe_total = sum(per_zone[z][grp] for z in zones)
            ember = EMBER_2024[grp].get(country)
            if ember is None:
                continue
            if ember < 0.05 and entsoe_total < 0.05:
                continue
            if ember > 0 and abs(entsoe_total - ember) / ember <= 0.10:
                basis, scale = "entsoe", 1.0
            elif entsoe_total > 0:
                basis, scale = "entsoe_rescaled_ember", ember / entsoe_total
            else:
                basis, scale = "ember", None
            dev = (entsoe_total - ember) / ember * 100 if ember else float("nan")
            print(
                f"{country:8s} {grp:6s} {entsoe_total:>8.2f} {ember:>8.2f} "
                f"{dev:>+5.0f}%  {basis}"
            )
            if basis == "ember":
                rows.append(
                    {
                        "group": f"{country}_{grp}",
                        "zones": ";".join(p for z in zones for p in ZONE_MAP[z]),
                        "carrier_group": grp,
                        "target_twh": round(ember, 3),
                        "basis": basis,
                    }
                )
            else:
                for z in zones:
                    rows.append(
                        {
                            "group": f"{z}_{grp}",
                            "zones": ";".join(ZONE_MAP[z]),
                            "carrier_group": grp,
                            "target_twh": round(per_zone[z][grp] * scale, 3),
                            "basis": basis,
                        }
                    )
        # Ember-only countries (no ENTSO-E coverage at all)
        for country, prefixes in EMBER_ONLY_GROUPS.items():
            ember = EMBER_2024[grp].get(country, 0.0)
            if ember <= 0.05:
                continue
            print(f"{country:8s} {grp:6s} {'-':>8s} {ember:>8.2f} {'':>6s}  ember (no ENTSO-E)")
            rows.append(
                {
                    "group": f"{country}_{grp}",
                    "zones": ";".join(prefixes),
                    "carrier_group": grp,
                    "target_twh": round(ember, 3),
                    "basis": "ember",
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"\nwrote {len(out_df)} target rows to {OUT}")

    # Hydro comparison for the EIA-patch decision (reservoir + ror,
    # excluding pumped storage).
    print("\n=== hydro (reservoir + ror + poundage), ENTSO-E vs Ember, TWh ===")
    HYDRO_COLS = ("Hydro Water Reservoir", "Hydro Run-of-river and poundage")
    EMBER_HYDRO = {
        "AT": 45.7, "BE": 0.6, "BG": 2.8, "CH": 44.9, "CZ": 2.7,
        "DE": 23.8, "ES": 34.4, "FI": 14.3, "FR": 71.1, "GR": 3.6,
        "HR": 6.7, "HU": 0.2, "IT": 53.1, "NO": 139.6, "PL": 2.1,
        "PT": 14.9, "RO": 14.2, "RS": 10.1, "SE": 64.6, "SI": 5.3,
        "SK": 4.8, "LV": 3.2, "LT": 0.4, "EE": 0.0, "IE": 0.8,
    }
    by_country = {}
    for z in ZONE_MAP:
        by_country.setdefault(ZONE_COUNTRY[z], []).append(z)
    for country, zones in sorted(by_country.items()):
        if country not in EMBER_HYDRO:
            continue
        tot = 0.0
        for z in zones:
            df = pd.read_csv(GEN_DIR / f"{z}.csv", index_col=0, parse_dates=True)
            hourly = df.resample("1h").mean()
            present = [c for c in HYDRO_COLS if c in hourly.columns]
            tot += float(hourly[present].sum().sum() / 1e6) if present else 0.0
        ember = EMBER_HYDRO[country]
        if ember < 0.5 and tot < 0.5:
            continue
        dev = (tot - ember) / ember * 100 if ember else float("nan")
        print(f"  {country}: entsoe {tot:7.2f}  ember {ember:7.2f}  ({dev:+.0f}%)")


if __name__ == "__main__":
    main()
