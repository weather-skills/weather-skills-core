#!/usr/bin/env python3
"""Rebuild the bundled Natural Earth countries GeoJSON.

`resolve-region` loads `src/weather_skills_core/data/countries.geojson` for
ISO3 / country-name lookups and for multi-country names (`East Africa`,
`Sub-Saharan Africa`, …). This script is how that file is produced. It is
not on the library hot path — run it when Natural Earth updates, or when
the slim property set changes.

Source
    Natural Earth 1:50m admin-0 (the 1:110m file leaves out small countries
    such as Mauritius, Seychelles and Comoros)
    https://github.com/nvkelso/natural-earth-vector
    `geojson/ne_50m_admin_0_countries.geojson` on `master`

Writes
    `src/weather_skills_core/data/countries.geojson` — one Feature per
    country, sorted by iso3. Geometry is copied from Natural Earth with
    coordinates rounded to 4 decimals (~11 m), which keeps the file ~2 MB. Properties
    kept:

        iso3        ISO 3166-1 alpha-3 (see ISO3 below)
        name        Natural Earth `NAME`, else `ADMIN`
        continent   `CONTINENT`
        region_un   `REGION_UN`
        subregion   `SUBREGION`
        region_wb   `REGION_WB`

    `resolve-region` groups features by those four region fields at runtime.
    There is no sidecar.

ISO3
    `ISO_A3`, else `ISO_A3_EH`, else `ADM0_A3`. Natural Earth stores `-99`
    on some disputed / unassigned codes; those are not three letters and
    fall through. Kosovo `KOS` is stored as `XKX`. The script exits if any
    feature has no ISO3 and no name.

Usage:
    uv run python tools/build_countries.py
    uv run python tools/build_countries.py --help
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

_NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
_OUT = Path(__file__).resolve().parents[1] / "src/weather_skills_core/data/countries.geojson"


def iso3(props: dict) -> str | None:
    """Return a 3-letter ISO3 from Natural Earth admin-0 properties, or None."""
    for key in ("ISO_A3", "ISO_A3_EH", "ADM0_A3"):
        val = props.get(key)
        if isinstance(val, str) and len(val) == 3 and val.isalpha():
            return "XKX" if val.upper() == "KOS" else val.upper()
    return None


def rounded(coords):
    """Nested coordinate lists with every number rounded to 4 decimals."""
    if isinstance(coords, list):
        return [rounded(c) for c in coords]
    return round(coords, 4)


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    req = urllib.request.Request(_NE_URL, headers={"User-Agent": "weather-skills-core-dev"})
    with urllib.request.urlopen(req, timeout=60) as response:
        ne = json.loads(response.read().decode())

    features = []
    skipped = []
    for src in ne["features"]:
        props = src["properties"]
        code = iso3(props)
        name = props.get("NAME") or props.get("ADMIN")
        if code is None or not name:
            skipped.append(name or props.get("ADMIN") or "?")
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "iso3": code,
                    "name": name,
                    "continent": props.get("CONTINENT") or None,
                    "region_un": props.get("REGION_UN") or None,
                    "subregion": props.get("SUBREGION") or None,
                    "region_wb": props.get("REGION_WB") or None,
                },
                "geometry": {
                    "type": src["geometry"]["type"],
                    "coordinates": rounded(src["geometry"]["coordinates"]),
                },
            }
        )

    if skipped:
        raise SystemExit(f"no ISO3/name for {skipped}")

    features.sort(key=lambda f: f["properties"]["iso3"])
    payload = {
        "type": "FeatureCollection",
        "_source": "Natural Earth 50m admin-0 (nvkelso/natural-earth-vector)",
        "features": features,
    }
    _OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"wrote {_OUT} ({len(features)} countries)")


if __name__ == "__main__":
    main()
