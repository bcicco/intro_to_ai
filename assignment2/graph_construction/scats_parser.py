"""
scats_parser.py — Step 1: parse SCATS site/approach data from the raw .xls.

Outputs a site-road map dict keyed by zero-padded 4-digit site ID string.
Run as a script to produce boroondara_sites.json and a printed summary.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / "data" / "Scats Data October 2006.xls"
OUT_FILE  = Path(__file__).parent / "data" / "processed" / "boroondara_sites.json"

# Longer compass codes first so "NE" matches before "N", etc.
_LOC_RE = re.compile(
    r"^(.+?)\s+(NE|NW|SE|SW|N|S|E|W)\s+of\s+(.+)$",
    re.IGNORECASE,
)

_OPPOSITE = {
    "N": "S", "S": "N", "E": "W",  "W": "E",
    "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW",
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_scats_data(filepath: str | Path) -> pd.DataFrame:
    """Load the Data sheet, normalise columns, return clean DataFrame."""
    df = pd.read_excel(
        filepath,
        sheet_name="Data",
        skiprows=[0],   # drop the partial timestamp header in row 1
        header=0,
        engine="xlrd",
    )

    df = df.dropna(subset=["SCATS Number"])

    df = df.rename(columns={
        "SCATS Number": "scats_number",
        "Location":     "location",
        "NB_LATITUDE":  "latitude",
        "NB_LONGITUDE": "longitude",
        "Date":         "date",
    })

    # Zero-pad to 4 digits: 970 → "0970", 2000 → "2000"
    df["scats_number"] = (
        df["scats_number"].astype(float).astype(int).astype(str).str.zfill(4)
    )

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    print(f"Loaded           : {len(df):,} rows")
    print(f"Unique sites     : {df['scats_number'].nunique()}")
    print(f"Unique approaches: {df.groupby(['scats_number', 'location']).ngroups}")

    return df


def parse_location_string(loc: str) -> dict:
    """
    Parse a SCATS location string into its components.

    Returns a dict with keys: sensor_road, sensor_side, cross_road,
    travel_direction, raw. All are None (except raw) on parse failure.
    """
    result = {
        "sensor_road":      None,
        "sensor_side":      None,
        "cross_road":       None,
        "travel_direction": None,
        "raw":              loc,
    }
    if not isinstance(loc, str):
        return result

    m = _LOC_RE.match(loc.strip())
    if not m:
        return result

    sensor_side = m.group(2).upper()
    result.update({
        "sensor_road":      m.group(1).strip(),
        "sensor_side":      sensor_side,
        "cross_road":       m.group(3).strip(),
        "travel_direction": _OPPOSITE[sensor_side],
    })
    return result


def build_site_road_map(df: pd.DataFrame) -> dict:
    """
    Build a dict keyed by site_id (4-digit str) containing roads, approaches,
    and averaged lat/lon for each SCATS site.
    """
    # One row per unique (site, location) with averaged lat/lon
    approach_df = (
        df.groupby(["scats_number", "location"], sort=True)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )

    site_map = {}

    for site_id, group in approach_df.groupby("scats_number"):
        approaches = []
        roads: set[str] = set()

        for _, row in group.iterrows():
            parsed = parse_location_string(row["location"])
            approaches.append({
                "location":         row["location"],
                "sensor_road":      parsed["sensor_road"],
                "sensor_side":      parsed["sensor_side"],
                "cross_road":       parsed["cross_road"],
                "travel_direction": parsed["travel_direction"],
                "latitude":         row["latitude"],
                "longitude":        row["longitude"],
            })
            # Pool both sides: sensor road AND the road it crosses
            if parsed["sensor_road"]:
                roads.add(parsed["sensor_road"])
            if parsed["cross_road"]:
                roads.add(parsed["cross_road"])

        # Exclude (0,0) sentinel values from centroid — they are missing-data
        # placeholders, not real coordinates.
        valid = group[~((group["latitude"] == 0.0) & (group["longitude"] == 0.0))]
        if valid.empty:
            print(f"WARNING: site {site_id} has no valid coordinates — centroid set to None")
            site_lat, site_lon = None, None
        else:
            site_lat = valid["latitude"].mean()
            site_lon = valid["longitude"].mean()

        site_map[site_id] = {
            "site_id":    site_id,
            "roads":      roads,
            "approaches": approaches,
            "latitude":   site_lat,
            "longitude":  site_lon,
        }

    return site_map


def summarise(site_road_map: dict) -> None:
    """Print a human-readable summary of the site-road map."""
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"Total sites : {len(site_road_map)}")

    road_counts = Counter(len(s["roads"]) for s in site_road_map.values())
    count_str = ",  ".join(
        f"{v} site{'s' if v > 1 else ''} on {k} road{'s' if k > 1 else ''}"
        for k, v in sorted(road_counts.items())
    )
    print(f"Roads/site  : {count_str}")

    # Unusual approach counts
    unusual = {
        sid: len(s["approaches"])
        for sid, s in site_road_map.items()
        if len(s["approaches"]) not in (2, 4)
    }
    if unusual:
        print(f"\nUnusual approach counts (not 2 or 4):")
        for sid, cnt in sorted(unusual.items()):
            locs = [a["location"] for a in site_road_map[sid]["approaches"]]
            print(f"  Site {sid} — {cnt} approaches:")
            for loc in locs:
                print(f"    {loc}")
    else:
        print("\nAll sites have 2 or 4 approaches.")

    # All unique road names
    all_roads = sorted({
        road
        for s in site_road_map.values()
        for road in s["roads"]
    })
    print(f"\nUnique road names ({len(all_roads)} total):")
    for road in all_roads:
        print(f"  {road}")

    # Failed parses
    failed = [
        a["location"]
        for s in site_road_map.values()
        for a in s["approaches"]
        if a["sensor_road"] is None
    ]
    if failed:
        print(f"\nFailed to parse ({len(failed)}):")
        for loc in failed:
            print(f"  {loc!r}")
    else:
        print("\nAll 139 location strings parsed successfully.")

    print(sep)


# ── Serialisation helper ──────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE

    df       = load_scats_data(filepath)
    site_map = build_site_road_map(df)
    summarise(site_map)

    # Sample site for sanity check
    sample_id = "2000"
    if sample_id in site_map:
        sample = {**site_map[sample_id], "roads": sorted(site_map[sample_id]["roads"])}
        print(f"\nSample — site {sample_id}:")
        print(json.dumps(sample, indent=2, default=str))

    # Save JSON
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(site_map, f, indent=2, default=_json_default)
    print(f"\nSaved -> {OUT_FILE}")
