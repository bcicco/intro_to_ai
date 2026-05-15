"""
propose_edges.py — Step 2: propose candidate graph edges between SCATS sites.

Algorithm:
  - Build an inverse index: road → sites that have an approach on it
  - Exclude freeway/ramp/arterial roads
  - For each remaining road with 2+ sites, sort sites geographically and
    propose an edge between each consecutive pair
  - Compute haversine distance; drop edges > 5 km (likely missing intermediate site)
  - Deduplicate site pairs across roads (keep all contributing road names)
  - Save to proposed_edges.json
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SITES_FILE = Path(__file__).parent / "data" / "processed" / "boroondara_sites.json"
OUT_FILE   = Path(__file__).parent / "data" / "processed" / "proposed_edges.json"

MAX_DIST_KM = 5.0

_EXCLUDE_KEYWORDS = {"FWY", "FREEWAY", "RAMP", "OFFRAMP", "ONRAMP", "ARTERIAL"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def is_excluded(road: str) -> bool:
    upper = road.upper()
    return any(kw in upper for kw in _EXCLUDE_KEYWORDS)


# ── Core algorithm ────────────────────────────────────────────────────────────

def load_sites(filepath: str | Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def build_road_index(sites: dict) -> dict[str, list[str]]:
    """road → sorted list of site_ids that have an approach on it."""
    index: dict[str, list[str]] = defaultdict(list)
    for site_id, site in sites.items():
        for road in site["roads"]:
            index[road].append(site_id)
    return dict(index)


def sort_axis_for_road(site_ids: list[str], sites: dict) -> str:
    """
    Decide whether to sort sites along this road by latitude or longitude.
    If the latitude range is larger the road runs roughly N-S; otherwise E-W.
    """
    valid = [sites[sid] for sid in site_ids
             if sites[sid]["latitude"] is not None]
    if not valid:
        return "longitude"
    lats = [s["latitude"]  for s in valid]
    lons = [s["longitude"] for s in valid]
    return "latitude" if (max(lats) - min(lats)) > (max(lons) - min(lons)) else "longitude"


def propose_edges(sites: dict) -> list[dict]:
    """
    Return list of proposed edge dicts, deduplicated by site pair.
    Each dict: from_site, to_site, roads, distance_km, sort_axis.
    """
    road_index = build_road_index(sites)

    # Excluded roads
    excluded = sorted(r for r in road_index if is_excluded(r))
    print(f"\nExcluded roads ({len(excluded)}):")
    for r in excluded:
        print(f"  {r}")

    # edge_key (min_id, max_id) → edge dict (accumulates roads across iterations)
    edge_map: dict[tuple, dict] = {}
    skipped_distance: list[dict] = []

    for road, site_ids in sorted(road_index.items()):
        if is_excluded(road):
            continue
        if len(site_ids) < 2:
            continue

        axis = sort_axis_for_road(site_ids, sites)
        # Sites with None coordinates sink to the end
        ordered = sorted(
            site_ids,
            key=lambda sid: (
                sites[sid][axis] is None,
                sites[sid][axis] if sites[sid][axis] is not None else 0,
            ),
        )

        for a, b in zip(ordered, ordered[1:]):
            lat_a = sites[a]["latitude"]
            lon_a = sites[a]["longitude"]
            lat_b = sites[b]["latitude"]
            lon_b = sites[b]["longitude"]

            if lat_a is None or lat_b is None:
                continue

            dist = haversine(lat_a, lon_a, lat_b, lon_b)

            if dist > MAX_DIST_KM:
                skipped_distance.append({
                    "from": a, "to": b, "road": road,
                    "distance_km": round(dist, 3),
                })
                continue

            key = (min(a, b), max(a, b))
            if key in edge_map:
                if road not in edge_map[key]["roads"]:
                    edge_map[key]["roads"].append(road)
                    edge_map[key]["roads"].sort()
            else:
                edge_map[key] = {
                    "from":        a,
                    "to":          b,
                    "roads":       [road],
                    "distance_km": round(dist, 3),
                    "sort_axis":   axis,
                }

    edges = sorted(edge_map.values(), key=lambda e: (e["from"], e["to"]))

    print(f"\nEdges skipped (distance > {MAX_DIST_KM} km): {len(skipped_distance)}")
    for s in skipped_distance:
        print(f"  {s['from']} -- {s['to']}  via {s['road']}  {s['distance_km']:.2f} km")

    return edges


def summarise(edges: list[dict], sites: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"Proposed edges  : {len(edges)}")

    # Degree count
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]]   += 1
    isolated = [sid for sid in sites if degree[sid] == 0]
    print(f"Sites connected : {len(sites) - len(isolated)} / {len(sites)}")
    if isolated:
        print(f"Isolated sites  : {', '.join(sorted(isolated))}")

    print(f"\nEdge list (from -- to  via road(s)  dist km):")
    for e in edges:
        roads_str = ", ".join(e["roads"])
        print(f"  {e['from']} -- {e['to']}  [{roads_str}]  {e['distance_km']:.3f} km")

    print(f"\nDegree distribution:")
    deg_counts: dict[int, int] = defaultdict(int)
    for d in degree.values():
        deg_counts[d] += 1
    for d, cnt in sorted(deg_counts.items()):
        print(f"  degree {d}: {cnt} site(s)")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else SITES_FILE

    sites = load_sites(filepath)
    print(f"Loaded {len(sites)} sites from {filepath}")

    edges = propose_edges(sites)
    summarise(edges, sites)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(edges, f, indent=2)
    print(f"\nSaved -> {OUT_FILE}")
