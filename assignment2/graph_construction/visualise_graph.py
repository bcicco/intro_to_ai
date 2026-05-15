"""
visualise_graph.py — render the Boroondara SCATS graph on an interactive map.

Inputs:
  data/processed/boroondara_sites.json
  data/processed/proposed_edges.json

Output:
  data/processed/boroondara_graph.html
"""

import json
import sys
from pathlib import Path

import folium

SITES_FILE = Path(__file__).parent / "data" / "processed" / "boroondara_sites.json"
EDGES_FILE = Path(__file__).parent / "data" / "processed" / "proposed_edges.json"
OUT_FILE   = Path(__file__).parent / "data" / "processed" / "boroondara_graph.html"


def load_data(sites_path: Path, edges_path: Path) -> tuple[dict, list]:
    with open(sites_path) as f:
        sites = json.load(f)
    with open(edges_path) as f:
        edges = json.load(f)
    return sites, edges


def build_map(sites: dict, edges: list) -> folium.Map:
    # ── Centre map on mean of valid site coordinates ──────────────────────────
    valid_coords = [
        (s["latitude"], s["longitude"])
        for s in sites.values()
        if s["latitude"] is not None and s["longitude"] is not None
    ]
    centre_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
    centre_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13,
                   tiles="CartoDB positron")

    # ── Determine which sites have at least one edge ──────────────────────────
    connected: set[str] = set()
    for e in edges:
        connected.add(e["from"])
        connected.add(e["to"])

    # ── Edges ─────────────────────────────────────────────────────────────────
    edges_plotted = 0
    for e in edges:
        a, b = e["from"], e["to"]
        if a not in sites or b not in sites:
            continue
        lat_a, lon_a = sites[a]["latitude"], sites[a]["longitude"]
        lat_b, lon_b = sites[b]["latitude"], sites[b]["longitude"]
        if None in (lat_a, lon_a, lat_b, lon_b):
            continue

        roads_str = ", ".join(e["roads"]) if isinstance(e["roads"], list) else e["roads"]
        tip = f"{a} → {b} ({e['distance_km']:.2f} km via {roads_str})"

        folium.PolyLine(
            locations=[[lat_a, lon_a], [lat_b, lon_b]],
            color="grey",
            weight=2,
            opacity=0.6,
            tooltip=tip,
        ).add_to(m)
        edges_plotted += 1

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes_plotted = 0
    isolated_sites: list[dict] = []

    for site_id, site in sorted(sites.items()):
        lat, lon = site["latitude"], site["longitude"]
        if lat is None or lon is None:
            print(f"WARNING: site {site_id} has no coordinates — skipped")
            continue

        if site_id not in connected:
            isolated_sites.append(site)
            continue

        roads_str = ", ".join(sorted(site["roads"]))
        n_approaches = len(site["approaches"])
        colour = "blue"
        radius = 6

        popup_html = (
            f"<b>Site {site_id}</b><br>"
            f"Roads: {roads_str}<br>"
            f"Approaches: {n_approaches}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            opacity=0.8,
            tooltip=site_id,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)

        nodes_plotted += 1

    # ── Title + legend ────────────────────────────────────────────────────────
    title_html = f"""
    <div style="
        position: fixed; top: 12px; left: 56px; z-index: 1000;
        background: white; padding: 8px 14px; border-radius: 6px;
        border: 1px solid #bbb; font-family: sans-serif; font-size: 14px;
        box-shadow: 2px 2px 6px rgba(0,0,0,.2);
    ">
        <b>Boroondara SCATS Graph</b> &mdash; {nodes_plotted} sites, {edges_plotted} edges
    </div>
    """
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 56px; z-index: 1000;
        background: white; padding: 8px 14px; border-radius: 6px;
        border: 1px solid #bbb; font-family: sans-serif; font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,.2);
    ">
        <span style="color:blue; font-size:16px;">&#9679;</span> SCATS site &nbsp;&nbsp;
        <span style="color:grey; font-size:16px;">&#9135;</span> Road segment
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, nodes_plotted, edges_plotted, isolated_sites


if __name__ == "__main__":
    sites_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SITES_FILE
    edges_path = Path(sys.argv[2]) if len(sys.argv) > 2 else EDGES_FILE

    sites, edges = load_data(sites_path, edges_path)
    m, nodes_plotted, edges_plotted, isolated = build_map(sites, edges)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_FILE))

    print(f"Nodes plotted : {nodes_plotted}")
    print(f"Edges plotted : {edges_plotted}")
    print(f"\nIsolated sites ({len(isolated)}):")
    for s in isolated:
        roads_str = ", ".join(sorted(s["roads"]))
        print(f"  {s['site_id']}  roads: {roads_str}")
    print(f"\nSaved -> {OUT_FILE}")
