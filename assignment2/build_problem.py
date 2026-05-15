import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from assignment1.models import Edge, Node, Problem

from assignment2.helpers.coordinate_helpers import latlon_to_xy, _compass, haversine_km
from assignment2.helpers.cost_functions import quadratic_cost, bpr_cost
from assignment2.helpers.data_lookup_helpers import _approach_volume

COST_FUNCTIONS: dict[str, str] = {
    "quadratic": "quadratic",
    "bpr": "bpr",
}


def build_problem(
    origin_id: str,
    dest_ids: list[str],
    sites: dict,
    edges: list[dict],
    volume_map: dict[str, float],
    cost_fn: str = "quadratic",
) -> Problem:
    """
    Convert the SCATS graph + GRU volume predictions into an assignment-1
    Problem.  Each undirected edge is split into two directed edges; costs
    are weighted by the predicted volume on the relevant approach.

    cost_fn: "quadratic" (default, seconds via assignment flow-speed model)
             "bpr"       (BPR travel-time in coord units)
    """
    if cost_fn not in COST_FUNCTIONS:
        raise ValueError(
            f"Unknown cost_fn '{cost_fn}'. Choose from: {', '.join(COST_FUNCTIONS)}"
        )

    # Directed cost lookup
    directed: dict[tuple[str, str], float] = {}

    for e in edges:
        a_id, b_id = e["from"], e["to"]
        if a_id not in sites or b_id not in sites:
            continue
        sa, sb = sites[a_id], sites[b_id]
        if sa["latitude"] is None or sb["latitude"] is None:
            continue

        dist = haversine_km(
            sa["latitude"], sa["longitude"], sb["latitude"], sb["longitude"]
        )
        road = e["roads"][0] if e["roads"] else None

        dir_ab = _compass(
            sa["latitude"], sa["longitude"], sb["latitude"], sb["longitude"]
        )
        vol_ab = _approach_volume(sb, b_id, road, dir_ab, volume_map) if road else None

        dir_ba = _compass(
            sb["latitude"], sb["longitude"], sa["latitude"], sa["longitude"]
        )
        vol_ba = _approach_volume(sa, a_id, road, dir_ba, volume_map) if road else None

        if cost_fn == "quadratic":
            # GRU predicts veh/15min; quadratic model expects veh/hr
            flow_ab = (vol_ab or 0.0) * 4
            flow_ba = (vol_ba or 0.0) * 4
            directed[(a_id, b_id)] = quadratic_cost(dist, flow_ab)
            directed[(b_id, a_id)] = quadratic_cost(dist, flow_ba)
        else:
            directed[(a_id, b_id)] = bpr_cost(dist, vol_ab or 0.0)
            directed[(b_id, a_id)] = bpr_cost(dist, vol_ba or 0.0)

    # ── Adjacency list ────────────────────────────────────────────────────────
    adjacency: dict[str, list[tuple[str, float]]] = {sid: [] for sid in sites}
    for (a, b), cost in directed.items():
        adjacency[a].append((b, cost))

    # ── Node list ─────────────────────────────────────────────────────────────
    node_list: list[Node] = []
    for site_id, site in sorted(sites.items()):
        lat, lon = site["latitude"], site["longitude"]
        if lat is None or lon is None:
            continue
        nid = int(site_id)
        node_edges = [
            Edge(start_node_id=nid, end_node_id=int(nb), cost=round(cost, 2))
            for nb, cost in sorted(adjacency[site_id], key=lambda t: int(t[0]))
        ]
        node_list.append(
            Node(id=nid, coordinates=latlon_to_xy(lat, lon), edges=node_edges)
        )

    return Problem(
        nodes=node_list,
        origin=int(origin_id),
        destinations=[int(d) for d in dest_ids],
    )
