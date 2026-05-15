import numpy as np


def _approach_volume(
    site: dict,
    site_id: str,
    road: str,
    travel_dir: str,
    volume_map: dict[str, float],
) -> float | None:
    """
    Predicted volume for the most relevant approach at `site` on `road`
    in direction `travel_dir`.  Falls back progressively:
      1. sensor_road == road  AND  travel_direction == travel_dir  (best match)
      2. Any approach on road (sensor or cross)
      3. Mean of all predicted approaches at the site
    """
    # Wide-parquet uses non-zero-padded SCATS numbers ("970" not "0970")
    prefix = str(int(site_id))

    def col(approach: dict) -> str:
        return f"{prefix}__{approach['location']}"

    for a in site["approaches"]:
        if a["sensor_road"] == road and a["travel_direction"] == travel_dir:
            v = volume_map.get(col(a))
            if v is not None:
                return v

    for a in site["approaches"]:
        if road in (a["sensor_road"], a["cross_road"]):
            v = volume_map.get(col(a))
            if v is not None:
                return v

    vals = [volume_map[col(a)] for a in site["approaches"] if col(a) in volume_map]
    return float(np.mean(vals)) if vals else None
