import math

_QF_A = 1.4648375
_QF_B = 93.75
_QF_MAX_FLOW = 1500.0  # avoid negative discriminant
_COORD_SCALE = 60  # units per km  ->  1 unit = 1 second at 60 km/h
CAPACITY = 450


def flow_to_speed(flow_per_hour: float, congested: bool = False) -> float:
    """
    Invert the quadratic flow-speed model to get speed (km/h).

    congested=False returns the free-flow (+ root, higher speed) branch.
    congested=True  returns the over-capacity (- root, lower speed) branch.
    Result is capped at 60 km/h.
    """
    if flow_per_hour <= 0:
        return 60.0
    flow = min(flow_per_hour, _QF_MAX_FLOW)
    disc = _QF_B**2 - 4 * _QF_A * flow
    if disc < 0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    v = (
        (_QF_B + sqrt_disc) / (2 * _QF_A)
        if not congested
        else (_QF_B - sqrt_disc) / (2 * _QF_A)
    )
    return min(v, 60.0)


def quadratic_cost(
    distance_km: float,
    flow_per_hour: float,
    intersection_delay_s: float = 30.0,
) -> float:
    """
    Return edge cost in seconds using the assignment's quadratic flow-speed model.

    cost = (distance_km / speed_km_h) * 3600  +  intersection_delay_s

    At free-flow (60 km/h) this equals distance_km * 60, matching the
    Euclidean heuristic (_COORD_SCALE = 60), so the heuristic is admissible.
    """
    speed = flow_to_speed(flow_per_hour)
    return (distance_km / speed) * 3600.0 + intersection_delay_s


def bpr_cost(distance_km: float, volume: float) -> float:
    """
    same scale as quadratic: 1 unit ≈ 1 s).

    cost = distance_coord_units * (1 + 0.15 * (v / capacity)^4)

    At zero traffic this equals distance_km * _COORD_SCALE = free-flow seconds,
    so the Euclidean heuristic remains admissible.
    """
    factor = 1.0 + 0.15 * (volume / CAPACITY) ** 4
    return distance_km * _COORD_SCALE * factor
