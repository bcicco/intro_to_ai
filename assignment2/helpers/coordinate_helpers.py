import math

_REF_LAT = -37.85
_REF_LON = 145.05
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON = 111.320 * math.cos(math.radians(_REF_LAT))  # roughly 87.7
_COORD_SCALE = 60  # units per km  ->  1 unit = 1 second at 60 km/h


def latlon_to_xy(lat: float, lon: float) -> tuple[int, int]:
    x = int((lon - _REF_LON) * _KM_PER_DEG_LON * _COORD_SCALE)
    y = int((lat - _REF_LAT) * _KM_PER_DEG_LAT * _COORD_SCALE)
    return x, y


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _compass(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> str:
    dlat, dlon = lat_b - lat_a, lon_b - lon_a
    return (
        ("N" if dlat > 0 else "S")
        if abs(dlat) >= abs(dlon)
        else ("E" if dlon > 0 else "W")
    )
