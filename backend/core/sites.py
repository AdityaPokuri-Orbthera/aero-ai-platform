from __future__ import annotations
import json, math, os
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sites")

def _load_json(name: str) -> List[Dict[str, Any]]:
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

LAUNCH_SITES = _load_json("launch_sites.json")
LUNAR_SITES  = _load_json("lunar_sites.json")

# very small state centroid map (expand as needed)
STATE_CENTROIDS = {
    "louisiana": (30.98, -91.96),
    "la": (30.98, -91.96),
    "texas": (31.0, -100.0),
    "florida": (27.8, -81.7),
    "virginia": (37.4, -78.4)
}

CITY_HINTS = {
    "new orleans": (29.95, -90.07),
    "baton rouge": (30.45, -91.15),
    "shreveport": (32.52, -93.75),
    "houston": (29.76, -95.37),
    "brownsville": (25.90, -97.50),
    "cape canaveral": (28.39, -80.60),
    "wallops": (37.84, -75.49)
}

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    from math import radians, sin, cos, sqrt, asin
    R = 6371.0
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2-lat1; dlon = lon2-lon1
    x = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2*R*asin(sqrt(x))

def infer_origin_coords(origin_hint: Optional[str]) -> Optional[Tuple[float,float,str]]:
    if not origin_hint:
        return None
    h = origin_hint.strip().lower()
    # exact city hits
    for k, v in CITY_HINTS.items():
        if k in h:
            return (v[0], v[1], k)
    # state-level hits
    for k, v in STATE_CENTROIDS.items():
        if k in h.split():
            return (v[0], v[1], k)
    # fallback: try loose contains
    for k, v in STATE_CENTROIDS.items():
        if k in h:
            return (v[0], v[1], k)
    return None

def score_site_for_mission(site: Dict[str,Any], target: str, payload_leo_kg: Optional[float]) -> float:
    """
    Very simple scoring:
      + latitude bonus for low-lat if target != polar
      + prefer vertical pads for LEO/TLI
      + slight bonus for FAA-licensed
    """
    lat = abs(float(site.get("lat_deg", 0)))
    low_lat_bonus = max(0.0, 30.0 - lat) / 30.0  # 0..1 (best near equator)
    vertical = 1.0 if site.get("type") == "vertical" else 0.0
    licensed = 1.0 if site.get("faa_licensed") else 0.0
    tli_bonus = 0.2 if target.upper() == "TLI" else 0.0
    return 0.6*low_lat_bonus + 0.3*vertical + 0.1*licensed + tli_bonus

def nearest_and_best_sites(origin: Optional[Tuple[float,float,str]], target: str, payload_leo_kg: Optional[float]) -> List[Dict[str,Any]]:
    rows = []
    for s in LAUNCH_SITES:
        row = dict(s)
        row["score"] = score_site_for_mission(s, target, payload_leo_kg)
        if origin:
            row["distance_km"] = round(haversine_km((origin[0], origin[1]), (s["lat_deg"], s["lon_deg"])))
        else:
            row["distance_km"] = None
        rows.append(row)
    # sort: by vertical-only first for LEO/TLI, then score desc, then distance asc
    rows.sort(key=lambda r: (
        0 if r["type"]=="vertical" else 1,
        -r["score"],
        r["distance_km"] if r["distance_km"] is not None else 1e9
    ))
    return rows

def pick_lunar_sites(limit: int = 3) -> List[Dict[str,Any]]:
    # simple static preference order for MVP
    return LUNAR_SITES[:limit]
