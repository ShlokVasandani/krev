"""District boundaries for the outbreak-spread map (simplified from GADM via github.com/geohacker/india).

Only districts around the monitored network are drawn. Boundaries are for visualisation; an official deployment
would use Survey of India boundaries.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .network import DISTRICTS, haversine_km

PATH = Path(__file__).resolve().parent.parent / "data" / "india_districts.geojson"
EXCLUDE_STATES = {"Jammu and Kashmir", "Ladakh"}


@lru_cache(maxsize=1)
def load_districts():
    fc = json.loads(PATH.read_text())
    south = [d for d in DISTRICTS if d[1] != "Delhi"]
    delhi = [d for d in DISTRICTS if d[1] == "Delhi"]
    feats = []
    for f in fc["features"]:
        p = f["properties"]
        if p["state"] in EXCLUDE_STATES:
            continue
        near_s = min(haversine_km(d[2], d[3], p["lat"], p["lon"]) for d in south)
        near_d = min(haversine_km(d[2], d[3], p["lat"], p["lon"]) for d in delhi)
        if near_s > 430 and near_d > 240:
            continue
        mon = p["monitored"]
        if mon == "Delhi":
            mon_names = [d[0] for d in delhi]
            lat, lon = delhi[0][2], delhi[0][3]
        elif mon:
            d = next(x for x in DISTRICTS if x[0] == mon)
            mon_names, lat, lon = [mon], d[2], d[3]
        else:
            mon_names, lat, lon = [], p["lat"], p["lon"]
        feats.append(dict(geometry=f["geometry"], name=p["name"], state=p["state"], monitored=mon_names,
                          lat=lat, lon=lon))
    lat = np.array([f["lat"] for f in feats])
    lon = np.array([f["lon"] for f in feats])
    return feats, lat, lon
