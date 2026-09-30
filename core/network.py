"""Facility network: states -> districts -> tiered facilities (PHC -> CHC -> District Hospital).

Blood products (platelets) and ICU-style capacity live only at the District Hospital tier,
which mirrors how India's public health system is actually organised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- geography
# (district, state, lat, lon, population scale, has_blood_bank)
DISTRICTS = [
    ("Mumbai", "Maharashtra", 19.08, 72.88, 1.8),
    ("Pune", "Maharashtra", 18.52, 73.86, 1.4),
    ("Nashik", "Maharashtra", 20.00, 73.79, 1.0),
    ("Kolhapur", "Maharashtra", 16.70, 74.24, 0.8),
    ("Nagpur", "Maharashtra", 21.15, 79.09, 1.1),
    ("Bengaluru Urban", "Karnataka", 12.97, 77.59, 1.7),
    ("Mysuru", "Karnataka", 12.30, 76.64, 0.9),
    ("Belagavi", "Karnataka", 15.85, 74.50, 0.9),
    ("Kalaburagi", "Karnataka", 17.33, 76.83, 0.8),
    ("New Delhi", "Delhi", 28.61, 77.21, 1.0),
    ("North Delhi", "Delhi", 28.70, 77.20, 1.2),
    ("South Delhi", "Delhi", 28.52, 77.22, 1.2),
]

STATES = ["Maharashtra", "Karnataka", "Delhi"]

# Dengue season peak (day of year) and width per state -- deliberately different (non-IID)
STATE_SEASON = {
    "Maharashtra": (263, 32),  # ~20 Sep
    "Karnataka": (237, 40),    # ~25 Aug
    "Delhi": (278, 25),        # ~5 Oct, sharp
}

TIERS = {
    # tier: (count per district, base daily OPD footfall, beds, sanctioned staff, central resupply lead days)
    "DH": (1, 500, 200, 250, 7),
    "CHC": (1, 150, 30, 40, 10),
    "PHC": (3, 60, 6, 12, 12),
}
TIER_FEVER_SHARE = {"DH": 0.45, "CHC": 0.25, "PHC": 0.10}  # share of district fever patients per facility


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    unit: str
    tiers: tuple          # tiers that stock it
    criticality: float    # weight on unmet demand in the optimiser
    base_rate: float      # units per non-fever OPD patient
    fever_rate: float     # units per fever patient
    min_batch: int        # smallest transfer worth dispatching
    max_km: float         # max road distance for a transfer (shelf life / handling)
    weight: float         # relative handling/transport cost per unit


ITEMS = [
    Item("pcm", "Paracetamol 500mg", "strips", ("PHC", "CHC", "DH"), 1.0, 0.25, 1.5, 20, 700, 0.2),
    Item("ors", "ORS sachets", "sachets", ("PHC", "CHC", "DH"), 1.2, 0.08, 0.6, 20, 700, 0.2),
    Item("ivns", "IV Normal Saline 500ml", "bottles", ("PHC", "CHC", "DH"), 2.0, 0.02, 0.5, 10, 600, 1.0),
    Item("ns1", "Dengue NS1 rapid test kits", "kits", ("PHC", "CHC", "DH"), 1.5, 0.004, 0.9, 10, 700, 0.3),
    Item("plt", "Platelet units (RDP)", "units", ("DH",), 3.0, 0.002, 0.07, 2, 350, 2.0),
]
ITEM_IDS = [it.id for it in ITEMS]
ITEM_BY_ID = {it.id: it for it in ITEMS}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_network(seed: int = 7) -> pd.DataFrame:
    """One row per facility."""
    rng = np.random.default_rng(seed)
    rows = []
    for d_idx, (dist, state, lat, lon, scale) in enumerate(DISTRICTS):
        for tier, (count, footfall, beds, staff, lead) in TIERS.items():
            for k in range(count):
                if tier == "DH":
                    flat, flon = lat, lon
                    name = f"District Hospital {dist}"
                else:
                    flat = lat + rng.uniform(-0.25, 0.25)
                    flon = lon + rng.uniform(-0.25, 0.25)
                    name = f"{tier} {dist} {k + 1}" if count > 1 else f"{tier} {dist}"
                code = f"{tier}-{dist[:3].upper()}" + (f"-{k + 1}" if count > 1 else "")
                rows.append(
                    dict(
                        name=name,
                        code=code,
                        district=dist,
                        district_idx=d_idx,
                        state=state,
                        tier=tier,
                        lat=flat,
                        lon=flon,
                        base_footfall=footfall * scale * rng.uniform(0.85, 1.15),
                        beds=beds,
                        staff_sanctioned=staff,
                        lead_days=lead,
                    )
                )
    df = pd.DataFrame(rows)
    df.index.name = "fid"
    return df


def district_distance_matrix() -> np.ndarray:
    lat = np.array([d[2] for d in DISTRICTS])
    lon = np.array([d[3] for d in DISTRICTS])
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def facility_road_km(fac: pd.DataFrame) -> np.ndarray:
    """Straight-line distance x 1.3 road factor."""
    lat, lon = fac.lat.values, fac.lon.values
    return 1.3 * haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def transport_days(road_km: np.ndarray) -> np.ndarray:
    return np.where(road_km < 150, 1, np.where(road_km < 450, 2, 3))


def eligibility(fac: pd.DataFrame) -> np.ndarray:
    """elig[f, i] = facility f stocks item i."""
    return np.array([[fac.tier.iloc[f] in it.tiers for it in ITEMS] for f in range(len(fac))])
