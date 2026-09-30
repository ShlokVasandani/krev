"""Synthetic but structured world: 2 years of daily history + 30 future days.

Design choices that matter for the demo's credibility:
* dengue has a real monsoon/post-monsoon season, with a different peak per state (non-IID)
* past outbreaks exist in history (so the model can learn how consumption responds to surveillance signals)
* surveillance case counts are reported with a 2-day delay and noise (like IDSP)
* random draws are made via fixed uniforms + inverse CDF, so the history *before* an outbreak
  is byte-identical across scenarios -> clean with/without comparisons
* Delhi only has 120 days of usable history -> the "data-poor state" for the federated cold-start story
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .network import (DISTRICTS, ITEMS, STATE_SEASON, STATES, TIER_FEVER_SHARE, build_network,
                      district_distance_matrix, eligibility)

HIST_DAYS = 730
FUT_DAYS = 30
TOTAL_DAYS = HIST_DAYS + FUT_DAYS
T_NOW = HIST_DAYS - 1                       # index of "today"
TODAY = pd.Timestamp("2026-09-30")
DATA_POOR_STATE = "Delhi"
DATA_POOR_HISTORY = 150
STOCK_ROLL_DAYS = 14                        # stock is initialised this many days before today, then consumed forward


@dataclass
class Scenario:
    epicentre: str | None = None            # district name, None = normal season
    days_since_onset: int = 9
    intensity: float = 4.0                  # peak extra cases, as a multiple of the district's peak-season level

    @property
    def label(self) -> str:
        if not self.epicentre:
            return "Normal season"
        return f"Dengue outbreak - {self.epicentre} (day {self.days_since_onset}, x{self.intensity:g})"


@dataclass
class World:
    fac: pd.DataFrame
    dates: pd.DatetimeIndex
    scenario: Scenario
    cases: np.ndarray            # (D, T) true dengue cases
    reported: np.ndarray         # (D, T) surveillance counts, 2-day lag
    footfall: np.ndarray         # (F, T)
    fever: np.ndarray            # (F, T) fever patients
    demand: np.ndarray           # (F, I, T)
    elig: np.ndarray             # (F, I)
    history_start: np.ndarray    # (F,) first usable history day per facility
    stock: np.ndarray            # (F, I) on hand today
    incoming: list               # [(arrival_day_index, f, i, qty)] pending central shipments
    near_expiry: np.ndarray      # (F, I) fraction of stock expiring within 30 days
    normal_rate: np.ndarray      # (F, I) pre-season typical daily demand (what a static policy uses)
    occupancy: np.ndarray        # (F, T) bed occupancy fraction
    staff_present: np.ndarray    # (F, T)
    unmet_before_now: np.ndarray = field(default=None)  # (F, I) unmet demand in the last 14 days
    alerts: list = field(default_factory=list)          # [(issue_day, district_idx)] IDSP outbreak alerts

    @property
    def F(self):
        return len(self.fac)

    @property
    def I(self):
        return len(ITEMS)


def outbreak_curve(tau: np.ndarray, tp: float = 10.0, k: float = 2.0) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    out = np.zeros_like(tau)
    m = tau >= 0
    x = tau[m] / tp
    out[m] = x ** k * np.exp(k * (1 - x))
    return out


def _season(doy: np.ndarray, peak: int, width: int) -> np.ndarray:
    d = np.abs(doy - peak)
    d = np.minimum(d, 365 - d)
    return 0.10 + np.exp(-0.5 * (d / width) ** 2)


def generate_world(scenario: Scenario | None = None, seed: int = 42) -> World:
    scenario = scenario or Scenario()
    rng = np.random.default_rng(seed)
    fac = build_network()
    F, D, I, T = len(fac), len(DISTRICTS), len(ITEMS), TOTAL_DAYS
    dates = pd.date_range(end=TODAY, periods=HIST_DAYS).append(
        pd.date_range(TODAY + pd.Timedelta(days=1), periods=FUT_DAYS))
    doy = dates.dayofyear.values
    t_idx = np.arange(T)
    dist = district_distance_matrix()
    d_state = np.array([d[1] for d in DISTRICTS])
    d_scale = np.array([d[4] for d in DISTRICTS])
    level = 22.0 * d_scale                                   # peak-season daily cases

    # ---- seasonal baseline
    lam = np.zeros((D, T))
    for d in range(D):
        pk, w = STATE_SEASON[d_state[d]]
        lam[d] = level[d] * _season(doy, pk, w)

    # ---- historic outbreaks (strictly in the past, spread by distance incl. across state borders)
    # Every district whose local wave is significant gets an IDSP-style outbreak alert ~3 days after its
    # local onset (time for lab confirmation / rapid-response-team investigation).
    alerts = []                                   # (issue_day, district)

    def add_outbreak(epi, onset, amp, tp, decay, speed):
        for d in range(D):
            if dist[epi, d] < 600:
                local_amp = amp * np.exp(-dist[epi, d] / decay)
                local_onset = onset + dist[epi, d] / speed
                lam[d] += level[d] * local_amp * outbreak_curve(t_idx - local_onset, tp=tp)
                if local_amp >= 1.0:
                    alerts.append((int(np.ceil(local_onset + 3)), d))

    for s in STATES:
        s_d = np.where(d_state == s)[0]
        season_days = np.where((lam[s_d[0]] > 0.35 * level[s_d[0]]) & (t_idx < T_NOW - 75))[0]
        for _ in range(rng.integers(5, 8)):
            onset = rng.choice(season_days)
            epi = rng.choice(s_d)
            add_outbreak(epi, onset, rng.uniform(2.0, 7.0), rng.uniform(9, 14), 280, 32)

    # ---- current scenario overlay
    if scenario.epicentre:
        epi = [d[0] for d in DISTRICTS].index(scenario.epicentre)
        add_outbreak(epi, T_NOW - scenario.days_since_onset, scenario.intensity, 11, 280, 32)

    u_cases = rng.random((D, T))
    cases = poisson.ppf(u_cases, lam).astype(float)
    rep_noise = rng.lognormal(0, 0.15, (D, T))
    reported = np.zeros_like(cases)
    reported[:, 2:] = np.round(cases[:, :-2] * rep_noise[:, 2:])

    # ---- facilities: footfall, fever patients, demand
    dow = dates.dayofweek.values
    weekly = np.array([1.15, 1.05, 1.0, 1.0, 1.05, 0.95, 0.55])[dow]
    annual = 1 + 0.08 * np.sin(2 * np.pi * doy / 365)
    fd = fac.district_idx.values
    share = fac.tier.map(TIER_FEVER_SHARE).values
    nonfever = fac.base_footfall.values[:, None] * weekly * annual * rng.lognormal(0, 0.07, (F, T))
    fever = cases[fd] * 3.0 * share[:, None]
    footfall = nonfever + fever

    elig = eligibility(fac)
    state_mult = {s: rng.uniform(0.75, 1.35, I) for s in STATES}          # prescribing practice differs by state
    sm = np.stack([state_mult[s] for s in fac.state])                   # (F, I)
    tier_iv = fac.tier.map({"PHC": 0.5, "CHC": 1.0, "DH": 1.5}).values
    base_rate = np.array([it.base_rate for it in ITEMS])
    fever_rate = np.array([it.fever_rate for it in ITEMS])
    mean = nonfever[:, None, :] * base_rate[None, :, None] + fever[:, None, :] * fever_rate[None, :, None]
    ivi = [it.id for it in ITEMS].index("ivns")
    mean[:, ivi, :] *= tier_iv[:, None]
    mean *= sm[:, :, None] * elig[:, :, None]
    gam = rng.gamma(25, 1 / 25, (F, I, T))
    u_dem = rng.random((F, I, T))
    demand = poisson.ppf(u_dem, np.maximum(mean * gam, 1e-9)).astype(float)

    # ---- operations snapshot
    t0 = T_NOW - STOCK_ROLL_DAYS + 1
    normal_rate = demand[:, :, t0 - 28:t0].mean(axis=2)
    cover = rng.uniform(32, 50, (F, I))
    over = rng.random(F) < 0.22
    cover[over] = rng.uniform(50, 80, (over.sum(), I))
    stock = np.round(normal_rate * cover) * elig
    unmet = np.zeros((F, I))
    for t in range(t0, T_NOW + 1):
        served = np.minimum(stock, demand[:, :, t])
        unmet += demand[:, :, t] - served
        stock -= served

    incoming = []
    for f in range(F):
        for i in range(I):
            if elig[f, i] and rng.random() < 0.3:
                incoming.append((T_NOW + int(rng.integers(2, 10)), f, i, float(np.round(normal_rate[f, i] * rng.uniform(10, 20)))))
    near_expiry = rng.uniform(0, 0.4, (F, I)) * elig
    near_expiry[:, [it.id for it in ITEMS].index("plt")] = 0.6 * elig[:, -1]   # 5-day shelf life

    adm_rate = fac.tier.map({"PHC": 0.03, "CHC": 0.08, "DH": 0.15}).values
    adm = fever * adm_rate[:, None] + nonfever * 0.01
    los = 4
    occ_raw = np.stack([np.convolve(a, np.ones(los), mode="full")[:T] for a in adm])
    occupancy = np.clip(0.40 + occ_raw / fac.beds.values[:, None], 0, 1.4)
    staff_present = rng.binomial(fac.staff_sanctioned.values[:, None], 0.86, (F, T))

    history_start = np.where(fac.state.values == DATA_POOR_STATE, T_NOW - DATA_POOR_HISTORY + 1, 0)

    return World(fac=fac, dates=dates, scenario=scenario, cases=cases, reported=reported, footfall=footfall,
                 fever=fever, demand=demand, elig=elig, history_start=history_start, stock=stock,
                 incoming=incoming, near_expiry=near_expiry, normal_rate=normal_rate, occupancy=occupancy,
                 staff_present=staff_present, unmet_before_now=unmet,
                 alerts=sorted(alerts))


def outbreak_field(scenario: Scenario, lat, lon, day_since_onset: float) -> np.ndarray:
    """Excess dengue intensity (multiple of a district's peak-season level) of the scenario outbreak at any point.
    Same spread law the simulator used to generate the monitored districts: amplitude decays with distance,
    arrival is delayed by distance (about 32 km/day), each local wave follows the outbreak curve."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if not scenario.epicentre:
        return np.zeros_like(lat)
    from .network import haversine_km
    e = DISTRICTS[[d[0] for d in DISTRICTS].index(scenario.epicentre)]
    d = haversine_km(e[2], e[3], lat, lon)
    return scenario.intensity * np.exp(-d / 280) * outbreak_curve(day_since_onset - d / 32, tp=11) * (d < 600)
