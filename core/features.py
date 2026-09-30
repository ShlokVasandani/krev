"""District-level features for the outbreak-trajectory (case) model.

Target: log growth of the next 14 days' dengue cases vs the last 7 days of *reported* surveillance counts.
Inputs combine the district's own curve (level, growth, acceleration) with *neighbouring* districts'
pressure -- which is what lets us warn a district before its own case count moves.
"""
from __future__ import annotations

import numpy as np

from .network import DISTRICTS
from .simdata import T_NOW, World

HORIZON = 14
LABEL_DELAY = 2      # reported[t] == cases[t-2]
MIN_LAG = 35
FEATURES = ["cl7", "cgrowth", "clevel", "caccel", "clatest", "slope5", "nb7", "nbgrowth", "nbrel", "sin", "cos",
            "cgrowth2", "cgrowth_x_clevel", "nbgrowth_x_nbrel", "cgrowth_x_nbrel",
            "alert", "alert_age", "alert_age2", "nb_alert", "nb_alert_age", "nb_alert_age2"]
ALERT_WINDOW = 35


def _cs(a):
    z = np.zeros(a.shape[:-1] + (1,))
    return np.concatenate([z, np.cumsum(a, axis=-1)], axis=-1)


def _wmean(cs, t, w):
    return (cs[..., t + 1] - cs[..., t + 1 - w]) / w


def neighbour_weights(dist_km: np.ndarray) -> np.ndarray:
    w = np.exp(-dist_km / 250.0) * (dist_km < 600)
    np.fill_diagonal(w, 0.0)
    return w


def case_features(world: World, ts: np.ndarray, dist_km: np.ndarray, with_target=False):
    """X (D, n, K), base = log1p(reported 7-day mean) (D, n), y (D, n) or None."""
    ts = np.asarray(ts)
    rcs = _cs(world.reported)
    r3, r7, r28 = _wmean(rcs, ts, 3), _wmean(rcs, ts, 7), _wmean(rcs, ts, 28)
    r7p = _wmean(rcs, ts - 7, 7)
    W = neighbour_weights(dist_km)
    nb3, nb7, nb7p = W @ r3, W @ r7, W @ r7p
    cl3, cl7, cl28, cl7p = np.log1p(r3), np.log1p(r7), np.log1p(r28), np.log1p(r7p)
    doy = world.dates.dayofyear.values[ts]
    D, n = r7.shape
    f = {
        "cl7": cl7, "cgrowth": cl3 - cl7, "clevel": cl7 - cl28, "caccel": (cl3 - cl7) - (cl7 - cl7p),
        "nb7": np.log1p(nb7), "nbgrowth": np.log1p(nb3) - np.log1p(nb7p), "nbrel": np.log1p(nb7) - cl7,
        "sin": np.broadcast_to(np.sin(2 * np.pi * doy / 365), (D, n)),
        "cos": np.broadcast_to(np.cos(2 * np.pi * doy / 365), (D, n)),
    }
    f["clatest"] = np.log1p(_wmean(rcs, ts, 2)) - cl7
    lr5 = np.stack([np.log1p(world.reported[:, ts - k]) for k in range(4, -1, -1)], -1)   # (D, n, 5)
    xs = np.arange(5) - 2.0
    f["slope5"] = (lr5 * xs).sum(-1) / (xs ** 2).sum()
    f["cgrowth2"] = f["cgrowth"] ** 2
    f["cgrowth_x_clevel"] = f["cgrowth"] * f["clevel"]
    f["nbgrowth_x_nbrel"] = f["nbgrowth"] * np.maximum(f["nbrel"], 0)
    f["cgrowth_x_nbrel"] = f["cgrowth"] * np.maximum(f["nbrel"], 0)
    f.update(alert_features(world, ts, dist_km))
    X = np.stack([np.asarray(f[k], float) for k in FEATURES], axis=-1)
    if not with_target:
        return X, cl7, None
    # label = mean reported over t+1+delay .. t+14+delay  (i.e. true cases t+1..t+14)
    s, e = ts + 1 + LABEL_DELAY, ts + HORIZON + LABEL_DELAY
    fut = (rcs[..., e + 1] - rcs[..., s]) / HORIZON
    return X, cl7, np.log1p(fut) - cl7


def alert_features(world: World, ts: np.ndarray, dist_km: np.ndarray):
    """IDSP outbreak alerts known at day t: own district, and the strongest nearby alert."""
    D, n = len(DISTRICTS), len(ts)
    age = np.full((D, n), np.inf)                      # days since most recent own alert
    for day, d in world.alerts:
        a = ts - day
        ok = (a >= 0) & (a < ALERT_WINDOW)
        age[d] = np.where(ok, np.minimum(age[d], a), age[d])
    active = np.isfinite(age)
    a14 = np.where(active, age / 14.0, 0.0)
    W = np.exp(-dist_km / 250.0) * (dist_km < 600)
    np.fill_diagonal(W, 0.0)
    strength = W[:, :, None] * active[None, :, :]      # (D, D', n)
    best = strength.argmax(1)                          # strongest neighbouring alert
    nb = strength.max(1)
    nb_age = np.take_along_axis(a14[None].repeat(D, 0), best[:, None, :], 1)[:, 0, :] * (nb > 0)
    return {"alert": active.astype(float), "alert_age": a14, "alert_age2": a14 ** 2,
            "nb_alert": nb, "nb_alert_age": nb_age, "nb_alert_age2": nb_age ** 2}


def district_history_start(world: World) -> np.ndarray:
    hs = np.zeros(len(DISTRICTS), int)
    for f in range(world.F):
        d = world.fac.district_idx.iloc[f]
        hs[d] = max(hs[d], world.history_start[f])
    return hs


def case_rows(world: World, dist_km, t_min, t_max, districts):
    t_max = min(t_max, T_NOW - HORIZON - LABEL_DELAY)
    ts = np.arange(max(t_min, MIN_LAG), t_max + 1)
    X, base, y = case_features(world, ts, dist_km, with_target=True)
    hs = district_history_start(world)
    mask = ts[None, :] >= hs[:, None] + MIN_LAG
    dm = np.zeros(len(DISTRICTS), bool)
    dm[districts] = True
    mask &= dm[:, None]
    return X[mask], y[mask], base[mask]
