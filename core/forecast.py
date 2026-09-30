"""Two-stage probabilistic forecaster with lead-time-aware stock-out risk.

Stage 1 - outbreak trajectory (federated): each state trains on its own districts' surveillance curves; the
          shared model predicts the next 14 days of dengue cases per district, including districts that
          are about to be hit (neighbour pressure features).
Stage 2 - demand response (local): each facility estimates, from its own history, a baseline daily demand
          per item plus how much extra each dengue case adds. These coefficients never leave the facility.

Forecast demand = baseline + response x forecast cases. Uncertainty combines demand noise with outbreak
uncertainty in proportion to how dengue-driven the item is at that facility. Risk is measured against each
facility's resupply lead time: a stock-out in 9 days is a non-event if resupply takes 3, a crisis if 12.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import HORIZON, LABEL_DELAY, case_features, case_rows
from .federated import Client, ForecastModel, fedavg, naive_wape, wape
from .network import DISTRICTS, ITEMS, STATES, district_distance_matrix
from .security import KeyRegistry
from .simdata import T_NOW, World

Z90 = 1.2816
FED_ROUNDS, FED_STEPS, FED_LR, FED_L2 = 40, 40, 0.05, 3e-3
FIT_WINDOW = 365


def _state_districts(s):
    return [k for k, d in enumerate(DISTRICTS) if d[1] == s]


def state_clients(world: World, dist, t_min, t_max):
    out = []
    for s in STATES:
        X, y, _ = case_rows(world, dist, t_min, t_max, _state_districts(s))
        out.append(Client(s, X, y))
    return out


# ------------------------------------------------------------------ stage 2: local demand response
def _case_proxy(world: World) -> np.ndarray:
    """cases on day t, known from surveillance 2 days later. NaN where not yet reported."""
    c = np.full(world.reported.shape, np.nan)
    c[:, :-LABEL_DELAY] = world.reported[:, LABEL_DELAY:]
    c[:, T_NOW - LABEL_DELAY + 1:] = np.nan
    return c


def fit_response(world: World, t_end: int = T_NOW):
    """demand[f,i,t] ~ a + b * cases[district(f), t]  (b >= 0), fit on each facility's own last year."""
    c = _case_proxy(world)
    last = t_end - LABEL_DELAY
    t0 = last - FIT_WINDOW + 1
    x = c[world.fac.district_idx.values, t0:last + 1]                     # (F, n)
    m = (np.arange(t0, last + 1)[None, :] >= world.history_start[:, None]).astype(float)
    y = world.demand[:, :, t0:last + 1]                                  # (F, I, n)
    n = m.sum(1)
    mx = (x * m).sum(1) / n
    my = (y * m[:, None]).sum(2) / n[:, None]
    dx = (x - mx[:, None]) * m
    b = ((y - my[:, :, None]) * dx[:, None]).sum(2) / np.maximum((dx ** 2).sum(1), 1e-9)[:, None]
    b = np.maximum(b, 0) * world.elig
    return b


@dataclass
class DemandModel:
    case_model: ForecastModel
    b: np.ndarray            # (F, I) extra units per dengue case
    s_base: np.ndarray       # (I,) demand noise, log space
    fed_log: object = None
    rows_per_state: dict = None

    @property
    def sigma_case(self):
        return float(self.case_model.sigma[0])


@dataclass
class Forecast:
    t: int
    p50: np.ndarray            # (F, I) median mean-daily demand, next 14 days
    p90: np.ndarray
    mu: np.ndarray             # log1p(p50)
    sig: np.ndarray            # (F, I) log-space std
    stockout_prob: np.ndarray  # (F, I) within resupply lead time
    days_cover: np.ndarray
    incoming_in_lead: np.ndarray
    lead: np.ndarray           # (F,)
    dengue_share: np.ndarray   # (F, I) share of forecast demand that is dengue-driven
    cases_now: np.ndarray      # (D,) reported 7-day mean
    cases_fut: np.ndarray      # (D,) forecast next-14-day mean true cases
    recent_rate: np.ndarray    # (F, I) last-7-day mean demand


def _demand_forecast(model: DemandModel, world: World, t: int, cases_fut: np.ndarray):
    fd = world.fac.district_idx.values
    rcs = np.concatenate([np.zeros((world.reported.shape[0], 1)), np.cumsum(world.reported, 1)], 1)
    c28 = (rcs[:, t + 1] - rcs[:, t + 1 - 28]) / 28                 # = cases over [t-29, t-2]
    d28 = world.demand[:, :, t - 29:t - 1].mean(2)
    base = np.maximum(d28 - model.b * c28[fd][:, None], 0.2 * d28)
    add = model.b * cases_fut[fd][:, None]
    p50 = (base + add) * world.elig
    share = np.where(p50 > 0, add / np.maximum(p50, 1e-9), 0)
    sig = np.sqrt(model.s_base[None, :] ** 2 + (share * model.sigma_case) ** 2)
    return p50, share, sig


def forecast_cases(model: DemandModel, world: World, t: int):
    X, base, _ = case_features(world, np.array([t]), district_distance_matrix())
    g = model.case_model.growth(X)[:, 0]
    return np.expm1(base[:, 0]), np.expm1(base[:, 0] + g)


def train_model(world: World, registry: KeyRegistry | None = None, attack=None, dp_clip=0.0, dp_noise=0.0):
    dist = district_distance_matrix()
    clients = state_clients(world, dist, 0, T_NOW)
    case_model, log = fedavg(clients, rounds=FED_ROUNDS, local_steps=FED_STEPS, lr=FED_LR, l2=FED_L2,
                             registry=registry, attack=attack, dp_clip=dp_clip, dp_noise=dp_noise)
    b = fit_response(world)
    # demand-noise sigma: backtest the response model with *realised* cases over the past year
    model = DemandModel(case_model, b, np.full(len(ITEMS), 0.1), log, {c.name: c.n for c in clients})
    res = [[] for _ in ITEMS]
    rcs = np.concatenate([np.zeros((world.reported.shape[0], 1)), np.cumsum(world.reported, 1)], 1)
    for t in range(T_NOW - 360, T_NOW - HORIZON - LABEL_DELAY, 5):
        s, e = t + 1 + LABEL_DELAY, t + HORIZON + LABEL_DELAY
        c_real = (rcs[:, e + 1] - rcs[:, s]) / HORIZON
        p50, _, _ = _demand_forecast(model, world, t, c_real)
        act = world.demand[:, :, t + 1:t + 1 + HORIZON].mean(2)
        ok = world.elig & (world.history_start[:, None] <= t - 30)
        r = np.log1p(act) - np.log1p(p50)
        for i in range(len(ITEMS)):
            res[i].append(r[ok[:, i], i])
    model.s_base = np.array([np.std(np.concatenate(r)) for r in res])
    return model


def forecast_at(model: DemandModel, world: World, t: int, stock: np.ndarray, incoming: list) -> Forecast:
    cases_now, cases_fut = forecast_cases(model, world, t)
    p50, share, sig = _demand_forecast(model, world, t, cases_fut)
    mu = np.log1p(p50)
    p90 = np.expm1(mu + Z90 * sig) * world.elig
    lead = world.fac.lead_days.values.astype(float)
    inc = np.zeros_like(stock)
    for p in incoming:
        day, f, i, q = p[0], p[1], p[2], p[3]
        if t < day <= t + lead[f]:
            inc[f, i] += q
    s_eff = stock + inc
    z = (np.log1p(s_eff / lead[:, None]) - mu) / np.maximum(sig, 1e-6)
    prob = (1 - norm.cdf(z)) * world.elig
    cover = np.where(p50 > 0, stock / np.maximum(p50, 1e-9), np.inf)
    recent = world.demand[:, :, t - 6:t + 1].mean(2)
    return Forecast(t, p50, p90, mu, sig, prob, cover, inc, lead, share, cases_now, cases_fut, recent)


def risk_level(p):
    return np.where(p >= 0.5, "HIGH", np.where(p >= 0.2, "MEDIUM", "LOW"))


def forecast_table(world: World, fc: Forecast, stock: np.ndarray) -> pd.DataFrame:
    fac = world.fac
    F, I = stock.shape
    ff, ii = np.where(world.elig)
    df = pd.DataFrame(dict(
        fid=ff, facility=fac.name.values[ff], tier=fac.tier.values[ff], district=fac.district.values[ff],
        state=fac.state.values[ff], item=[ITEMS[i].name for i in ii], item_id=[ITEMS[i].id for i in ii],
        unit=[ITEMS[i].unit for i in ii], stock=stock[ff, ii], incoming=fc.incoming_in_lead[ff, ii],
        last7=fc.recent_rate[ff, ii], p50=fc.p50[ff, ii], p90=fc.p90[ff, ii], days_cover=fc.days_cover[ff, ii],
        lead_days=fc.lead[ff], dengue_share=fc.dengue_share[ff, ii], stockout_prob=fc.stockout_prob[ff, ii]))
    df["risk"] = risk_level(df.stockout_prob.values)
    return df


# ------------------------------------------------------------------ benchmark for the federated story
COLD_START_DAYS = 150


def federated_benchmark(world: World, focus_state: str = "Delhi"):
    """Cold-start test, repeated for each state: the state joins with only its last 150 days of data.
    Every model is then scored on that state's *earlier* two seasons (simulated, withheld from training),
    which contain full dengue seasons and past outbreaks it has never seen locally."""
    import copy
    w = copy.copy(world)
    w.history_start = np.zeros_like(world.history_start)      # the simulator knows the full hidden history
    dist = district_distance_matrix()
    cut = T_NOW - COLD_START_DAYS
    kw = dict(rounds=FED_ROUNDS, local_steps=FED_STEPS, lr=FED_LR, l2=FED_L2)
    rows, curve = [], []
    for s in STATES:
        clients = []
        for s2 in STATES:
            X, y, _ = case_rows(w, dist, cut if s2 == s else 0, T_NOW, _state_districts(s2))
            clients.append(Client(s2, X, y))
        Xt, yt, bt = case_rows(w, dist, 0, cut - HORIZON - LABEL_DELAY - 1, _state_districts(s))
        track = (lambda r, m: curve.append((r, wape(m, Xt, yt, bt)))) if s == focus_state else None
        fed, _ = fedavg(clients, on_round=track, **kw)
        loc, _ = fedavg([c for c in clients if c.name == s], **kw)
        pooled = Client("pooled", np.vstack([c.X for c in clients]), np.concatenate([c.y for c in clients]))
        cen, _ = fedavg([pooled], **kw)
        rows.append(dict(state=s, local_rows=[c.n for c in clients if c.name == s][0], test_rows=len(yt),
                         persistence=naive_wape(yt, bt), local_only=wape(loc, Xt, yt, bt),
                         federated=wape(fed, Xt, yt, bt), centralized=wape(cen, Xt, yt, bt)))
    return pd.DataFrame(rows), pd.DataFrame(curve, columns=["round", "wape"])
