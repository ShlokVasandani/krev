"""Counterfactual impact: replay the next N days of the *same* true demand under three policies.

static        - today's practice: reorder from the state warehouse when stock falls below a fixed reorder point
forecast      - reorder points driven by the AI forecast (P90), refreshed daily
forecast+redistribution - the above, plus optimised inter-facility transfers every `review_every` days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import forecast_at
from .network import ITEMS, facility_road_km
from .optimizer import recommend_transfers
from .simdata import T_NOW, World

POLICIES = ["static", "forecast", "forecast+redistribution"]
POLICY_LABEL = {"static": "Without our system (static reorder points)",
                "forecast": "AI forecast only",
                "forecast+redistribution": "With our system (forecast + redistribution)"}


def simulate(world: World, model, policy: str, horizon: int = 21, review_every: int = 2):
    F, I = world.F, world.I
    stock = world.stock.copy()
    pipeline = list(world.incoming)
    L = world.fac.lead_days.values
    crit = np.array([it.criticality for it in ITEMS])
    road = facility_road_km(world.fac)
    static_rop = world.normal_rate * (L[:, None] + 7)
    static_upto = world.normal_rate * (L[:, None] + 21)
    daily, transfers_log = [], []
    unmet_tot = np.zeros((F, I))
    so_days = np.zeros((F, I))
    for step in range(horizon):
        t = T_NOW + step            # decision made at end of day t, demand of day t+1 then realised
        # ---- decisions
        if policy == "static":
            rop, upto = static_rop, static_upto
        else:
            fc = forecast_at(model, world, t, stock, pipeline)
            rop = fc.p90 * (L[:, None] + 7)
            upto = fc.p90 * (L[:, None] + 21)
            if policy == "forecast+redistribution" and step % review_every == 0:
                tr = recommend_transfers(world, fc, stock, road)
                for r in tr.itertuples() if not tr.empty else []:
                    q = min(r.qty, stock[r.from_fid, r.i])
                    stock[r.from_fid, r.i] -= q
                    pipeline.append((t + r.days, r.to_fid, r.i, float(q), "transfer"))
                    transfers_log.append(dict(day=step, item=r.item, frm=r.frm, to=r.to, qty=q))
        position = stock.copy()
        for p in pipeline:
            if p[0] > t:
                position[p[1], p[2]] += p[3]
        order = (position < rop) & world.elig
        for f, i in zip(*np.where(order)):
            pipeline.append((t + L[f], f, i, float(np.ceil(upto[f, i] - position[f, i])), "central"))
        # ---- next day
        tn = t + 1
        for p in [p for p in pipeline if p[0] == tn]:
            stock[p[1], p[2]] += p[3]
        pipeline = [p for p in pipeline if p[0] > tn]
        d = world.demand[:, :, tn]
        served = np.minimum(stock, d)
        unmet = d - served
        stock -= served
        unmet_tot += unmet
        so = (unmet > 0) & world.elig
        so_days += so
        daily.append(dict(day=step + 1, date=world.dates[tn], stockouts=int(so.sum()),
                          unmet_weighted=float((unmet * crit).sum()), unmet_units=float(unmet.sum())))
    df = pd.DataFrame(daily)
    df["cum_unmet_weighted"] = df.unmet_weighted.cumsum()
    per_item = pd.DataFrame({"item": [it.name for it in ITEMS], "unmet_units": unmet_tot.sum(0),
                             "stockout_facility_days": so_days.sum(0)})
    return dict(daily=df, per_item=per_item, unmet=unmet_tot, so_days=so_days,
                transfers=pd.DataFrame(transfers_log), policy=policy)


def compare(world: World, model, horizon: int = 21):
    return {p: simulate(world, model, p, horizon) for p in POLICIES}
