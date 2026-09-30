"""Cross-facility redistribution as a min-cost transportation problem (linear programme, HiGHS via SciPy).

Donor surplus  = on-hand + incoming - median demand over (lead time + 7-day safety)   -> never creates a new shortage
Receiver need  = P90 demand over the lead time - (on-hand + incoming)                  -> conservative on the at-risk side
Cost per unit  = road km x item handling weight, discounted for near-expiry donor stock (move it before it expires)
               + penalty for inter-state transfers (needs approval from two state directorates)
Unmet need is allowed but penalised by item criticality, so life-critical items are prioritised.
Constraints    = item-specific max distance (platelets: 350 km, blood-bank tier only), minimum dispatch batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

from .forecast import Forecast
from .network import ITEMS, facility_road_km, transport_days
from .simdata import World

INTERSTATE_PENALTY = 300.0      # in km-equivalents
SAFETY_DAYS = 7
RECEIVER_MIN_PROB = 0.2
DONOR_MAX_PROB = 0.05


def recommend_transfers(world: World, fc: Forecast, stock: np.ndarray, road_km: np.ndarray | None = None,
                        max_km_scale: float = 1.0) -> pd.DataFrame:
    fac = world.fac
    road_km = facility_road_km(fac) if road_km is None else road_km
    state = fac.state.values
    L = fc.lead
    rows = []
    for i, it in enumerate(ITEMS):
        elig = world.elig[:, i]
        pos = stock[:, i] + fc.incoming_in_lead[:, i]
        need = np.maximum(0, fc.p90[:, i] * L - pos)
        need[(fc.stockout_prob[:, i] < RECEIVER_MIN_PROB) | ~elig] = 0
        surplus = np.maximum(0, pos - fc.p50[:, i] * (L + SAFETY_DAYS))
        # never rely on a pending shipment: what stays on the shelf must cover the donor's own resupply window
        surplus = np.minimum(surplus, np.maximum(0, stock[:, i] - fc.p50[:, i] * L))
        surplus[(fc.stockout_prob[:, i] > DONOR_MAX_PROB) | ~elig] = 0
        R = np.where(need >= it.min_batch)[0]
        Dn = np.where(surplus >= it.min_batch)[0]
        if len(R) == 0 or len(Dn) == 0:
            continue
        arcs = [(k, j) for k in Dn for j in R if k != j and road_km[k, j] <= it.max_km * max_km_scale]
        if not arcs:
            continue
        nA, nR = len(arcs), len(R)
        rpos = {j: n for n, j in enumerate(R)}
        dpos = {k: n for n, k in enumerate(Dn)}
        c = np.empty(nA + nR)
        for a, (k, j) in enumerate(arcs):
            c[a] = road_km[k, j] * it.weight * (1 - 0.3 * world.near_expiry[k, i]) + \
                (INTERSTATE_PENALTY * it.weight if state[k] != state[j] else 0.0) + 5.0
        c[nA:] = 5000.0 * it.criticality
        A = np.zeros((len(Dn) + nR, nA + nR))
        b = np.zeros(len(Dn) + nR)
        for a, (k, j) in enumerate(arcs):
            A[dpos[k], a] = 1
            A[len(Dn) + rpos[j], a] = -1
        for n, j in enumerate(R):
            A[len(Dn) + n, nA + n] = -1
            b[len(Dn) + n] = -need[j]
        b[:len(Dn)] = surplus[Dn]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        if res.status != 0:
            continue
        x = np.floor(res.x[:nA] + 1e-6)
        for a, (k, j) in enumerate(arcs):
            q = x[a]
            if q < it.min_batch:
                continue
            rows.append(dict(item_id=it.id, item=it.name, unit=it.unit, i=i, from_fid=k, to_fid=j, qty=int(q),
                             road_km=float(road_km[k, j]), days=int(transport_days(road_km[k, j])),
                             interstate=bool(state[k] != state[j])))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _explain(world, fc, stock, df)


def _explain(world: World, fc: Forecast, stock: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    fac = world.fac
    recv_add = df.groupby(["to_fid", "i"]).qty.sum()
    don_sub = df.groupby(["from_fid", "i"]).qty.sum()
    out = []
    for r in df.itertuples():
        j, k, i = r.to_fid, r.from_fid, r.i
        L = fc.lead[j]
        pos = stock[j, i] + fc.incoming_in_lead[j, i]
        mu, s = fc.mu[j, i], fc.sig[j, i]
        before = fc.stockout_prob[j, i]
        after = 1 - norm.cdf((np.log1p((pos + recv_add[(j, i)]) / L) - mu) / s)
        donor_cover = (stock[k, i] - don_sub[(k, i)]) / max(fc.p50[k, i], 1e-9)
        share = fc.dengue_share[j, i]
        text = (f"{fac.name.iloc[j]} has {stock[j, i]:.0f} {r.unit} of {r.item}; forecast demand is "
                f"{fc.p50[j, i]:.0f}/day (P90 {fc.p90[j, i]:.0f}, {share:.0%} dengue-driven, recent {fc.recent_rate[j, i]:.0f}/day). "
                f"Stock-out probability before its {L:.0f}-day resupply: {before:.0%}. "
                f"{fac.name.iloc[k]} can release {r.qty} and still keep {donor_cover:.0f} days of cover. "
                f"Arrives in ~{r.days} day(s) ({r.road_km:.0f} km). Risk after all transfers: {after:.0%}.")
        if r.interstate:
            text += " Inter-state: needs sign-off from both state drug stores."
        out.append(dict(**r._asdict(), frm=fac.name.iloc[k], to=fac.name.iloc[j],
                        from_state=fac.state.iloc[k], to_state=fac.state.iloc[j],
                        prob_before=before, prob_after=after, donor_cover_after=donor_cover, why=text))
    res = pd.DataFrame(out).drop(columns=["Index"])
    crit = {it.id: it.criticality for it in ITEMS}
    res["priority"] = res.item_id.map(crit) * res.prob_before
    return res.sort_values("priority", ascending=False).reset_index(drop=True)


def escalations(world: World, fc: Forecast, stock: np.ndarray, transfers: pd.DataFrame) -> pd.DataFrame:
    """Need that redistribution cannot cover -> raise emergency procurement to the state warehouse."""
    recv = np.zeros_like(stock)
    if len(transfers):
        for r in transfers.itertuples():
            recv[r.to_fid, r.i] += r.qty
    pos = stock + fc.incoming_in_lead + recv
    short = np.maximum(0, fc.p90 * fc.lead[:, None] - pos)
    short[fc.stockout_prob < RECEIVER_MIN_PROB] = 0
    fac = world.fac
    rows = []
    for f, i in zip(*np.where(short >= np.array([it.min_batch for it in ITEMS])[None, :])):
        rows.append(dict(state=fac.state.iloc[f], district=fac.district.iloc[f], facility=fac.name.iloc[f],
                         item=ITEMS[i].name, unit=ITEMS[i].unit, shortfall=float(short[f, i])))
    df = pd.DataFrame(rows, columns=["state", "district", "facility", "item", "unit", "shortfall"])
    if df.empty:
        return df
    return (df.groupby(["state", "district", "item", "unit"], as_index=False)
              .agg(shortfall=("shortfall", "sum"), facilities=("facility", "count"))
              .sort_values("shortfall", ascending=False))
