"""Federated training (FedAvg) of the shared outbreak-trajectory model across state nodes.

Each state trains on its own districts' data and sends only a signed weight *update*; raw surveillance and
patient records never leave the state. Feature-scaling statistics are also computed federatedly
(each node shares only counts, sums and sums of squares).

Model: linear term + one small ReLU hidden layer (captures the non-linear early phase of an outbreak).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .security import KeyRegistry, clip_and_noise, sign_update, verify_update

HIDDEN = 0          # 0 = linear (chosen: as accurate as a 16-unit MLP here, stabler and explainable)


def _unpack(w, K, H):
    i = 0
    wl = w[i:i + K]; i += K
    W1 = w[i:i + K * H].reshape(K, H); i += K * H
    b1 = w[i:i + H]; i += H
    w2 = w[i:i + H]; i += H
    b2 = w[i]
    return wl, W1, b1, w2, b2


def n_params(K, H):
    return K + K * H + 2 * H + 1


def _forward(w, Z, H):
    wl, W1, b1, w2, b2 = _unpack(w, Z.shape[1], H)
    A = np.maximum(Z @ W1 + b1, 0)
    return Z @ wl + A @ w2 + b2, A


def _grad(w, Z, y, H, l2):
    K = Z.shape[1]
    wl, W1, b1, w2, b2 = _unpack(w, K, H)
    pred, A = _forward(w, Z, H)
    r = pred - y
    d = r / len(y)
    dA = d[:, None] * w2[None, :] * (A > 0)
    g = np.concatenate([Z.T @ d + l2 * wl, (Z.T @ dA + l2 * W1).ravel(), dA.sum(0), A.T @ d + l2 * w2, [d.sum()]])
    return g, float(np.mean(r ** 2))


@dataclass
class ForecastModel:
    w: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sigma: np.ndarray        # residual std (log space) per group
    hidden: int = HIDDEN

    def growth(self, X: np.ndarray) -> np.ndarray:
        shp = X.shape[:-1]
        Z = ((X - self.mean) / self.std).reshape(-1, X.shape[-1])
        return np.clip(_forward(self.w, Z, self.hidden)[0], -3, 3).reshape(shp)


@dataclass
class Client:
    name: str
    X: np.ndarray
    y: np.ndarray
    item: np.ndarray = None

    def __post_init__(self):
        if self.item is None:
            self.item = np.zeros(len(self.y), int)

    @property
    def n(self):
        return len(self.y)


@dataclass
class FedLog:
    rounds: list = field(default_factory=list)   # dicts: round, state, loss, accepted, reason
    security: list = field(default_factory=list)


def federated_scaler(clients):
    n = sum(c.n for c in clients)
    s = sum(c.X.sum(0) for c in clients)
    ss = sum((c.X ** 2).sum(0) for c in clients)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    std[std < 1e-6] = 1.0
    return mean, std


def _local_train(w, Z, y, steps, lr, l2, H):
    """Full-batch Adam on regularised squared loss (log space)."""
    w = w.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for k in range(1, steps + 1):
        g, _ = _grad(w, Z, y, H, l2)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
    loss = float(np.mean((_forward(w, Z, H)[0] - y) ** 2))
    return w, loss


def fedavg(clients: list, rounds: int = 30, local_steps: int = 25, lr: float = 0.03, l2: float = 1e-3,
           registry: KeyRegistry | None = None, dp_clip: float = 0.0, dp_noise: float = 0.0,
           attack: dict | None = None, seed: int = 0, n_items: int = 1, on_round=None, hidden: int = HIDDEN):
    """attack = {"state": name, "round": r, "kind": "tamper" | "poison"} for the security demo."""
    rng = np.random.default_rng(seed)
    mean, std = federated_scaler(clients)
    K = clients[0].X.shape[1]
    H = hidden
    w = np.zeros(n_params(K, H))
    w[K:K + K * H] = np.random.default_rng(1234).normal(0, 0.3 / np.sqrt(K), K * H)
    w[K + K * H + H:K + K * H + 2 * H] = np.random.default_rng(99).normal(0, 0.1, H)
    log = FedLog()
    registry = registry or KeyRegistry()
    for c in clients:
        if c.name not in registry.keys:
            registry.provision(c.name)
    Zs = {c.name: (c.X - mean) / std for c in clients}

    for r in range(1, rounds + 1):
        updates = []
        for c in clients:
            w_loc, loss = _local_train(w, Zs[c.name], c.y, local_steps, lr, l2, H)
            delta = w_loc - w
            if dp_clip > 0:
                delta = clip_and_noise(delta, dp_clip, dp_noise, rng)
            sig = sign_update(registry.keys[c.name], r, c.name, delta)
            if attack and attack["state"] == c.name and attack["round"] == r:
                if attack["kind"] == "tamper":      # modified in transit, after signing
                    delta = delta.copy()
                    delta += rng.normal(0, 0.5, delta.shape)
                elif attack["kind"] == "poison":    # compromised node signs a malicious update
                    delta = -40 * delta
                    sig = sign_update(registry.keys[c.name], r, c.name, delta)
            updates.append((c, delta, sig, loss))

        norms = np.array([np.linalg.norm(u[1]) for u in updates])
        med = np.median(norms)
        accepted = []
        for (c, delta, sig, loss), nrm in zip(updates, norms):
            ok, reason = True, "signature verified"
            if not verify_update(registry.keys[c.name], r, c.name, delta, sig):
                ok, reason = False, "signature mismatch - update altered in transit"
            elif len(updates) >= 3 and nrm > 4 * med + 1e-9:
                ok, reason = False, f"anomalous update norm ({nrm:.2f} vs median {med:.2f}) - possible poisoning"
            log.rounds.append(dict(round=r, state=c.name, loss=loss, accepted=ok, reason=reason))
            if not ok:
                log.security.append(dict(round=r, state=c.name, event=reason))
                registry.rotate(c.name, "security event")
                log.security.append(dict(round=r, state=c.name, event="signing key revoked and rotated"))
            else:
                accepted.append((c.n, delta))
        if accepted:
            tot = sum(n for n, _ in accepted)
            w = w + sum(n * d for n, d in accepted) / tot
        if on_round is not None:
            on_round(r, ForecastModel(w=w.copy(), mean=mean, std=std, sigma=None, hidden=H))

    model = ForecastModel(w=w, mean=mean, std=std, sigma=None, hidden=H)
    # residual uncertainty per group, aggregated from per-node sums of squares
    ss = np.zeros(n_items)
    cnt = np.zeros(n_items)
    for c in clients:
        res = model.growth(c.X) - c.y
        ss += np.bincount(c.item, res ** 2, minlength=n_items)
        cnt += np.bincount(c.item, minlength=n_items)
    model.sigma = np.sqrt(ss / np.maximum(cnt, 1))
    return model, log


def wape(model: ForecastModel, X, y, base):
    pred = np.expm1(base + model.growth(X))
    act = np.expm1(base + y)
    return float(np.abs(pred - act).sum() / max(act.sum(), 1e-9))


def naive_wape(y, base):
    """Persistence baseline: next 14 days look like the last 7."""
    pred = np.expm1(base)
    act = np.expm1(base + y)
    return float(np.abs(pred - act).sum() / max(act.sum(), 1e-9))
