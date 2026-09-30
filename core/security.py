"""Security for the federated channel -- proportionate, standard, and relevant to health data.

* every state node signs its model update (HMAC-SHA256 over round id + state id + weights)
* the aggregator rejects unsigned/tampered updates and statistically anomalous ones (poisoning guard)
* optional update clipping + Gaussian noise (differential-privacy style) so individual rows can't be reverse-engineered
* transport would be mutual TLS 1.3 (per-session ephemeral keys); certificates are short-lived and rotated on
  schedule AND immediately on a security event. We deliberately don't hand-roll crypto.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class KeyRegistry:
    """Per-node signing keys, issued at provisioning. Rotated on schedule or on a security event."""
    keys: dict = field(default_factory=dict)
    issued: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    ttl: timedelta = timedelta(hours=24)

    def provision(self, node: str, now: datetime | None = None):
        now = now or datetime.now()
        self.keys[node] = secrets.token_bytes(32)
        self.issued[node] = now
        self.events.append((now, node, "key issued"))

    def rotate(self, node: str, reason: str, now: datetime | None = None):
        now = now or datetime.now()
        self.keys[node] = secrets.token_bytes(32)
        self.issued[node] = now
        self.events.append((now, node, f"key rotated ({reason})"))

    def due_for_rotation(self, node: str, now: datetime) -> bool:
        return now - self.issued[node] >= self.ttl


def _msg(round_id: int, node: str, weights: np.ndarray) -> bytes:
    return f"{round_id}|{node}|".encode() + np.ascontiguousarray(weights, dtype=np.float64).tobytes()


def sign_update(key: bytes, round_id: int, node: str, weights: np.ndarray) -> str:
    return hmac.new(key, _msg(round_id, node, weights), hashlib.sha256).hexdigest()


def verify_update(key: bytes, round_id: int, node: str, weights: np.ndarray, signature: str) -> bool:
    return hmac.compare_digest(sign_update(key, round_id, node, weights), signature)


def clip_and_noise(delta: np.ndarray, clip: float, noise_mult: float, rng: np.random.Generator) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if clip > 0 and norm > clip:
        delta = delta * (clip / norm)
    if noise_mult > 0:
        delta = delta + rng.normal(0, noise_mult * clip, delta.shape)
    return delta
