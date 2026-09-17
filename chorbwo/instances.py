"""Problem instances for QoS-aware service choreography.

An instance consists of
  * a choreography specification: m roles connected by a message-flow DAG,
  * for each role, n candidate services,
  * for each candidate service, a 3-state operational DTMC (nominal / degraded / faulty)
    and state-conditional QoS (latency, reliability, throughput, energy) plus a static cost,
  * peer-to-peer link latencies between candidate services of connected roles,
  * a historical trace of length T_train per service (used to LEARN the models),
  * the observed initial state z0 of every candidate at the start of the session,
  * global QoS constraints and user weights.

Two generators are provided: ``make_synthetic`` (fully synthetic) and ``make_wsdream``
(real QoS levels and real service coordinates from WS-DREAM dataset 1, synthetic
state dynamics).  Both are deterministic functions of ``seed``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

METRICS = ["lat", "rel", "thr", "cost", "eng"]
MINIMIZE = {"lat": True, "rel": False, "thr": False, "cost": True, "eng": True}
L = 3  # operational states: 0 nominal, 1 degraded, 2 faulty
STATE_NAMES = ["nominal", "degraded", "faulty"]


@dataclass
class Instance:
    name: str
    seed: int
    m: int
    n: int
    H: int                       # session length (decision slots)
    edges: List[Tuple[int, int]]  # (pred, succ), roles are in topological order
    preds: List[List[int]]
    # true state-conditional QoS, shape (m, n, L)
    lat: np.ndarray
    rel: np.ndarray
    thr: np.ndarray
    eng: np.ndarray
    cost: np.ndarray             # (m, n) static
    P: np.ndarray                # true transition matrices (m, n, L, L)
    coords: np.ndarray           # (m, n, 2)
    link: Dict[Tuple[int, int], np.ndarray]      # edge -> (n, n) latency ms
    link_eng: Dict[Tuple[int, int], np.ndarray]  # edge -> (n, n) energy mJ
    z0: np.ndarray               # observed initial states (m, n)
    trace_z: np.ndarray          # historical states (m, n, T_train)
    trace_q: Dict[str, np.ndarray]  # historical noisy observations per metric (m, n, T_train)
    weights: Dict[str, float]
    thresholds: Dict[str, float] = field(default_factory=dict)  # lat<=, cost<=, rel>=, thr>=
    switch_ms: float = 40.0      # handshake latency paid when a role is rebound
    lat_cv: float = 0.25         # lognormal CV of realized latency
    regime: dict = None          # optional hidden global regime {"Q": (2,2), "P": (m,n,2,L,L)}

    @property
    def total_services(self) -> int:
        return self.m * self.n


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def layered_dag(m: int, rng: np.random.Generator) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Random layered DAG with a single initiator; roles are numbered in topological order."""
    layers: List[List[int]] = [[0]]
    r = 1
    while r < m:
        width = int(rng.integers(1, 4))
        width = min(width, m - r)
        layers.append(list(range(r, r + width)))
        r += width
    edges = []
    preds = [[] for _ in range(m)]
    for li in range(1, len(layers)):
        prev = layers[li - 1]
        for v in layers[li]:
            k = int(rng.integers(1, min(2, len(prev)) + 1))
            for p in rng.choice(prev, size=k, replace=False):
                edges.append((int(p), int(v)))
                preds[v].append(int(p))
        # every role of the previous layer must have at least one successor
        used = {p for (p, v) in edges if v in layers[li]}
        for p in prev:
            if p not in used:
                v = int(rng.choice(layers[li]))
                if p not in preds[v]:
                    edges.append((p, v))
                    preds[v].append(p)
    edges.sort(key=lambda e: (e[1], e[0]))
    return edges, preds


def stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a (..., L, L) row-stochastic matrix by power iteration."""
    pi = np.full(P.shape[:-1], 1.0 / P.shape[-1])
    for _ in range(300):
        pi = np.einsum("...i,...ij->...j", pi, P)
    return pi


def sample_transition_matrices(shape, rng: np.random.Generator) -> np.ndarray:
    """Sticky 3-state chains: nominal is the most persistent state, faulty the least."""
    P = np.zeros(shape + (L, L))
    stay = np.stack([rng.uniform(0.85, 0.98, shape),
                     rng.uniform(0.60, 0.90, shape),
                     rng.uniform(0.50, 0.85, shape)], axis=-1)
    # fraction of the leaving mass that goes to each other state
    P[..., 0, 0] = stay[..., 0]
    P[..., 0, 1] = (1 - stay[..., 0]) * rng.uniform(0.7, 0.9, shape)
    P[..., 0, 2] = 1 - P[..., 0, 0] - P[..., 0, 1]
    P[..., 1, 1] = stay[..., 1]
    P[..., 1, 0] = (1 - stay[..., 1]) * rng.uniform(0.5, 0.8, shape)
    P[..., 1, 2] = 1 - P[..., 1, 1] - P[..., 1, 0]
    P[..., 2, 2] = stay[..., 2]
    P[..., 2, 0] = (1 - stay[..., 2]) * rng.uniform(0.3, 0.7, shape)
    P[..., 2, 1] = 1 - P[..., 2, 2] - P[..., 2, 0]
    return P


def simulate_chain(P: np.ndarray, z_init: np.ndarray, T: int, rng: np.random.Generator,
                   regime: dict = None) -> np.ndarray:
    """Sample trajectories of independent chains; P (m,n,L,L), z_init (m,n) -> (m,n,T).
    With ``regime`` the transition matrix of every service is modulated by one hidden global
    2-state chain (services become correlated and non-Markov individually)."""
    m, n = z_init.shape
    out = np.empty((m, n, T), dtype=np.int8)
    z = z_init.copy()
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    if regime is None:
        cum = np.cumsum(P, axis=-1)
        for t in range(T):
            out[:, :, t] = z
            u = rng.random((m, n))[..., None]
            c = cum[ii, jj, z]                      # (m,n,L)
            z = np.minimum((u > c).sum(-1), L - 1).astype(np.int8)
        return out
    cumR = np.cumsum(regime["P"], axis=-1)          # (m,n,2,L,L)
    Q = np.cumsum(regime["Q"], axis=-1)
    g = int(rng.random() < 0.5)
    for t in range(T):
        out[:, :, t] = z
        u = rng.random((m, n))[..., None]
        c = cumR[ii, jj, g, z]
        z = np.minimum((u > c).sum(-1), L - 1).astype(np.int8)
        g = int(rng.random() > Q[g, 0])
    return out


def sample_initial_states(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pi = stationary(P)
    cum = np.cumsum(pi, axis=-1)
    u = rng.random(pi.shape[:-1])[..., None]
    return np.minimum((u > cum).sum(-1), L - 1).astype(np.int8)


def make_trace_observations(inst_arrays: Dict[str, np.ndarray], trace_z: np.ndarray,
                            rng: np.random.Generator, lat_cv: float) -> Dict[str, np.ndarray]:
    """Noisy per-slot observations of the QoS of every service along its historical trace."""
    m, n, T = trace_z.shape
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    obs = {}
    sig = np.sqrt(np.log(1 + lat_cv ** 2))
    for k in ["lat", "thr", "eng"]:
        mu = inst_arrays[k][ii[..., None], jj[..., None], trace_z]  # (m,n,T)
        if k == "lat" or k == "thr":
            obs[k] = mu * np.exp(rng.normal(-0.5 * sig ** 2, sig, mu.shape))
        else:
            obs[k] = mu * (1 + rng.normal(0, 0.1, mu.shape))
    p = inst_arrays["rel"][ii[..., None], jj[..., None], trace_z]
    obs["rel"] = (rng.random(p.shape) < p).astype(np.float64)
    return obs


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,2), b (k,2) in degrees -> (n,k) km."""
    R = 6371.0
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def _finish(name, seed, m, n, H, edges, preds, arrays, cost, P, coords, link, link_eng,
            weights, rng, T_train, lat_cv, switch_ms, regime=None) -> Instance:
    z_init = sample_initial_states(P, rng)
    trace_z = simulate_chain(P, z_init, T_train, rng, regime)
    trace_q = make_trace_observations(arrays, trace_z, rng, lat_cv)
    z0 = sample_initial_states(P, rng)
    inst = Instance(name=name, seed=seed, m=m, n=n, H=H, edges=edges, preds=preds,
                    lat=arrays["lat"], rel=arrays["rel"], thr=arrays["thr"], eng=arrays["eng"],
                    cost=cost, P=P, coords=coords, link=link, link_eng=link_eng, z0=z0,
                    trace_z=trace_z, trace_q=trace_q, weights=weights,
                    switch_ms=switch_ms, lat_cv=lat_cv, regime=regime)
    return inst


# --------------------------------------------------------------------------------------
# synthetic generator
# --------------------------------------------------------------------------------------

def make_synthetic(m: int, n: int, seed: int, H: int = 20, T_train: int = 400,
                   region_km: float = 1000.0, correlated: bool = False) -> Instance:
    rng = np.random.default_rng([seed, 7919])
    edges, preds = layered_dag(m, rng)
    shape = (m, n)
    lat0 = np.exp(rng.normal(np.log(150.0), 0.5, shape))
    mult_lat = np.stack([np.ones(shape), rng.uniform(1.5, 2.5, shape), rng.uniform(3.0, 5.0, shape)], -1)
    lat = lat0[..., None] * mult_lat
    rel = np.stack([rng.uniform(0.97, 0.999, shape), rng.uniform(0.85, 0.96, shape),
                    rng.uniform(0.30, 0.70, shape)], -1)
    thr0 = np.exp(rng.normal(np.log(50.0), 0.6, shape))
    thr = thr0[..., None] * np.stack([np.ones(shape), rng.uniform(0.4, 0.7, shape),
                                      rng.uniform(0.1, 0.3, shape)], -1)
    eng0 = rng.uniform(50.0, 200.0, shape)
    eng = eng0[..., None] * np.stack([np.ones(shape), rng.uniform(1.2, 1.6, shape),
                                      rng.uniform(1.6, 2.2, shape)], -1)
    cost = rng.uniform(1.0, 10.0, shape)
    P = sample_transition_matrices(shape, rng)
    coords = rng.uniform(0, region_km, (m, n, 2))
    link, link_eng = {}, {}
    for (p, v) in edges:
        d = np.linalg.norm(coords[p][:, None, :] - coords[v][None, :, :], axis=-1)
        link[(p, v)] = 3.0 + 0.015 * d
        link_eng[(p, v)] = 5.0 + 0.02 * d
    weights = {"lat": 0.30, "rel": 0.25, "thr": 0.15, "cost": 0.15, "eng": 0.15}
    arrays = {"lat": lat, "rel": rel, "thr": thr, "eng": eng}
    regime = None
    if correlated:
        # hidden global congestion regime: in the congested regime every service leaves the
        # nominal state 3x more often and recovers 2x less often (violates A1 and A2)
        Pc = P.copy()
        leave = 1 - Pc[..., 0, 0]
        Pc[..., 0, 0] = 1 - np.minimum(3 * leave, 0.6)
        Pc[..., 0, 1:] = Pc[..., 0, 1:] / np.maximum(leave[..., None], 1e-12) * (1 - Pc[..., 0, 0])[..., None]
        for s_ in (1, 2):
            rec = Pc[..., s_, 0] * 0.5
            Pc[..., s_, s_] += Pc[..., s_, 0] - rec
            Pc[..., s_, 0] = rec
        regime = {"Q": np.array([[0.9, 0.1], [0.8, 0.2]]), "P": np.stack([P, Pc], axis=2)}
    inst = _finish(f"syn_m{m}_n{n}", seed, m, n, H, edges, preds, arrays, cost, P, coords,
                   link, link_eng, weights, rng, T_train, 0.25, 40.0, regime)
    if correlated:
        inst.name = f"synreg_m{m}_n{n}"
    return inst


# --------------------------------------------------------------------------------------
# WS-DREAM (dataset 1) generator: real QoS levels + real coordinates, synthetic dynamics
# --------------------------------------------------------------------------------------

_WSD_CACHE = {}


def load_wsdream(data_dir: str) -> Dict[str, np.ndarray]:
    if "pool" in _WSD_CACHE:
        return _WSD_CACHE["pool"]
    rt = np.loadtxt(os.path.join(data_dir, "rtMatrix.txt"))
    tp = np.loadtxt(os.path.join(data_dir, "tpMatrix.txt"))
    with open(os.path.join(data_dir, "wslist.txt"), encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()[2:]
    coords = np.full((rt.shape[1], 2), np.nan)
    for ln in lines:
        p = ln.split("\t")
        try:
            coords[int(p[0])] = (float(p[7]), float(p[8]))
        except (ValueError, IndexError):
            pass
    valid = rt >= 0
    n_valid = valid.sum(0)
    keep = (n_valid >= 200) & np.isfinite(coords).all(1)
    idx = np.where(keep)[0]
    q_lat = np.zeros((len(idx), L)); q_thr = np.zeros((len(idx), L)); avail = np.zeros(len(idx))
    for a, s in enumerate(idx):
        r = rt[valid[:, s], s] * 1000.0  # seconds -> ms
        t = tp[valid[:, s], s]
        q_lat[a] = np.quantile(r, [0.25, 0.75, 0.95])
        q_thr[a] = np.quantile(t, [0.75, 0.25, 0.05])
        avail[a] = n_valid[s] / rt.shape[0]
    q_lat = np.maximum(q_lat, 1.0)
    q_thr = np.maximum(q_thr, 0.05)
    pool = {"ids": idx, "lat": q_lat, "thr": q_thr, "avail": avail, "coords": coords[idx]}
    _WSD_CACHE["pool"] = pool
    return pool


def make_wsdream(m: int, n: int, seed: int, data_dir: str, H: int = 20, T_train: int = 400) -> Instance:
    pool = load_wsdream(data_dir)
    rng = np.random.default_rng([seed, 104729])
    edges, preds = layered_dag(m, rng)
    shape = (m, n)
    pick = rng.choice(len(pool["ids"]), size=m * n, replace=False).reshape(m, n)
    lat = pool["lat"][pick]                     # (m,n,L) real quantiles
    thr = pool["thr"][pick]
    av = np.clip(pool["avail"][pick], 0.60, 0.999)
    rel = np.stack([av, av * rng.uniform(0.85, 0.95, shape), av * rng.uniform(0.3, 0.6, shape)], -1)
    eng = np.ones(shape + (L,))                 # not available in WS-DREAM (weight 0)
    cost = np.ones(shape)                       # not available in WS-DREAM (weight 0)
    P = sample_transition_matrices(shape, rng)
    coords = pool["coords"][pick]               # (m,n,2) lat/long degrees
    link, link_eng = {}, {}
    for (p, v) in edges:
        d = haversine_km(coords[p], coords[v])
        link[(p, v)] = 2.0 + 0.01 * d
        link_eng[(p, v)] = np.zeros_like(d)
    weights = {"lat": 0.40, "rel": 0.35, "thr": 0.25, "cost": 0.0, "eng": 0.0}
    arrays = {"lat": lat, "rel": rel, "thr": thr, "eng": eng}
    return _finish(f"wsd_m{m}_n{n}", seed, m, n, H, edges, preds, arrays, cost, P, coords,
                   link, link_eng, weights, rng, T_train, 0.25, 40.0)
