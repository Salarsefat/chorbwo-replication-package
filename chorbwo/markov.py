"""Learning the per-service operational-state DTMC and state-conditional QoS from traces,
and turning them into the expected QoS used by the planner.

Three estimators of the session QoS of candidate (r, j) are compared in the paper:
  * ``static``   : historical time-average of the trace (what QoS-aware composition normally uses),
  * ``last``     : QoS of the currently observed state z0, assumed to persist ("last observed"),
  * ``markov``   : session-averaged expectation under the learned DTMC started from z0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .instances import Instance, L, METRICS


@dataclass
class LearnedModel:
    P_hat: np.ndarray            # (m, n, L, L)
    q_state: Dict[str, np.ndarray]   # metric -> (m, n, L) estimated state-conditional mean
    q_mean: Dict[str, np.ndarray]    # metric -> (m, n) historical average
    omega: np.ndarray            # (m, n, L, L) session occupancy: (1/H) sum_{h<H} P_hat^h


def learn_model(inst: Instance, alpha: float = 0.5) -> LearnedModel:
    z = inst.trace_z.astype(int)
    m, n, T = z.shape
    counts = np.full((m, n, L, L), alpha)
    for t in range(T - 1):
        np.add.at(counts, (np.repeat(np.arange(m), n), np.tile(np.arange(n), m),
                           z[:, :, t].ravel(), z[:, :, t + 1].ravel()), 1.0)
    P_hat = counts / counts.sum(-1, keepdims=True)
    q_state, q_mean = {}, {}
    for k in ["lat", "rel", "thr", "eng"]:
        obs = inst.trace_q[k]
        q_mean[k] = obs.mean(-1)
        qs = np.zeros((m, n, L))
        for s in range(L):
            mask = (z == s)
            cnt = mask.sum(-1)
            tot = (obs * mask).sum(-1)
            qs[:, :, s] = np.where(cnt > 0, tot / np.maximum(cnt, 1), q_mean[k])
        q_state[k] = qs
    omega = np.zeros_like(P_hat)
    Pk = np.broadcast_to(np.eye(L), P_hat.shape).copy()
    for _ in range(inst.H):
        omega += Pk
        Pk = Pk @ P_hat
    omega /= inst.H
    return LearnedModel(P_hat=P_hat, q_state=q_state, q_mean=q_mean, omega=omega)


def occupancy(P: np.ndarray, H: int, gamma: float = 1.0) -> np.ndarray:
    """Discounted occupancy  Omega = sum_{h<H} w_h P^h  with w_h proportional to gamma^h
    (gamma = 1 gives the uniform session average, gamma -> 0 the current state)."""
    w = gamma ** np.arange(H)
    w = w / w.sum()
    om = np.zeros_like(P)
    Pk = np.broadcast_to(np.eye(P.shape[-1]), P.shape).copy()
    for h in range(H):
        om += w[h] * Pk
        Pk = Pk @ P
    return om


def qos_model(inst: Instance, lm: LearnedModel, kind: str, gamma: float = 1.0) -> Dict[str, np.ndarray]:
    """Expected session QoS of every candidate under the requested estimator."""
    m, n = inst.m, inst.n
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    out = {"cost": inst.cost}
    if kind == "static":
        for k in ["lat", "rel", "thr", "eng"]:
            out[k] = lm.q_mean[k]
    elif kind == "last":
        for k in ["lat", "rel", "thr", "eng"]:
            out[k] = lm.q_state[k][ii, jj, inst.z0.astype(int)]
    elif kind == "markov":
        om = lm.omega if gamma >= 1.0 else occupancy(lm.P_hat, inst.H, gamma)
        w = om[ii, jj, inst.z0.astype(int)]                # (m, n, L) occupancy from z0
        for k in ["lat", "rel", "thr", "eng"]:
            out[k] = (w * lm.q_state[k]).sum(-1)
    elif kind == "oracle":  # true expectation with the true chain (for diagnostics only)
        Pk = np.broadcast_to(np.eye(L), inst.P.shape).copy()
        om = np.zeros_like(inst.P)
        for _ in range(inst.H):
            om += Pk
            Pk = Pk @ inst.P
        om /= inst.H
        w = om[ii, jj, inst.z0.astype(int)]
        for k in ["lat", "rel", "thr", "eng"]:
            out[k] = (w * getattr(inst, k)).sum(-1)
    else:
        raise ValueError(kind)
    return out


def forecast_error(inst: Instance, lm: LearnedModel, kinds=("static", "last", "markov"),
                   n_sessions: int = 200, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Diagnostic: RMSE of each estimator against realized session-average QoS of every candidate."""
    from .instances import simulate_chain
    rng = np.random.default_rng(seed)
    m, n = inst.m, inst.n
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    real = {k: np.zeros((m, n)) for k in ["lat", "rel", "thr", "eng"]}
    for _ in range(n_sessions):
        traj = simulate_chain(inst.P, inst.z0, inst.H, rng, inst.regime).astype(int)   # (m,n,H)
        for k in real:
            real[k] += getattr(inst, k)[ii[..., None], jj[..., None], traj].mean(-1)
    for k in real:
        real[k] /= n_sessions
    out = {}
    for kind in kinds:
        q = qos_model(inst, lm, kind)
        out[kind] = {k: float(np.sqrt(np.mean(((q[k] - real[k]) / (np.abs(real[k]) + 1e-9)) ** 2)))
                     for k in real}
    return out
