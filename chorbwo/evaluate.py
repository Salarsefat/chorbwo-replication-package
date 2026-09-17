"""Vectorized evaluation of choreography bindings.

A binding is an integer vector x of length m (x[r] = index of the service bound to role r).
``Evaluator.evaluate(X, qos)`` evaluates a whole population X (P, m) at once under a dict of
expected per-candidate QoS arrays ``qos`` (keys lat, rel, thr, eng: (m, n); cost: (m, n)).

Aggregation over the choreography (Section III-B of the paper):
  latency    : critical path of the DAG including peer-to-peer link latencies   (min)
  cost       : sum over roles                                                    (min)
  energy     : sum over roles + sum of link energies over message edges          (min)
  reliability: product over roles                                                (max)
  throughput : minimum over roles                                                (max)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .instances import Instance, METRICS, MINIMIZE


class Evaluator:
    def __init__(self, inst: Instance, ref_qos: Dict[str, np.ndarray], lam: float = 2.0):
        self.inst = inst
        self.lam = lam
        self.m, self.n = inst.m, inst.n
        self.ar = np.arange(self.m)
        # normalization bounds from per-role extreme plans under the reference (static) model
        lo, hi = {}, {}
        for k in METRICS:
            q = ref_qos[k]
            best = np.argmin(q, axis=1) if MINIMIZE[k] else np.argmax(q, axis=1)
            worst = np.argmax(q, axis=1) if MINIMIZE[k] else np.argmin(q, axis=1)
            agg = self.aggregate(np.stack([best, worst]), ref_qos)
            a, b = agg[k]
            lo[k], hi[k] = (min(a, b), max(a, b))
            if hi[k] - lo[k] < 1e-9:
                hi[k] = lo[k] + 1e-9
        self.lo, self.hi = lo, hi
        self.w = inst.weights

    # ------------------------------------------------------------------ aggregation
    def aggregate(self, X: np.ndarray, qos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = np.atleast_2d(X)
        inst = self.inst
        ar = self.ar
        lat = qos["lat"][ar, X]              # (P, m)
        F = np.zeros_like(lat)
        for r in range(self.m):
            best = np.zeros(X.shape[0])
            for p in inst.preds[r]:
                cand = F[:, p] + inst.link[(p, r)][X[:, p], X[:, r]]
                best = np.maximum(best, cand)
            F[:, r] = lat[:, r] + best
        T = F.max(1)
        cost = qos["cost"][ar, X].sum(1)
        eng = qos["eng"][ar, X].sum(1)
        for (p, r) in inst.edges:
            eng = eng + inst.link_eng[(p, r)][X[:, p], X[:, r]]
        rel = np.prod(qos["rel"][ar, X], axis=1)
        thr = qos["thr"][ar, X].min(1)
        return {"lat": T, "cost": cost, "eng": eng, "rel": rel, "thr": thr}

    def normalize(self, agg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k in METRICS:
            v = (self.hi[k] - agg[k]) if MINIMIZE[k] else (agg[k] - self.lo[k])
            out[k] = np.clip(v / (self.hi[k] - self.lo[k]), 0.0, 1.0)
        return out

    def violation(self, agg: Dict[str, np.ndarray]) -> np.ndarray:
        th = self.inst.thresholds
        V = np.zeros_like(agg["lat"])
        for k, c in th.items():
            if MINIMIZE[k]:
                V = V + np.maximum(0.0, (agg[k] - c) / c)
            else:
                V = V + np.maximum(0.0, (c - agg[k]) / c)
        return V

    def utility(self, agg: Dict[str, np.ndarray]) -> np.ndarray:
        nq = self.normalize(agg)
        return sum(self.w[k] * nq[k] for k in METRICS)

    def evaluate(self, X: np.ndarray, qos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        agg = self.aggregate(X, qos)
        F = self.utility(agg)
        V = self.violation(agg)
        agg.update({"F": F, "V": V, "Fpen": F - self.lam * V})
        return agg

    # ------------------------------------------------------------------ helpers for optimizers
    def marginal_scores(self, qos: Dict[str, np.ndarray]) -> np.ndarray:
        """Link-free per-candidate score in [0,1]: weighted normalized own QoS (used as heuristic)."""
        S = np.zeros((self.m, self.n))
        for k in METRICS:
            q = qos[k]
            lo, hi = q.min(1, keepdims=True), q.max(1, keepdims=True)
            nq = (hi - q) if MINIMIZE[k] else (q - lo)
            nq = nq / np.maximum(hi - lo, 1e-12)
            S += self.w[k] * nq
        return S

    def greedy_plan(self, qos: Dict[str, np.ndarray]) -> np.ndarray:
        """Decentralized greedy binding: roles bound in topological order, each role picks the
        candidate maximizing the penalized utility given the already-bound predecessors."""
        x = np.zeros(self.m, dtype=int)
        S = self.marginal_scores(qos)
        for r in range(self.m):
            x[r] = int(np.argmax(S[r]))
        for r in range(self.m):
            X = np.repeat(x[None, :], self.n, axis=0)
            X[:, r] = np.arange(self.n)
            x[r] = int(np.argmax(self.evaluate(X, qos)["Fpen"]))
        return x

    def local_lookahead(self, x: np.ndarray, r: int, qos: Dict[str, np.ndarray]) -> np.ndarray:
        """Penalized utility of x with role r rebound to every candidate (n evaluations)."""
        X = np.repeat(x[None, :], self.n, axis=0)
        X[:, r] = np.arange(self.n)
        return self.evaluate(X, qos)["Fpen"]


def set_thresholds(inst: Instance, ev: Evaluator, qos_ref: Dict[str, np.ndarray],
                   q: float = 0.5, n_samples: int = 2000) -> None:
    """Global constraints calibrated on the reference (static) model so that a uniformly random
    binding satisfies each individual constraint with probability q: the deadline and budget are
    the q-quantiles of random-binding latency and cost, the reliability and throughput floors are
    the (1-q)-quantiles.  Deterministic given the instance seed."""
    rng = np.random.default_rng([inst.seed, 424242])
    X = rng.integers(0, inst.n, size=(n_samples, inst.m))
    agg = ev.aggregate(X, qos_ref)
    th = {"lat": float(np.quantile(agg["lat"], q)),
          "rel": float(np.quantile(agg["rel"], 1 - q)),
          "thr": float(np.quantile(agg["thr"], 1 - q))}
    if inst.weights.get("cost", 0) > 0:
        th["cost"] = float(np.quantile(agg["cost"], q))
    inst.thresholds = th
