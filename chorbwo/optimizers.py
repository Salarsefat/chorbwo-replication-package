"""Offline binding optimizers.  All population-based methods share one interface

    result = optimize_<name>(ev, qos, rng, budget, **params)

and stop when ``budget`` fitness evaluations have been consumed (every evaluated binding is
counted, including initial populations).  ``result`` is a dict with keys
  x        : best binding found (np.ndarray of ints, length m)
  f        : its penalized fitness
  history  : list of (evaluations_used, best_fpen_so_far)
  evals    : evaluations actually used
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from .evaluate import Evaluator


class _Budget:
    """Counts evaluations and records the best-so-far curve."""

    def __init__(self, ev: Evaluator, qos, budget: int, feasibility_first: bool = False):
        self.ev, self.qos, self.budget = ev, qos, budget
        self.used = 0
        self.best_f = -np.inf
        self.best_x = None
        self.history: List[Tuple[int, float]] = []
        self.ff = feasibility_first

    def exhausted(self) -> bool:
        return self.used >= self.budget

    def evaluate(self, X: np.ndarray):
        X = np.atleast_2d(X)
        room = self.budget - self.used
        if room <= 0:
            return None
        if X.shape[0] > room:
            X = X[:room]
        res = self.ev.evaluate(X, self.qos)
        f = res["Fpen"]
        self.used += X.shape[0]
        i = int(np.argmax(f))
        if f[i] > self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        self.history.append((self.used, self.best_f))
        return X, res

    def result(self, **extra):
        out = {"x": self.best_x, "f": self.best_f, "history": self.history, "evals": self.used}
        out.update(extra)
        return out


def rank_key(res, feasibility_first: bool) -> np.ndarray:
    """Higher is better.  With feasibility-first ranking (constraint domination), any feasible
    binding outranks any infeasible one; infeasible ones are ordered by violation."""
    if not feasibility_first:
        return res["Fpen"]
    feas = res["V"] <= 1e-12
    return np.where(feas, 1.0 + res["F"], -res["V"])


# ======================================================================================
#  Black Widow Optimization (Hayyolalam & Pourhaji Kazem, 2020), discrete adaptation
# ======================================================================================

def _bwo_core(ev: Evaluator, qos, rng: np.random.Generator, budget: int, N: int, pp: float,
              cr0: float, cr1: float, pm: float, cpp, init_frac: float, beta: float,
              guided_mut: float, feasibility_first: bool):
    """Shared skeleton.  Standard BWO: init_frac=0, guided_mut=0, cr0=cr1, feasibility_first=False."""
    m, n = ev.m, ev.n
    B = _Budget(ev, qos, budget)
    D = m if cpp == "m" else int(cpp)
    D = max(2, D - (D % 2))
    S = ev.marginal_scores(qos)                        # (m, n) link-free heuristic scores
    probs = np.exp(beta * (S - S.max(1, keepdims=True)))
    probs /= probs.sum(1, keepdims=True)
    cum = np.cumsum(probs, axis=1)

    def guided_sample(rows):
        u = rng.random(len(rows))
        return np.minimum((u[:, None] > cum[rows]).sum(1), n - 1)

    # ---- initialization (M1: a fraction of the widows is seeded by the marginal scores)
    X = rng.integers(0, n, size=(N, m))
    k_seed = int(round(init_frac * N))
    for i in range(k_seed):
        X[i] = guided_sample(np.arange(m))
    X, res = B.evaluate(X)
    key = rank_key(res, feasibility_first)
    it = 0
    max_it_est = max(1, budget // max(1, (int(round(pp * N)) * D + int(round(pm * N)))))
    while not B.exhausted():
        order = np.argsort(-key)
        X, key = X[order], key[order]
        nr = max(2, int(round(pp * N)))
        pop1 = X[:nr]
        # ---- M3: adaptive cannibalism rate (fraction of children destroyed)
        cr = cr0 + (cr1 - cr0) * min(1.0, it / max_it_est)
        keep_children = max(1, int(round((1 - cr) * D)))
        pop2 = []
        for _ in range(nr):
            a, b = rng.choice(nr, size=2, replace=False)
            mother, father = pop1[a], pop1[b]          # a ranks better than b iff a<b
            if b < a:
                mother, father = father, mother
            kids = np.empty((D, m), dtype=int)
            for c in range(0, D, 2):
                alpha = rng.random(m)
                u = rng.random(m)
                mask = u < alpha
                kids[c] = np.where(mask, mother, father)
                kids[c + 1] = np.where(mask, father, mother)
            out = B.evaluate(kids)
            if out is None:
                break
            kids, kres = out
            kkey = rank_key(kres, feasibility_first)
            srt = np.argsort(-kkey)[:keep_children]
            pop2.append(mother[None, :])
            pop2.append(kids[srt])
            # father is destroyed (sexual cannibalism): it is simply not carried over
        if B.exhausted():
            break
        # ---- mutation (M2: guided mutation re-samples the role from the marginal-score softmax)
        nm = max(1, int(round(pm * N)))
        idx = rng.integers(0, nr, size=nm)
        mut = pop1[idx].copy()
        roles = rng.integers(0, m, size=nm)
        use_guided = rng.random(nm) < guided_mut
        new = rng.integers(0, n, size=nm)
        if use_guided.any():
            new[use_guided] = guided_sample(roles[use_guided])
        mut[np.arange(nm), roles] = new
        out = B.evaluate(mut)
        if out is None:
            break
        mut, _ = out
        pool = np.vstack(pop2 + [mut])
        pres = ev.evaluate(pool, qos)                  # bookkeeping only: already counted
        pkey = rank_key(pres, feasibility_first)
        order = np.argsort(-pkey)[:N]
        X, key = pool[order], pkey[order]
        if X.shape[0] < N:                             # refill by uniform random widows (rare)
            extra = rng.integers(0, n, size=(N - X.shape[0], m))
            out = B.evaluate(extra)
            if out is None:
                break
            extra, eres = out
            X = np.vstack([X, extra])
            key = np.concatenate([key, rank_key(eres, feasibility_first)])
        it += 1
    return B.result()


def optimize_bwo_std(ev, qos, rng, budget, N=40, pp=0.6, cr=0.44, pm=0.4, cpp=2, **_):
    return _bwo_core(ev, qos, rng, budget, N, pp, cr, cr, pm, cpp, init_frac=0.0, beta=1.0,
                     guided_mut=0.0, feasibility_first=False)


def optimize_mdpbwo(ev, qos, rng, budget, N=40, pp=0.6, cr0=0.3, cr1=0.7, pm=0.4, cpp=2,
                   init_frac=0.5, beta=8.0, guided_mut=0.5, feasibility_first=True,
                   M1=True, M2=True, M3=True, **_):
    """Proposed optimizer.  Switches M1/M2/M3 exist for the ablation study."""
    if not M1:
        init_frac = 0.0
    if not M2:
        guided_mut = 0.0
    if not M3:
        cr0 = cr1 = 0.5 * (cr0 + cr1)
        feasibility_first = False
    return _bwo_core(ev, qos, rng, budget, N, pp, cr0, cr1, pm, cpp, init_frac, beta,
                     guided_mut, feasibility_first)


# ======================================================================================
#  Genetic algorithm
# ======================================================================================

def optimize_ga(ev, qos, rng, budget, N=40, pc=0.9, pm=None, pm_mult=1.0, elite=2, k_tour=2, **_):
    m, n = ev.m, ev.n
    pm = (pm_mult / m) if pm is None else pm
    B = _Budget(ev, qos, budget)
    X, res = B.evaluate(rng.integers(0, n, size=(N, m)))
    f = res["Fpen"]
    while not B.exhausted():
        order = np.argsort(-f)
        X, f = X[order], f[order]
        elites = X[:elite]
        # tournament selection
        cand = rng.integers(0, N, size=(N - elite, 2, k_tour))
        winners = np.take_along_axis(cand, np.argmax(f[cand], axis=2)[..., None], axis=2)[..., 0]
        p1, p2 = X[winners[:, 0]], X[winners[:, 1]]
        mask = rng.random(p1.shape) < 0.5
        child = np.where(mask, p1, p2)
        no_cx = rng.random(N - elite) >= pc
        child[no_cx] = p1[no_cx]
        mut = rng.random(child.shape) < pm
        child[mut] = rng.integers(0, n, size=mut.sum())
        out = B.evaluate(child)
        if out is None:
            break
        child, cres = out
        X = np.vstack([elites, child])
        f = np.concatenate([f[:elite], cres["Fpen"]])
    return B.result()


# ======================================================================================
#  Particle swarm optimization (continuous positions, nearest-integer projection)
# ======================================================================================

def optimize_pso(ev, qos, rng, budget, N=40, w=0.7, c1=1.5, c2=1.5, vmax_frac=0.5, **_):
    m, n = ev.m, ev.n
    B = _Budget(ev, qos, budget)
    pos = rng.uniform(0, n - 1, size=(N, m))
    vmax = vmax_frac * (n - 1)
    vel = rng.uniform(-vmax, vmax, size=(N, m))
    X = np.clip(np.rint(pos), 0, n - 1).astype(int)
    X, res = B.evaluate(X)
    f = res["Fpen"]
    pbest, pbest_f = pos.copy(), f.copy()
    g = int(np.argmax(f))
    gbest, gbest_f = pos[g].copy(), f[g]
    while not B.exhausted():
        r1, r2 = rng.random((N, m)), rng.random((N, m))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, 0, n - 1)
        X = np.rint(pos).astype(int)
        out = B.evaluate(X)
        if out is None:
            break
        X, res = out
        f = res["Fpen"]
        improved = f > pbest_f[:len(f)]
        pbest[:len(f)][improved] = pos[:len(f)][improved]
        pbest_f[:len(f)][improved] = f[improved]
        g = int(np.argmax(pbest_f))
        if pbest_f[g] > gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    return B.result()


# ======================================================================================
#  Grey wolf optimizer (continuous update + nearest-integer projection, as in BN-GWO)
# ======================================================================================

def optimize_gwo(ev, qos, rng, budget, N=40, **_):
    m, n = ev.m, ev.n
    B = _Budget(ev, qos, budget)
    pos = rng.uniform(0, n - 1, size=(N, m))
    X = np.rint(pos).astype(int)
    X, res = B.evaluate(X)
    f = res["Fpen"]
    T_max = max(1, (budget - N) // N)
    t = 0
    lead_pos = np.zeros((3, m)); lead_f = np.full(3, -np.inf)

    def update_leaders(pos_, f_):
        for i in range(len(f_)):
            if f_[i] > lead_f[0]:
                lead_f[1:] = lead_f[:-1].copy(); lead_pos[1:] = lead_pos[:-1].copy()
                lead_f[0], lead_pos[0] = f_[i], pos_[i]
            elif f_[i] > lead_f[1]:
                lead_f[2], lead_pos[2] = lead_f[1], lead_pos[1]
                lead_f[1], lead_pos[1] = f_[i], pos_[i]
            elif f_[i] > lead_f[2]:
                lead_f[2], lead_pos[2] = f_[i], pos_[i]

    update_leaders(pos, f)
    while not B.exhausted():
        alpha, beta, delta = lead_pos[0], lead_pos[1], lead_pos[2]
        a = 2.0 - 2.0 * t / max(1, T_max)
        newpos = np.zeros_like(pos)
        for leader in (alpha, beta, delta):
            A = 2 * a * rng.random((N, m)) - a
            C = 2 * rng.random((N, m))
            D = np.abs(C * leader - pos)
            newpos += leader - A * D
        pos = np.clip(newpos / 3.0, 0, n - 1)
        X = np.rint(pos).astype(int)
        out = B.evaluate(X)
        if out is None:
            break
        X, res = out
        f = res["Fpen"]
        pos = pos[:len(f)]
        update_leaders(pos, f)
        t += 1
    return B.result()


# ======================================================================================
#  Ant colony optimization (MAX-MIN ant system with pheromone on (role, candidate))
# ======================================================================================

def optimize_aco(ev, qos, rng, budget, N=40, alpha=1.0, beta=2.0, rho=0.1, **_):
    m, n = ev.m, ev.n
    B = _Budget(ev, qos, budget)
    eta = 0.05 + ev.marginal_scores(qos)
    tau_max = 1.0 / rho
    tau_min = tau_max / (2.0 * n)
    tau = np.full((m, n), tau_max)
    best_x, best_f = None, -np.inf
    while not B.exhausted():
        p = (tau ** alpha) * (eta ** beta)
        p /= p.sum(1, keepdims=True)
        cum = np.cumsum(p, axis=1)
        u = rng.random((N, m))
        X = np.minimum((u[:, :, None] > cum[None, :, :]).sum(2), n - 1)
        out = B.evaluate(X)
        if out is None:
            break
        X, res = out
        f = res["Fpen"]
        i = int(np.argmax(f))
        if f[i] > best_f:
            best_x, best_f = X[i].copy(), f[i]
        tau *= (1 - rho)
        # deposit from the global-best solution (MMAS)
        tau[np.arange(m), best_x] += rho * tau_max
        tau = np.clip(tau, tau_min, tau_max)
    return B.result()


# ======================================================================================
#  MILP on the static linearized surrogate (HiGHS via scipy)
# ======================================================================================

def optimize_milp(ev, qos, rng, budget, time_limit=10.0, **_):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    inst, m, n = ev.inst, ev.m, ev.n
    w, lo, hi = ev.w, ev.lo, ev.hi
    edges = inst.edges
    nx = m * n
    ny = len(edges) * n * n
    # variable layout: x (m*n) | y (E*n*n) | t (m) | T | Theta
    ix = lambda r, j: r * n + j
    iy = lambda e, i, j: nx + e * n * n + i * n + j
    it = lambda r: nx + ny + r
    iT, iTh = nx + ny + m, nx + ny + m + 1
    nv = nx + ny + m + 2
    c = np.zeros(nv)
    # maximize utility -> minimize negative
    for r in range(m):
        for j in range(n):
            c[ix(r, j)] -= w["cost"] * (-qos["cost"][r, j]) / (hi["cost"] - lo["cost"])
            c[ix(r, j)] -= w["eng"] * (-qos["eng"][r, j]) / (hi["eng"] - lo["eng"])
            c[ix(r, j)] -= w["rel"] * np.log(max(qos["rel"][r, j], 1e-9)) / (
                np.log(hi["rel"]) - np.log(max(lo["rel"], 1e-9)))
    for e, (p, r) in enumerate(edges):
        le = inst.link_eng[(p, r)]
        for i in range(n):
            for j in range(n):
                c[iy(e, i, j)] -= w["eng"] * (-le[i, j]) / (hi["eng"] - lo["eng"])
    c[iT] += w["lat"] / (hi["lat"] - lo["lat"])
    c[iTh] -= w["thr"] / (hi["thr"] - lo["thr"])
    rows, cols, vals, lb, ub = [], [], [], [], []
    k = 0

    def add_row(entries, l, u):
        nonlocal k
        for (col, v) in entries:
            rows.append(k); cols.append(col); vals.append(v)
        lb.append(l); ub.append(u); k += 1

    for r in range(m):
        add_row([(ix(r, j), 1.0) for j in range(n)], 1.0, 1.0)
    for e, (p, r) in enumerate(edges):
        for i in range(n):
            for j in range(n):
                add_row([(iy(e, i, j), 1.0), (ix(p, i), -1.0), (ix(r, j), -1.0)], -1.0, np.inf)
        # t_r - t_p - sum_j lat_pj x_pj - sum_ij d_ij y_ij >= 0
        d = inst.link[(p, r)]
        ent = [(it(r), 1.0), (it(p), -1.0)] + [(ix(p, j), -qos["lat"][p, j]) for j in range(n)]
        ent += [(iy(e, i, j), -d[i, j]) for i in range(n) for j in range(n)]
        add_row(ent, 0.0, np.inf)
    for r in range(m):
        add_row([(iT, 1.0), (it(r), -1.0)] + [(ix(r, j), -qos["lat"][r, j]) for j in range(n)], 0.0, np.inf)
        add_row([(iTh, -1.0)] + [(ix(r, j), qos["thr"][r, j]) for j in range(n)], 0.0, np.inf)
    th = inst.thresholds
    if "cost" in th:
        add_row([(ix(r, j), qos["cost"][r, j]) for r in range(m) for j in range(n)], -np.inf, th["cost"])
    add_row([(ix(r, j), np.log(max(qos["rel"][r, j], 1e-9))) for r in range(m) for j in range(n)],
            np.log(max(th["rel"], 1e-9)), np.inf)
    A = coo_matrix((vals, (rows, cols)), shape=(k, nv)).tocsr()
    lbv = np.zeros(nv); ubv = np.full(nv, np.inf)
    ubv[:nx] = 1.0
    ubv[iT] = th["lat"]
    lbv[iTh] = th["thr"]
    integrality = np.zeros(nv); integrality[:nx] = 1
    t0 = time.time()
    r = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
             bounds=Bounds(lbv, ubv), options={"time_limit": time_limit, "disp": False})
    elapsed = time.time() - t0
    if r.x is None:
        x = ev.greedy_plan(qos)  # fallback when no incumbent within the limit
        status = "no_incumbent"
    else:
        x = np.argmax(r.x[:nx].reshape(m, n), axis=1)
        status = r.message
    f = float(ev.evaluate(x, qos)["Fpen"][0])
    # milp_status_code is scipy's result code (0 = optimal, 1 = time/iteration limit, 2 = infeasible, ...)
    return {"x": x, "f": f, "history": [(1, f)], "evals": 1, "milp_status": status,
            "milp_status_code": int(r.status), "milp_time": elapsed,
            "milp_gap": getattr(r, "mip_gap", np.nan)}


# ======================================================================================
#  Greedy (decentralized, no population search)
# ======================================================================================

def optimize_greedy(ev, qos, rng, budget, **_):
    x = ev.greedy_plan(qos)
    f = float(ev.evaluate(x, qos)["Fpen"][0])
    return {"x": x, "f": f, "history": [(ev.n * ev.m, f)], "evals": ev.n * ev.m}


OPTIMIZERS = {
    "MDP-BWO": optimize_mdpbwo,
    "BWO": optimize_bwo_std,
    "GA": optimize_ga,
    "PSO": optimize_pso,
    "GWO": optimize_gwo,
    "ACO": optimize_aco,
    "MILP": optimize_milp,
    "Greedy": optimize_greedy,
}
