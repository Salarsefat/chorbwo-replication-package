"""Runtime (choreography execution) layer.

Each role r keeps a shortlist A_r = (primary, K alternatives).  At every slot the participant
responsible for role r observes the operational states of its shortlisted services and decides
which one to bind for that slot.  Three rebinding policies are implemented:

  * "none"     : never rebind (pure static binding),
  * "reactive" : fail-over — switch to the best-ranked non-faulty shortlisted service only when
                 the currently bound service is observed faulty (the common practice),
  * "mdp"      : the proposed policy — a finite-horizon MDP over the joint state of the shortlist
                 solved offline by value iteration with the learned DTMCs, trading the expected
                 utility gain of a switch against its handshake cost and the recovery probability.

The executor draws the TRUE state trajectories (unknown to the planner), realized latencies and
failures, and returns the executed QoS of the choreography.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .evaluate import Evaluator
from .instances import Instance, L, METRICS, MINIMIZE
from .markov import LearnedModel


def build_shortlists(ev: Evaluator, x: np.ndarray, qos: Dict[str, np.ndarray], K: int) -> np.ndarray:
    """Primary + K best alternatives per role by one-role lookahead on the planning model."""
    m = ev.m
    sl = np.zeros((m, K + 1), dtype=int)
    for r in range(m):
        f = ev.local_lookahead(x, r, qos)
        f[x[r]] = -np.inf
        alt = np.argsort(-f)[:K]
        sl[r, 0] = x[r]
        sl[r, 1:] = alt
    return sl


def _joint_index(Z: np.ndarray) -> np.ndarray:
    """Z (..., K+1) of states -> joint index (...)."""
    idx = np.zeros(Z.shape[:-1], dtype=int)
    for a in range(Z.shape[-1]):
        idx = idx * L + Z[..., a]
    return idx


def _joint_states(Kp1: int) -> np.ndarray:
    nJ = L ** Kp1
    Zj = np.zeros((nJ, Kp1), dtype=int)
    for j in range(nJ):
        v = j
        for a in range(Kp1 - 1, -1, -1):
            Zj[j, a] = v % L; v //= L
    return Zj


def role_utility_table(ev: Evaluator, lm: LearnedModel, x: np.ndarray, r: int, s: int,
                       qos: Dict[str, np.ndarray]) -> np.ndarray:
    """u[z] = global penalized utility of x with role r bound to service s observed in state z
    (other roles at their planning-model expectation)."""
    u = np.zeros(L)
    for z in range(L):
        q = {k: v.copy() for k, v in qos.items()}
        for k in ["lat", "rel", "thr", "eng"]:
            q[k][r, s] = lm.q_state[k][r, s, z]
        xx = x.copy(); xx[r] = s
        u[z] = ev.evaluate(xx, q)["Fpen"][0]
    return u


def role_mdp(inst: Instance, lm: LearnedModel, r: int, sl_r: np.ndarray, u_tab: np.ndarray,
             sigma: float, horizon_mode: str = "full"):
    """Finite-horizon value iteration for one role.
    u_tab (Kp1, L): per-slot reward of binding shortlist position a while it is in state z.
    Returns policy (H, nJ, Kp1) and the value V_0 of starting bound to position 0 in state z0."""
    H = inst.H
    Kp1 = len(sl_r)
    nJ = L ** Kp1
    PJ = np.ones((1, 1))
    for a in range(Kp1):
        PJ = np.kron(PJ, lm.P_hat[r, sl_r[a]])
    Zj = _joint_states(Kp1)
    U = np.stack([u_tab[a, Zj[:, a]] for a in range(Kp1)], axis=1)   # (nJ, Kp1)
    pen = sigma * (1 - np.eye(Kp1))[None, :, :]
    pol = np.zeros((H, nJ, Kp1), dtype=np.int8)
    V_next = np.zeros((nJ, Kp1))
    for h in range(H - 1, -1, -1):
        cont = PJ @ V_next
        if horizon_mode == "myopic":
            cont = np.zeros_like(cont)
        Q = (U + cont)[:, None, :] - pen
        pol[h] = np.argmax(Q, axis=2)
        V_next = np.max(Q, axis=2)
    j0 = _joint_index(inst.z0[r, sl_r].astype(int))
    return pol, float(V_next[j0, 0])


def switch_penalty(ev: Evaluator, inst: Instance) -> float:
    return ev.w["lat"] * inst.switch_ms / (ev.hi["lat"] - ev.lo["lat"])


def mdp_policies(ev: Evaluator, inst: Instance, lm: LearnedModel, x: np.ndarray, sl: np.ndarray,
                 qos: Dict[str, np.ndarray], horizon_mode: str = "full") -> np.ndarray:
    """Policies for given shortlists: (m, H, L^(K+1), K+1) -> action."""
    m, H = ev.m, inst.H
    Kp1 = sl.shape[1]
    sigma = switch_penalty(ev, inst)
    pol = np.zeros((m, H, L ** Kp1, Kp1), dtype=np.int8)
    for r in range(m):
        u_tab = np.stack([role_utility_table(ev, lm, x, r, sl[r, a], qos) for a in range(Kp1)])
        pol[r], _ = role_mdp(inst, lm, r, sl[r], u_tab, sigma, horizon_mode)
    return pol


def build_shortlists_mdp(ev: Evaluator, inst: Instance, lm: LearnedModel, x: np.ndarray,
                         qos: Dict[str, np.ndarray], K: int, J: int = 6,
                         horizon_mode: str = "full"):
    """Value-based shortlist construction (proposed).  For each role, alternatives are added
    greedily from the J best lookahead candidates so as to maximize the MDP value V_0 of the
    resulting shortlist policy.  Returns shortlists (m, K+1) and policies."""
    m, H = ev.m, inst.H
    sigma = switch_penalty(ev, inst)
    sl = np.zeros((m, K + 1), dtype=int)
    pols = []
    for r in range(m):
        f = ev.local_lookahead(x, r, qos)
        f[x[r]] = -np.inf
        cands = [int(c) for c in np.argsort(-f)[:J]]
        utab = {x[r]: role_utility_table(ev, lm, x, r, x[r], qos)}
        for c in cands:
            utab[c] = role_utility_table(ev, lm, x, r, c, qos)
        chosen = [int(x[r])]
        pol_best = None
        for _ in range(K):
            best_v, best_c, best_p = -np.inf, None, None
            for c in cands:
                if c in chosen:
                    continue
                trial = np.array(chosen + [c])
                p, v = role_mdp(inst, lm, r, trial, np.stack([utab[t] for t in trial]), sigma, horizon_mode)
                if v > best_v:
                    best_v, best_c, best_p = v, c, p
            chosen.append(best_c)
            pol_best = best_p
        sl[r] = chosen
        pols.append(pol_best)
    pol = np.stack(pols)
    return sl, pol


def execute(inst: Instance, ev: Evaluator, x: np.ndarray, sl: np.ndarray, policy: str,
            pol_table, E: int, rng: np.random.Generator, T_max: float) -> Dict[str, float]:
    """Simulate E independent sessions of H slots; return executed metrics (means over requests)."""
    m, n, H = inst.m, inst.n, inst.H
    Kp1 = sl.shape[1]
    cum = np.cumsum(inst.P, axis=-1)
    if inst.regime is not None:
        cumR = np.cumsum(inst.regime["P"], axis=-1)           # (m,n,2,L,L)
        Qc = np.cumsum(inst.regime["Q"], axis=-1)
        g = (rng.random(E) < 0.5).astype(int)                 # hidden regime per session
    z = np.broadcast_to(inst.z0.astype(int), (E, m, n)).copy()
    b = np.zeros((E, m), dtype=int)                    # shortlist position currently bound
    sig = np.sqrt(np.log(1 + inst.lat_cv ** 2))
    lat_all = np.zeros((E, H)); succ_all = np.zeros((E, H), dtype=bool)
    cost_all = np.zeros((E, H)); eng_all = np.zeros((E, H)); thr_all = np.zeros((E, H))
    n_switch = np.zeros(E)
    ee = np.arange(E)
    for h in range(H):
        F = np.zeros((E, m)); ok = np.ones(E, dtype=bool)
        cost = np.zeros(E); eng = np.zeros(E); thr = np.full(E, np.inf)
        s_bound = np.zeros((E, m), dtype=int)
        for r in range(m):
            zs = z[:, r, :][:, sl[r]]                   # (E, Kp1) observed shortlist states
            if policy == "none":
                a = b[:, r]
            elif policy == "reactive":
                a = b[:, r].copy()
                faulty = zs[ee, a] == L - 1
                if faulty.any():
                    # best-ranked non-faulty position; keep current if all faulty
                    okpos = zs != L - 1
                    first = np.argmax(okpos, axis=1)
                    has = okpos.any(axis=1)
                    a = np.where(faulty & has, first, a)
            elif policy == "mdp":
                a = pol_table[r, h][_joint_index(zs), b[:, r]].astype(int)
            else:
                raise ValueError(policy)
            switched = a != b[:, r]
            n_switch += switched
            b[:, r] = a
            s = sl[r][a]
            s_bound[:, r] = s
            zt = z[ee, r, s]
            mu = inst.lat[r, s, zt]
            lat = mu * np.exp(rng.normal(-0.5 * sig ** 2, sig, E)) + inst.switch_ms * switched
            best = np.zeros(E)
            for p in inst.preds[r]:
                best = np.maximum(best, F[:, p] + inst.link[(p, r)][s_bound[:, p], s])
            F[:, r] = lat + best
            ok &= rng.random(E) < inst.rel[r, s, zt]
            cost += inst.cost[r, s]
            eng += inst.eng[r, s, zt]
            for p in inst.preds[r]:
                eng += inst.link_eng[(p, r)][s_bound[:, p], s]
            thr = np.minimum(thr, inst.thr[r, s, zt])
        lat_all[:, h] = F.max(1); succ_all[:, h] = ok
        cost_all[:, h] = cost; eng_all[:, h] = eng; thr_all[:, h] = thr
        # true state evolution
        u = rng.random((E, m, n))[..., None]
        if inst.regime is None:
            c = cum[np.arange(m)[:, None], np.arange(n)[None, :]][None]   # (1,m,n,L,L)
            c = np.take_along_axis(c.repeat(E, 0), z[..., None, None], axis=3)[:, :, :, 0, :]
        else:
            cg = cumR[:, :, g]                                             # (m,n,E,L,L)
            cg = np.transpose(cg, (2, 0, 1, 3, 4))                         # (E,m,n,L,L)
            c = np.take_along_axis(cg, z[..., None, None], axis=3)[:, :, :, 0, :]
            g = (rng.random(E) > Qc[g, 0]).astype(int)
        z = np.minimum((u > c).sum(-1), L - 1)
    sr = succ_all.mean()
    out = {"lat": float(lat_all.mean()), "rel": float(sr), "cost": float(cost_all.mean()),
           "eng": float(eng_all.mean()), "thr": float(thr_all.mean()),
           "deadline": float((lat_all <= T_max).mean()),
           "solved": float((succ_all & (lat_all <= T_max)).mean()),
           "switches": float(n_switch.mean()),
           "lat_p95": float(np.percentile(lat_all, 95))}
    agg = {k: np.array([out[k]]) for k in METRICS}
    out["U"] = float(ev.utility(agg)[0])
    out["Vexec"] = float(ev.violation(agg)[0])
    return out
