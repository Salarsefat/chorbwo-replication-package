"""Experiment protocol shared by development, validation and test runs.

A *method specification* is a dict
    {"name": str, "opt": optimizer key, "params": {...}, "model": "static"|"last"|"markov",
     "policy": "none"|"reactive"|"mdp", "K": shortlist alternatives, "horizon": "full"|"myopic"}

``run_case`` builds the instance for (dataset, m, n, seed), learns the models from the
historical trace, optimizes with every method under the same evaluation budget, and executes
the resulting bindings on E sessions with common random numbers (same execution seed for all
methods of a case) so that paired statistics are meaningful.
"""
from __future__ import annotations

import time
import zlib
from typing import Dict, List

import numpy as np

from .evaluate import Evaluator, set_thresholds
from .instances import Instance, make_synthetic, make_wsdream
from .markov import learn_model, qos_model
from .optimizers import OPTIMIZERS
from .runtime import build_shortlists, build_shortlists_mdp, execute, mdp_policies

DATA_DIR = None  # set by scripts


def build_case(dataset: str, m: int, n: int, seed: int, H: int = 20, overrides: Dict = None):
    if dataset == "synthetic":
        inst = make_synthetic(m, n, seed, H=H)
    elif dataset == "synthetic-regime":
        inst = make_synthetic(m, n, seed, H=H, correlated=True)
    elif dataset == "wsdream":
        inst = make_wsdream(m, n, seed, DATA_DIR, H=H)
    else:
        raise ValueError(dataset)
    for k, v in (overrides or {}).items():
        setattr(inst, k, v)
    lm = learn_model(inst)
    qos = {k: qos_model(inst, lm, k) for k in ["static", "last", "markov"]}
    ev = Evaluator(inst, qos["static"])
    set_thresholds(inst, ev, qos["static"])
    return inst, lm, qos, ev


def run_method(inst: Instance, lm, qos_all, ev: Evaluator, spec: Dict, seed: int, budget: int,
               E: int, exec_seed: int, milp_time_limit: float = None) -> Dict:
    # zlib.crc32 (not the built-in hash) so that the optimizer stream is identical across
    # processes: Python salts str hashing per process unless PYTHONHASHSEED is fixed.
    rng = np.random.default_rng([seed, zlib.crc32(spec["name"].encode("utf-8"))])
    gamma = spec.get("gamma", 1.0)
    if spec["model"] == "markov" and gamma < 1.0:
        q = qos_model(inst, lm, "markov", gamma=gamma)
    else:
        q = qos_all[spec["model"]]
    t0 = time.perf_counter()
    params = dict(spec.get("params", {}))
    if spec["opt"] == "MILP" and milp_time_limit is not None:
        params.setdefault("time_limit", float(milp_time_limit))
    res = OPTIMIZERS[spec["opt"]](ev, q, rng, budget, **params)
    t_opt = time.perf_counter() - t0
    x = res["x"]
    K = spec.get("K", 2)
    t1 = time.perf_counter()
    pol = None
    if spec["policy"] == "mdp" and spec.get("shortlist", "lookahead") == "mdp":
        sl, pol = build_shortlists_mdp(ev, inst, lm, x, q, K, spec.get("J", 6), spec.get("horizon", "full"))
    else:
        sl = build_shortlists(ev, x, q, K)
        if spec["policy"] == "mdp":
            pol = mdp_policies(ev, inst, lm, x, sl, q, spec.get("horizon", "full"))
    t_rt = time.perf_counter() - t1
    ex_rng = np.random.default_rng([exec_seed, 12345])
    met = execute(inst, ev, x, sl, spec["policy"], pol, E, ex_rng, inst.thresholds["lat"])
    plan_eval = ev.evaluate(x, qos_all["markov"])
    rec = {"method": spec["name"], "seed": seed, "dataset": inst.name.split("_")[0], "m": inst.m,
           "n": inst.n, "services": inst.m * inst.n, "fit_plan": float(res["f"]),
           "fit_markov": float(plan_eval["Fpen"][0]), "V_plan_markov": float(plan_eval["V"][0]),
           "evals": int(res["evals"]), "t_opt": t_opt, "t_runtime": t_rt}
    rec.update(met)
    for k in ("milp_status", "milp_status_code", "milp_time", "milp_gap"):
        if k in res:
            rec[k] = res[k]
    rec["history"] = res["history"]
    return rec


def run_case(dataset: str, m: int, n: int, seed: int, specs: List[Dict], budget: int = 3000,
             E: int = 30, H: int = 20, overrides: Dict = None,
             milp_time_limit: float = None) -> List[Dict]:
    inst, lm, qos, ev = build_case(dataset, m, n, seed, H, overrides)
    out = []
    for spec in specs:
        rec = run_method(inst, lm, qos, ev, spec, seed, budget, E, exec_seed=seed,
                         milp_time_limit=milp_time_limit)
        rec["switch_ms"] = inst.switch_ms
        rec["dataset"] = inst.name.split("_")[0]
        out.append(rec)
    return out
