#!/usr/bin/env python3
"""Hyperparameter tuning on the VALIDATION seeds (100-104, both validation configurations) with the
same cap of random-search trials for every method (--trials, default 12); a method whose grid is
smaller than the cap is searched exhaustively (GWO: 6 settings, MILP: 2).  The score of a trial is
the mean executed utility U over the validation cases.

Writes configs/tuned_params.json and results/tuning.jsonl.  The test runs read
configs/tuned_params_frozen.json, so re-tuning does not change them until you copy the new file
over the frozen one (and update configs/FROZEN_AT.txt).  An interrupted run resumes from
results/tuning.jsonl; resuming requires the same --methods list and --trials, otherwise the trial
numbers point to different settings and the script stops."""
import argparse, itertools, json, os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chorbwo import experiment as X  # noqa: E402

X.DATA_DIR = os.path.join(ROOT, "data")

SPACES = {
    "MDP-BWO": {"N": [30, 40, 60], "pp": [0.4, 0.6, 0.8], "cr0": [0.2, 0.3, 0.4], "cr1": [0.5, 0.7, 0.9],
               "pm": [0.2, 0.4, 0.6], "cpp": [2, 4], "init_frac": [0.3, 0.5, 0.7], "beta": [4, 8, 16],
               "guided_mut": [0.3, 0.5, 0.7], "gamma": [0.5, 0.7, 0.85, 1.0]},
    "BWO": {"N": [30, 40, 60], "pp": [0.4, 0.6, 0.8], "cr": [0.3, 0.44, 0.6], "pm": [0.2, 0.4, 0.6],
            "cpp": [2, 4], "model": ["static", "last"]},
    "GA": {"N": [30, 40, 60], "pc": [0.7, 0.9], "pm_mult": [0.5, 1.0, 2.0], "k_tour": [2, 3],
           "model": ["static", "last"]},
    "PSO": {"N": [30, 40, 60], "w": [0.5, 0.7, 0.9], "c1": [1.0, 1.5, 2.0], "c2": [1.0, 1.5, 2.0],
            "vmax_frac": [0.1, 0.2, 0.5], "model": ["static", "last"]},
    "GWO": {"N": [30, 40, 60], "model": ["static", "last"]},
    "ACO": {"N": [30, 40, 60], "alpha": [1.0, 2.0], "beta": [1.0, 2.0, 3.0], "rho": [0.05, 0.1, 0.2],
            "model": ["static", "last"]},
    "MILP": {"model": ["static", "last"]},
}
POLICY = {"MDP-BWO": ("markov", "mdp", "mdp")}


def sample_trials(space, n_trials, rng):
    keys = list(space)
    grid = list(itertools.product(*[space[k] for k in keys]))
    if len(grid) <= n_trials:
        picks = grid
    else:
        idx = rng.choice(len(grid), size=n_trials, replace=False)
        picks = [grid[i] for i in idx]
    return [dict(zip(keys, p)) for p in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--methods", default="MDP-BWO,BWO,GA,PSO,GWO,ACO,MILP")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tuning.jsonl"))
    ap.add_argument("--params-out", default=os.path.join(ROOT, "configs", "tuned_params.json"))
    args = ap.parse_args()
    with open(os.path.join(ROOT, "configs", "protocol.yaml")) as f:
        proto = yaml.safe_load(f)
    ph = proto["phases"]["val"]
    seeds = list(range(*ph["seeds"]))
    configs = [tuple(c) for c in ph["configs"]]
    rng = np.random.default_rng(2026)
    cases = {}
    for (ds, m, n) in configs:
        for seed in seeds:
            cases[(ds, m, n, seed)] = X.build_case(ds, m, n, seed, proto["H"])
    best = {}
    if os.path.exists(args.params_out):
        best = json.load(open(args.params_out))
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line); done[(r["method"], r["trial"])] = r
    with open(args.out, "a") as fout:
        for meth in args.methods.split(","):
            trials = sample_trials(SPACES[meth], args.trials, rng)
            scores = []
            for ti, tr in enumerate(trials):
                tr = dict(tr)
                model = tr.pop("model", "markov")
                gamma = tr.pop("gamma", 1.0)
                if (meth, ti) in done:
                    prev = done[(meth, ti)]
                    if (prev["params"], prev["model"], prev["gamma"]) != (tr, model, gamma):
                        raise SystemExit(f"{args.out}: trial {ti} of {meth} was recorded with different settings "
                                         f"({prev['params']}, model={prev['model']}, gamma={prev['gamma']}); "
                                         "resume with the original --methods/--trials or use a new --out file")
                    scores.append(prev["score_U"]); continue
                policy = "mdp" if meth == "MDP-BWO" else "reactive"
                spec = {"name": f"{meth}#{ti}", "opt": meth, "params": tr, "model": model,
                        "policy": policy, "gamma": gamma, "shortlist": "mdp", "K": proto["K"]}
                Us, t0 = [], time.time()
                for (ds, m, n, seed), (inst, lm, qos, ev) in cases.items():
                    r = X.run_method(inst, lm, qos, ev, spec, seed, proto["budget"], proto["E"], exec_seed=seed,
                                     milp_time_limit=proto.get("milp_time_limit"))
                    Us.append(r["U"])
                score = float(np.mean(Us))
                scores.append(score)
                rec = {"method": meth, "trial": ti, "params": tr, "model": model, "gamma": gamma,
                       "score_U": score, "U_sd": float(np.std(Us)), "time": time.time() - t0}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"{meth} trial {ti}: U={score:.4f} {tr} model={model} gamma={gamma} ({time.time()-t0:.0f}s)", flush=True)
            bi = int(np.argmax(scores))
            tr = dict(trials[bi]); model = tr.pop("model", "markov"); gamma = tr.pop("gamma", 1.0)
            best[meth] = {"params": tr, "model": model, "gamma": gamma, "score_U": scores[bi],
                          "n_trials": len(trials)}
            json.dump(best, open(args.params_out, "w"), indent=2)
            print(f"== best {meth}: {best[meth]}", flush=True)


if __name__ == "__main__":
    main()
