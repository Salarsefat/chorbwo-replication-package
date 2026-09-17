#!/usr/bin/env python3
"""Run a set of methods on a grid of (dataset, m, n) configurations and seeds.

Examples
  python scripts/run_experiments.py --phase dev  --out results/dev.jsonl
  python scripts/run_experiments.py --phase test --out results/test_main.jsonl --group main
Seeds (half-open ranges in configs/protocol.yaml): dev 0-4, val 100-104, test 1000-1029.
Hyperparameters are read from configs/tuned_params_frozen.json, the snapshot taken before the
test phase (configs/FROZEN_AT.txt); scripts/tune.py writes configs/tuned_params.json instead.
"""
import argparse, json, os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chorbwo import experiment as X  # noqa: E402

X.DATA_DIR = os.path.join(ROOT, "data")


def load_specs(group: str, tuned_path: str):
    with open(os.path.join(ROOT, "configs", "methods.yaml")) as f:
        groups = yaml.safe_load(f)
    specs = groups[group]
    tuned = {}
    if tuned_path:
        if not os.path.exists(tuned_path):
            raise FileNotFoundError(f"tuned parameter file not found: {tuned_path} "
                                    "(pass --no-tuned to run with the default hyperparameters)")
        with open(tuned_path) as f:
            tuned = json.load(f)
    out = []
    for s in specs:
        s = dict(s)
        s.setdefault("params", {})
        key = s.get("tune_key", s["opt"])
        if key in tuned and s.get("use_tuned", True):
            s["params"] = {**tuned[key]["params"], **s["params"]}
            if "model" in tuned[key] and s.get("model") == "tuned":
                s["model"] = tuned[key]["model"]
            if "gamma" in tuned[key] and "gamma" not in s:
                s["gamma"] = tuned[key]["gamma"]
        if s.get("model") == "tuned":
            s["model"] = "static"
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="dev", choices=["dev", "val", "test"])
    ap.add_argument("--group", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default=None, help="comma list dataset:m:n, default from protocol")
    ap.add_argument("--seeds", default=None, help="comma list or a:b range")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--E", type=int, default=None)
    ap.add_argument("--tuned", default=os.path.join(ROOT, "configs", "tuned_params_frozen.json"),
                    help="hyperparameter file (default: the frozen snapshot used for the paper)")
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--switch-ms", type=float, default=None, help="override handshake latency")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "configs", "protocol.yaml")) as f:
        proto = yaml.safe_load(f)
    ph = proto["phases"][args.phase]
    seeds = list(range(*ph["seeds"])) if args.seeds is None else (
        list(range(*map(int, args.seeds.split(":")))) if ":" in args.seeds else list(map(int, args.seeds.split(","))))
    if args.configs is None:
        configs = [tuple(c) for c in ph["configs"]]
    else:
        configs = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in args.configs.split(",")]
    budget = args.budget or proto["budget"]
    milp_tl = proto.get("milp_time_limit")
    E = args.E or proto["E"]
    H = proto["H"]
    specs = load_specs(args.group, None if args.no_tuned else args.tuned)
    for s in specs:
        s.setdefault("K", proto["K"])  # shortlist alternatives from the protocol unless a spec overrides it
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["m"], r["n"], r["seed"], r["method"]))
    t_start = time.time()
    with open(args.out, "a") as fout:
        for (ds, m, n) in configs:
            for seed in seeds:
                tag = {"synthetic": "syn", "wsdream": "wsd", "synthetic-regime": "synreg"}[ds]
                todo = [s for s in specs if (tag, m, n, seed, s["name"]) not in done]
                if not todo:
                    continue
                t0 = time.time()
                ov = {"switch_ms": args.switch_ms} if args.switch_ms is not None else None
                recs = X.run_case(ds, m, n, seed, todo, budget=budget, E=E, H=H, overrides=ov,
                                  milp_time_limit=milp_tl)
                for r in recs:
                    r["phase"] = args.phase
                    r["group"] = args.group
                    r["budget"] = budget
                    fout.write(json.dumps(r) + "\n")
                fout.flush()
                print(f"[{args.phase}/{args.group}] {ds} m={m} n={n} seed={seed}: {len(recs)} methods in "
                      f"{time.time()-t0:.1f}s (elapsed {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
