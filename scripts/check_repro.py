#!/usr/bin/env python3
"""Re-run one (dataset, m, n, seed) case and compare it field by field with the shipped records.

By default only the deterministic methods are compared: MILP runs that the shipped record marks as
solved to optimality, and the greedy+MDP ablation variant.  Together they check that the instance
generator, the learned models, the evaluator and the execution simulator are unchanged.  --all also
compares the population-based optimizers; that only succeeds for records written by this version of
the code (see "Reproducibility" in README.md).  Timing fields are never compared, and MILP runs
that hit the time limit are skipped because their result depends on machine speed.
Exit status: 0 if every compared record matches, 1 otherwise.

Examples
  python scripts/check_repro.py --group main --config synthetic:5:20 --seed 1000
  python scripts/check_repro.py --group ablation --config synthetic:5:20 --seed 1000
  python scripts/check_repro.py --group sens --switch-ms 200 --config wsdream:10:20 --seed 1000 --all
"""
import argparse, gzip, json, math, os, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_experiments import X, load_specs  # noqa: E402  (importing also sets the data directory)

DETERMINISTIC = {"MILP", "A12 greedy+MDP"}
TIMING = {"t_opt", "t_runtime", "milp_time"}
TAG = {"synthetic": "syn", "wsdream": "wsd", "synthetic-regime": "synreg"}


def shipped_file(group, dataset, switch_ms):
    if group == "sens":
        if switch_ms is None:
            raise SystemExit("--group sens needs --switch-ms (0, 200 or 1000)")
        return f"test_sens_sw{int(switch_ms)}.jsonl.gz"
    if group == "main" and dataset == "synthetic-regime":
        return "test_regime.jsonl.gz"
    return {"main": "test_main.jsonl.gz", "optonly": "test_optonly.jsonl.gz",
            "ablation": "test_ablation.jsonl.gz"}[group]


def milp_optimal(rec):
    if "milp_status_code" in rec:
        return rec["milp_status_code"] == 0
    return str(rec.get("milp_status", "")).startswith("Optimization terminated successfully")


def same(a, b, rel=1e-9, abs_=1e-12):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rel, abs_) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["main", "optonly", "ablation", "sens"])
    ap.add_argument("--config", required=True, help="dataset:m:n, e.g. synthetic:5:20")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--switch-ms", type=float, default=None, help="for --group sens")
    ap.add_argument("--methods", default=None, help="comma list of method names to compare")
    ap.add_argument("--all", action="store_true", help="compare every method of the group")
    ap.add_argument("--shipped", default=None, help="records file (default: chosen from --group)")
    args = ap.parse_args()

    ds, m, n = args.config.split(":"); m, n = int(m), int(n)
    with open(os.path.join(ROOT, "configs", "protocol.yaml")) as f:
        proto = yaml.safe_load(f)
    specs = load_specs(args.group, os.path.join(ROOT, "configs", "tuned_params_frozen.json"))
    names = [s["name"] for s in specs]
    if args.methods:
        wanted = [x.strip() for x in args.methods.split(",")]
    elif args.all:
        wanted = names
    else:
        wanted = [x for x in names if x in DETERMINISTIC]
        if not wanted:
            raise SystemExit(f"group '{args.group}' has no deterministic method; use --all or --methods")
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise SystemExit(f"not in group '{args.group}': {unknown}")

    path = args.shipped or os.path.join(ROOT, "results", shipped_file(args.group, ds, args.switch_ms))
    key0 = (TAG[ds], m, n, args.seed)
    shipped = {}
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            r = json.loads(line)
            if (r["dataset"], r["m"], r["n"], r["seed"]) == key0 and r["method"] in wanted:
                shipped[r["method"]] = r
    missing = sorted(set(wanted) - set(shipped))
    if missing:
        raise SystemExit(f"{os.path.basename(path)} has no record for {key0} and {missing}")

    todo = [s for s in specs if s["name"] in wanted]
    for s in todo:
        s.setdefault("K", proto["K"])
    ov = {"switch_ms": args.switch_ms} if args.switch_ms is not None else None
    recs = X.run_case(ds, m, n, args.seed, todo, budget=proto["budget"], E=proto["E"], H=proto["H"],
                      overrides=ov, milp_time_limit=proto.get("milp_time_limit"))

    failed = 0
    print(f"{os.path.basename(path)}  {ds} m={m} n={n} seed={args.seed}")
    for r in recs:
        r.update(phase="test", group=args.group, budget=proto["budget"])  # as run_experiments.py adds
        old = shipped[r["method"]]
        if r["method"] == "MILP" and not milp_optimal(old):
            print(f"  skip  {r['method']:24s} shipped run hit the time limit (machine-dependent)")
            continue
        bad = [k for k in old if k not in TIMING and not same(old[k], r.get(k, "<absent>"))]
        if bad:
            failed += 1
            print(f"  FAIL  {r['method']:24s} differs in: {', '.join(bad)}")
        else:
            print(f"  ok    {r['method']:24s} all {len([k for k in old if k not in TIMING])} non-timing fields match")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
