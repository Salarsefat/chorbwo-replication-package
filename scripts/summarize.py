#!/usr/bin/env python3
"""Print a compact per-config summary table from a results .jsonl file."""
import argparse, gzip, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--cols", default="U,solved,rel,lat,cost,thr,fit_markov,V_plan_markov,switches,t_opt")
ap.add_argument("--by", default="dataset,m,n")
args = ap.parse_args()
rows = []
for fn in args.files:
    with (gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn)) as f:
        for line in f:
            r = json.loads(line); r.pop("history", None); rows.append(r)
df = pd.DataFrame(rows)
cols = [c for c in args.cols.split(",") if c in df.columns]
by = args.by.split(",")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for key, g in df.groupby(by):
    print("==", dict(zip(by, key)), f"(seeds={g.seed.nunique()})")
    t = g.groupby("method")[cols].mean().round(4)
    t["n"] = g.groupby("method").size()
    print(t.sort_values("U", ascending=False).to_string())
