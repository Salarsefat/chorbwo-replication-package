#!/usr/bin/env python3
"""Export every figure's data for GraphPad Prism.

For each figure two files are written to paper/prism/:
  * <name>.xlsx  — one sheet per panel in Prism's "XY, enter Mean/SD/N" layout
                   (X column, then Mean | SD | N sub-columns for each method);
  * <name>.pzfx  — a Prism XML project (PrismXMLVersion 5.00) with one XY table per panel
                   (Mean/SD/N sub-columns), generated from the published XML layout.
The .pzfx files have not been validated in GraphPad Prism; if Prism rejects one, import the .xlsx
(File > Import) or paste the sheet into an XY table with "Mean, SD, N" sub-columns.  Output goes to
paper/prism/, which is not committed.
"""
import os, sys, json
import numpy as np, pandas as pd
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze import load, METHODS, MET, DSNAME  # noqa: E402

OUT = os.path.join(ROOT, "paper", "prism"); os.makedirs(OUT, exist_ok=True)


def pzfx(tables, title):
    """tables: list of (table_title, x_title, x_values, {series: (mean, sd, n)})."""
    head = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<GraphPadPrismFile PrismXMLVersion="5.00">',
            '<Created><OriginalVersion CreatedByProgram="chorbwo export" CreatedByVersion="1.0" Login="" DateTime="2026-09-06"/></Created>',
            '<InfoSequence><Ref ID="Info0" Selected="1"/></InfoSequence>',
            f'<Info ID="Info0"><Title>{escape(title)}</Title></Info>',
            '<TableSequence>' + ''.join(f'<Ref ID="Table{i}"{" Selected=\"1\"" if i == 0 else ""}/>' for i in range(len(tables))) + '</TableSequence>']
    body = []
    for i, (ttl, xt, xs, series) in enumerate(tables):
        body.append(f'<Table ID="Table{i}" XFormat="numbers" YFormat="SDN" Replicates="1" TableType="XY" EVFormat="AsteriskAfterNumber">')
        body.append(f'<Title>{escape(ttl)}</Title>')
        body.append(f'<XColumn Width="81" Decimals="2" Subcolumns="1"><Title>{escape(xt)}</Title><Subcolumn>' +
                    ''.join(f'<d>{x:g}</d>' for x in xs) + '</Subcolumn></XColumn>')
        for name, (mu, sd, n) in series.items():
            body.append(f'<YColumn Width="243" Decimals="4" Subcolumns="3"><Title>{escape(name)}</Title>')
            for arr, dec in ((mu, 6), (sd, 6), (n, 0)):
                body.append('<Subcolumn>' + ''.join(f'<d>{v:.{dec}f}</d>' for v in arr) + '</Subcolumn>')
            body.append('</YColumn>')
        body.append('</Table>')
    return '\n'.join(head + body + ['</GraphPadPrismFile>'])


def panel_stats(df, metric, xvar, fixed, methods):
    g = df[df[fixed[0]] == fixed[1]]
    xs = sorted(g[xvar].unique())
    series = {}
    for meth in methods:
        gm = g[g.method == meth].groupby(xvar)[metric]
        series[meth] = (np.array([gm.get_group(x).mean() if x in gm.groups else np.nan for x in xs]),
                        np.array([gm.get_group(x).std(ddof=1) if x in gm.groups else np.nan for x in xs]),
                        np.array([gm.get_group(x).size if x in gm.groups else 0 for x in xs]))
    return xs, series


def write_figure(name, panels):
    """panels: list of (sheet_name, x_title, xs, series)."""
    with pd.ExcelWriter(os.path.join(OUT, f"{name}.xlsx")) as xw:
        for (sheet, xt, xs, series) in panels:
            cols = {xt: xs}
            for meth, (mu, sd, n) in series.items():
                cols[f"{meth} Mean"] = mu; cols[f"{meth} SD"] = sd; cols[f"{meth} N"] = n
            pd.DataFrame(cols).to_excel(xw, sheet_name=sheet[:31], index=False)
    open(os.path.join(OUT, f"{name}.pzfx"), "w").write(pzfx([(s, xt, xs, ser) for (s, xt, xs, ser) in panels], name))


def main():
    dm = load("test_main.jsonl")
    for metric, (label, _) in MET.items():
        if metric == "fit_markov":
            continue
        panels = []
        for ds in (["syn"] if metric in ("cost", "eng") else ["syn", "wsd"]):
            d = dm[dm.dataset == ds]
            for xvar, fixed, xt in [("n", ("m", 10), "Candidates per role n (m=10)"), ("m", ("n", 20), "Roles m (n=20)")]:
                xs, series = panel_stats(d, metric, xvar, fixed, METHODS)
                panels.append((f"{DSNAME[ds]} vs {xvar}", xt, xs, series))
        write_figure(f"fig_{metric}", panels)
    # convergence (mean/SD over seeds of best-so-far fitness at 40-evaluation grid)
    from analyze import load_hist
    hist = load_hist("test_main.jsonl") + load_hist("test_optonly.jsonl")
    grid = np.arange(40, 3001, 40); panels = []
    for ds in ["syn", "wsd"]:
        series = {}
        for name in ["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]:
            key = name if name == "MDP-BWO" else name + "+MF"
            Y = [np.interp(grid, [e for e, _ in h], [f for _, f in h]) for (d, m, n, seed, meth, h) in hist
                 if d == ds and m == 10 and n == 20 and meth == key]
            Y = np.array(Y)
            series[name] = (Y.mean(0), Y.std(0, ddof=1), np.full(len(grid), len(Y)))
        panels.append((f"{DSNAME[ds]} m=10 n=20", "Fitness evaluations", list(grid), series))
    write_figure("fig_convergence", panels)
    # switching-cost sensitivity
    parts = [dm[dm.method.isin(["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]) & (dm.m == 10) & (dm.n == 20)].copy()]
    parts[0]["switch_ms"] = parts[0].get("switch_ms", pd.Series(40.0, index=parts[0].index)).fillna(40.0)
    for sw in [0, 200, 1000]:
        parts.append(load(f"test_sens_sw{sw}.jsonl"))
    abl = load("test_ablation.jsonl"); r0 = abl[(abl.method == "A10 reactive runtime") & (abl.m == 10) & (abl.n == 20)].copy()
    r0["method"] = "MDP-BWO+R0"; parts.append(r0)
    d = pd.concat(parts); panels = []
    for ds in ["syn", "wsd"]:
        g = d[d.dataset == ds]; xs = sorted(g.switch_ms.unique()); series = {}
        for meth in ["MDP-BWO", "MDP-BWO+R0", "BWO", "GA", "PSO", "GWO", "ACO"]:
            gm = g[g.method == meth].groupby("switch_ms")["U"]
            series[meth] = (np.array([gm.get_group(x).mean() for x in xs]), np.array([gm.get_group(x).std(ddof=1) for x in xs]),
                            np.array([gm.get_group(x).size for x in xs]))
        panels.append((f"{DSNAME[ds]} m=10 n=20", "Handshake latency (ms)", xs, series))
    write_figure("fig_sensitivity_switch", panels)
    # ablation (grouped bars): mean/SD/N of U per variant per dataset
    panels = []
    for ds in ["syn", "wsd"]:
        g = abl[abl.dataset == ds]; names = sorted(g.method.unique(), key=lambda s: int(s.split()[0][1:]))
        series = {nm: (np.array([g[g.method == nm].U.mean()]), np.array([g[g.method == nm].U.std(ddof=1)]), np.array([g[g.method == nm].U.size])) for nm in names}
        panels.append((f"Ablation U {DSNAME[ds]}", "variant", [1], series))
    write_figure("fig_ablation_U", panels)
    # ablation as paired differences vs the full method (what the paper plots)
    panels = []
    for ds in ["syn", "wsd"]:
        g = abl[abl.dataset == ds]
        piv = g.pivot_table(index=["m", "n", "seed"], columns="method", values="U")
        names = [n for n in sorted(piv.columns, key=lambda s_: int(s_.split()[0][1:])) if not n.startswith("A0")]
        series = {}
        for nm in names:
            d_ = (piv[nm] - piv["A0 full"]).dropna().values
            series[nm.split(" ", 1)[1]] = (np.array([d_.mean()]), np.array([d_.std(ddof=1)]), np.array([len(d_)]))
        panels.append((f"Ablation dU {DSNAME[ds]}", "variant", [1], series))
    write_figure("fig_ablation_delta", panels)
    # forecast accuracy (relative RMSE per estimator and metric)
    fc = pd.read_csv(os.path.join(ROOT, "results", "summary_forecast.csv"))
    panels = []
    for ds in ["synthetic", "wsdream", "synthetic-regime"]:
        metrics = ["lat", "rel", "thr"] + (["eng"] if ds != "wsdream" else [])
        xs = list(range(1, len(metrics) + 1)); series = {}
        for kind in ["static", "last", "markov", "oracle"]:
            mu, sd, n_ = [], [], []
            for met in metrics:
                v = fc[(fc.dataset == ds) & (fc.kind == kind) & (fc.metric == met)].rmse.values
                mu.append(v.mean()); sd.append(v.std(ddof=1)); n_.append(len(v))
            series[kind] = (np.array(mu), np.array(sd), np.array(n_))
        panels.append((f"Forecast RMSE {ds}", "metric (1=lat,2=rel,3=thr,4=eng)", xs, series))
    write_figure("fig_forecast", panels)
    # one combined workbook with every panel, for convenience
    import glob
    with pd.ExcelWriter(os.path.join(OUT, "all_figures_data.xlsx")) as xw:
        for fn in sorted(glob.glob(os.path.join(OUT, "fig_*.xlsx"))):
            base = os.path.basename(fn)[4:-5]
            for sheet, df_ in pd.read_excel(fn, sheet_name=None).items():
                df_.to_excel(xw, sheet_name=(base + "_" + sheet)[:31], index=False)
    print("written to", OUT, sorted(os.listdir(OUT))[:6], "...")


if __name__ == "__main__":
    main()
