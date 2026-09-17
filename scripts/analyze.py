#!/usr/bin/env python3
"""Generate every table and figure of the paper from the raw results (results/*.jsonl, or the
shipped results/*.jsonl.gz when no uncompressed copy exists).

Statistics (per configuration, paired over the 30 test seeds):
  * Shapiro-Wilk normality of the paired differences (reported; Wilcoxon is used regardless),
  * two-sided Wilcoxon signed-rank test MDP-BWO vs each baseline, Holm-corrected within a
    configuration (6 comparisons),
  * Cliff's delta effect size and win/tie/loss counts,
  * bootstrap 95 % confidence intervals of the means (figure error bars; tables report mean (SD)),
  * Friedman test + mean ranks across methods.
Outputs: paper/tables/*.tex, paper/figures/*.pdf, results/summary_*.csv
"""
import gzip, json, os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); TAB = os.path.join(ROOT, "paper", "tables"); FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, ROOT)

METHODS = ["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO", "MILP"]
BASELINES = METHODS[1:]
DSNAME = {"syn": "Synthetic", "wsd": "WS-DREAM", "synreg": "Synthetic (regime)"}
MET = {"U": ("Executed utility $U$", "max"), "solved": ("Solved requests", "max"), "rel": ("Reliability (success rate)", "max"),
       "thr": ("Throughput (req/s)", "max"), "lat": ("End-to-end latency (ms)", "min"), "cost": ("Cost", "min"),
       "eng": ("Energy (mJ)", "min"), "t_opt": ("Planning time (s)", "min"), "fit_markov": ("Plan fitness", "max")}
# publication style (GraphPad-Prism-like): open axes, outward ticks, sans-serif, symbols + lines
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.dpi": 600, "lines.markersize": 4.5, "lines.linewidth": 1.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3.5,
                     "ytick.major.size": 3.5, "legend.frameon": False, "axes.grid": False,
                     "errorbar.capsize": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})
MARK = {"MDP-BWO": "o", "BWO": "s", "GA": "^", "PSO": "v", "GWO": "D", "ACO": "P", "MILP": "X"}
PDF_META = {"CreationDate": None}  # no timestamp, so regenerated PDFs are byte-stable


def _open(fn):
    """Open a results file, transparently handling the gzipped copies shipped in the repository."""
    path = os.path.join(RES, fn)
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    raise FileNotFoundError(path)


def load(fn):
    rows = []
    with _open(fn) as f:
        for line in f:
            r = json.loads(line); r.pop("history", None); rows.append(r)
    return pd.DataFrame(rows)


def load_hist(fn):
    rows = []
    with _open(fn) as f:
        for line in f:
            r = json.loads(line); rows.append((r["dataset"], r["m"], r["n"], r["seed"], r["method"], r["history"]))
    return rows


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    if len(x) < 2:
        return (x.mean(), x.mean())
    bs = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sign(a[:, None] - b[None, :])).mean())


def holm(pvals):
    p = np.asarray(pvals); k = len(p); order = np.argsort(p); adj = np.empty(k)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (k - i) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def paired_stats(df, metric, ref="MDP-BWO", methods=BASELINES, better="max"):
    """Per config: paired comparison of ref vs each method over seeds."""
    out = []
    for (ds, m, n), g in df.groupby(["dataset", "m", "n"]):
        piv = g.pivot_table(index="seed", columns="method", values=metric)
        if ref not in piv:
            continue
        rows = []
        for meth in methods:
            if meth not in piv:
                continue
            pair = piv[[ref, meth]].dropna()
            d = pair[ref].values - pair[meth].values
            if better == "min":
                d = -d
            if np.all(d == 0):
                p = 1.0
            else:
                p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
            sw = stats.shapiro(d).pvalue if len(d) >= 3 and np.ptp(d) > 0 else np.nan
            delta = cliffs_delta(pair[ref].values, pair[meth].values) * (1 if better == "max" else -1)
            rows.append(dict(dataset=ds, m=m, n=n, method=meth, mean_ref=pair[ref].mean(), mean_other=pair[meth].mean(),
                             diff=d.mean(), p=p, shapiro=sw, delta=delta, wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                             losses=int((d < 0).sum()), n_pairs=len(d)))
        if rows:
            adj = holm([r["p"] for r in rows])
            for r, a in zip(rows, adj):
                r["p_holm"] = a
            out += rows
    return pd.DataFrame(out)


def fmt_p(p):
    if p < 1e-4: return "$<$1e-4"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def friedman_ranks(df, metric, methods, better="max"):
    res = {}
    for (ds, m, n), g in df.groupby(["dataset", "m", "n"]):
        piv = g.pivot_table(index="seed", columns="method", values=metric)[methods].dropna()
        vals = piv.values * (1 if better == "max" else -1)
        ranks = stats.rankdata(-vals, axis=1).mean(0)
        try:
            fp = stats.friedmanchisquare(*[vals[:, i] for i in range(vals.shape[1])]).pvalue
        except Exception:
            fp = np.nan
        res[(ds, m, n)] = (dict(zip(methods, ranks)), fp)
    return res


# ======================================================================================
def main_tables(df):
    sizes = {"services": [(10, 10), (10, 20), (10, 30), (10, 50)], "roles": [(5, 20), (15, 20), (20, 20)]}
    # --- Table: main results per configuration, mean (SD) over seeds, for U and solved
    lines = []
    lines.append(r"\begin{tabular}{llrrrrrrrr}\toprule")
    lines.append(r"Data & $(m,n)$ & Metric & " + " & ".join(METHODS) + r"\\\midrule")
    stat_U = paired_stats(df, "U"); stat_S = paired_stats(df, "solved")
    for ds in ["syn", "wsd"]:
        for (m, n) in [(10, 10), (10, 20), (10, 30), (10, 50), (5, 20), (15, 20), (20, 20)]:
            g = df[(df.dataset == ds) & (df.m == m) & (df.n == n)]
            if g.empty:
                continue
            for metric, st in [("U", stat_U), ("solved", stat_S)]:
                cells = []
                means = g.groupby("method")[metric].mean()
                best = means[METHODS].idxmax()
                for meth in METHODS:
                    x = g[g.method == meth][metric].values
                    s = f"{x.mean():.3f} ({x.std(ddof=1):.3f})"
                    if meth == best:
                        s = r"\textbf{" + s + "}"
                    if meth != "MDP-BWO":
                        row = st[(st.dataset == ds) & (st.m == m) & (st.n == n) & (st.method == meth)]
                        if len(row):
                            ph = row.p_holm.values[0]
                            if ph < 0.05:
                                s += (r"$^{+}$" if row["diff"].values[0] > 0 else r"$^{-}$")
                    cells.append(s)
                lab = "$U$" if metric == "U" else "Solved"
                lines.append(f"{DSNAME[ds] if metric=='U' else ''} & {f'({m},{n})' if metric=='U' else ''} & {lab} & " + " & ".join(cells) + r"\\")
        if lines[-1] != r"\midrule":
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "main_results.tex"), "w").write("\n".join(lines))
    stat_U.to_csv(os.path.join(RES, "summary_stats_U.csv"), index=False)
    stat_S.to_csv(os.path.join(RES, "summary_stats_solved.csv"), index=False)

    # --- Table: statistical summary across configurations (U): wins/ties/losses, sig counts, median delta
    lines = [r"\begin{tabular}{lrrrrrr}\toprule",
             r"Baseline & W/T/L & Sig.\ better & Sig.\ worse & Median $\delta$ & Mean $\Delta U$ & Mean $\Delta$Solved\\\midrule"]
    for meth in BASELINES:
        su = stat_U[stat_U.method == meth]; ss = stat_S[stat_S.method == meth]
        lines.append(f"{meth} & {su.wins.sum()}/{su.ties.sum()}/{su.losses.sum()} & "
                     f"{int(((su.p_holm < 0.05) & (su['diff'] > 0)).sum())}/{len(su)} & {int(((su.p_holm < 0.05) & (su['diff'] < 0)).sum())}/{len(su)} & "
                     f"{su.delta.median():.2f} & {su['diff'].mean():+.3f} & {ss['diff'].mean():+.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "stats_summary.tex"), "w").write("\n".join(lines))

    # --- Table: per-metric means pooled over all configurations, per dataset
    for ds in ["syn", "wsd"]:
        g = df[df.dataset == ds]
        mets = ["U", "solved", "rel", "lat", "thr"] + (["cost", "eng"] if ds == "syn" else []) + ["t_opt"]
        lines = [r"\begin{tabular}{l" + "r" * len(mets) + r"}\toprule",
                 "Method & " + " & ".join(MET[k][0].split(" (")[0] for k in mets) + r"\\\midrule"]
        for meth in METHODS:
            x = g[g.method == meth]
            cells = []
            for k in mets:
                v = x[k].mean()
                cells.append(f"{v:.3f}" if k in ("U", "solved", "rel") else (f"{v:.2f}" if k == "t_opt" else f"{v:.1f}"))
            lines.append(f"{meth} & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, f"metrics_pooled_{ds}.tex"), "w").write("\n".join(lines))

    # --- Friedman ranks
    lines = [r"\begin{tabular}{llrrrrrrrr}\toprule", r"Data & $(m,n)$ & " + " & ".join(METHODS) + r" & Friedman $p$\\\midrule"]
    fr = friedman_ranks(df, "U", METHODS)
    for (ds, m, n), (ranks, fp) in fr.items():
        lines.append(f"{DSNAME[ds]} & ({m},{n}) & " + " & ".join(f"{ranks[k]:.2f}" for k in METHODS) + f" & {fmt_p(fp)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "friedman_ranks.tex"), "w").write("\n".join(lines))
    # per-metric direction summary (which metrics MDP-BWO wins on) pooled
    rows = []
    for k in ["U", "solved", "rel", "thr", "lat", "cost", "eng", "t_opt"]:
        st = paired_stats(df[df.method.isin(METHODS)], k, better=MET[k][1])
        for meth in BASELINES:
            s = st[st.method == meth]
            rows.append(dict(metric=k, baseline=meth, sig_better=int(((s.p_holm < 0.05) & (s["diff"] > 0)).sum()),
                             sig_worse=int(((s.p_holm < 0.05) & (s["diff"] < 0)).sum()), n=len(s), mean_diff=s["diff"].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "summary_metric_directions.csv"), index=False)
    lines = [r"\begin{tabular}{l" + "c" * len(BASELINES) + r"}\toprule", "Metric & " + " & ".join(BASELINES) + r"\\\midrule"]
    dd = pd.DataFrame(rows)
    for k in ["U", "solved", "rel", "thr", "lat", "cost", "eng", "t_opt"]:
        cells = []
        for meth in BASELINES:
            s = dd[(dd.metric == k) & (dd.baseline == meth)].iloc[0]
            cells.append(f"{s.sig_better}/{s.sig_worse}/{s.n}")
        nm = MET[k][0].split(" (")[0] + (" ($\\downarrow$)" if MET[k][1] == "min" else " ($\\uparrow$)")
        lines.append(f"{nm} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "metric_directions.tex"), "w").write("\n".join(lines))


def scaling_figures(df, methods=METHODS, prefix="main"):
    for metric, (label, _) in MET.items():
        if metric == "fit_markov":
            continue
        dsets = ["syn"] if metric in ("cost", "eng") else ["syn", "wsd"]
        fig, axes = plt.subplots(len(dsets), 2, figsize=(6.6, 2.3 * len(dsets)), squeeze=False)
        for i, ds in enumerate(dsets):
            for j, (xvar, fixed) in enumerate([("n", ("m", 10)), ("m", ("n", 20))]):
                ax = axes[i, j]
                g = df[(df.dataset == ds) & (df[fixed[0]] == fixed[1])]
                for meth in methods:
                    gm = g[g.method == meth].groupby(xvar)[metric]
                    xs = sorted(gm.groups.keys())
                    if not xs:
                        continue
                    means = np.array([gm.get_group(x).mean() for x in xs])
                    cis = np.array([boot_ci(gm.get_group(x).values) for x in xs])
                    ax.errorbar(xs, means, yerr=[means - cis[:, 0], cis[:, 1] - means], marker=MARK.get(meth, "o"),
                                label=meth, capsize=2, lw=1)
                ax.set_xlabel("Candidates per role $n$ ($m$=10)" if xvar == "n" else "Roles $m$ ($n$=20)")
                ax.set_xticks(sorted(g[xvar].unique()))
                ax.set_ylabel(label if j == 0 else "")
                ax.set_title(DSNAME[ds], fontsize=8)
                if metric == "t_opt":
                    ax.set_yscale("log")
                ax.grid(False)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.06 if len(dsets) == 1 else 0.05, 1, 1))
        fig.savefig(os.path.join(FIG, f"{prefix}_{metric}.pdf"), metadata=PDF_META)
        fig.savefig(os.path.join(FIG, f"{prefix}_{metric}.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)


def convergence_figure(hist_main, hist_opt):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, ds in zip(axes, ["syn", "wsd"]):
        curves = {}
        for (d, m, n, seed, meth, h) in hist_main + hist_opt:
            if d != ds or m != 10 or n != 20:
                continue
            name = meth.replace("+MF", "")
            if meth == "MDP-BWO+R0" or meth in ("MILP", "Greedy"):
                continue
            if meth in BASELINES:      # main-group baselines optimize a different objective
                continue
            xs = np.array([e for e, _ in h]); ys = np.array([f for _, f in h])
            grid = np.arange(40, 3001, 40)
            yi = np.interp(grid, xs, ys)
            curves.setdefault(name, []).append(yi)
        for name in ["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]:
            if name not in curves:
                continue
            Y = np.array(curves[name]); mu = Y.mean(0); sd = Y.std(0)
            ax.plot(grid, mu, label=name, marker=MARK[name], markevery=10)
            ax.fill_between(grid, mu - sd, mu + sd, alpha=0.12)
        ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best penalized fitness $F_\\mathrm{pen}$")
        ax.set_title(f"{DSNAME[ds]}, $m$=10, $n$=20", fontsize=8)
    axes[0].legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergence.pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, "convergence.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)


def ablation_table(df):
    names = [n for n in df.method.unique()]
    order = sorted(names, key=lambda s: int(s.split()[0][1:]))
    lines = [r"\begin{tabular}{l" + "rrr" * 2 + r"}\toprule",
             r" & \multicolumn{3}{c}{Synthetic} & \multicolumn{3}{c}{WS-DREAM}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Variant & $U$ & $\Delta U$ ($p_\mathrm{Holm}$) & Solved & $U$ & $\Delta U$ ($p_\mathrm{Holm}$) & Solved\\\midrule"]
    stats_rows = []
    for name in order:
        cells = []
        for ds in ["syn", "wsd"]:
            g = df[df.dataset == ds]
            piv = g.pivot_table(index=["m", "n", "seed"], columns="method", values="U")
            fpv = g.pivot_table(index=["m", "n", "seed"], columns="method", values="fit_markov")
            spv = g.pivot_table(index=["m", "n", "seed"], columns="method", values="solved")
            if name not in piv:
                cells += ["--"] * 3; continue
            U = piv[name].mean(); S = spv[name].mean(); Fm = fpv[name].mean()
            if name.startswith("A0"):
                cells += [f"{U:.3f}", "--", f"{S:.3f}"]
                stats_rows.append(dict(dataset=ds, variant=name, dU=0.0, p=1.0, fit_markov=Fm, U=U, solved=S))
            else:
                pair = piv[["A0 full", name]].dropna(); d = pair["A0 full"].values - pair[name].values
                p = 1.0 if np.all(d == 0) else stats.wilcoxon(d).pvalue
                stats_rows.append(dict(dataset=ds, variant=name, dU=-d.mean(), p=p, fit_markov=Fm, U=U, solved=S))
                cells += [f"{U:.3f}", f"{-d.mean():+.3f} ({{P{len(stats_rows)-1}}})", f"{S:.3f}"]
        lines.append(name.split(" ", 1)[1].replace("_", " ").replace("(g=1)", r"($\gamma{=}1$)") + " & " + " & ".join(cells) + r"\\")
    # Holm within dataset
    sr = pd.DataFrame(stats_rows)
    for ds in ["syn", "wsd"]:
        idx = sr.index[sr.dataset == ds]
        sr.loc[idx, "p_holm"] = holm(sr.loc[idx, "p"].values)
    text = "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])
    for i, r in sr.iterrows():
        text = text.replace("{P%d}" % i, fmt_p(r.p_holm))
    open(os.path.join(TAB, "ablation.tex"), "w").write(text)
    sr.to_csv(os.path.join(RES, "summary_ablation.csv"), index=False)


def sensitivity_figure(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, ds in zip(axes, ["syn", "wsd"]):
        g = df_all[df_all.dataset == ds]
        for meth in ["MDP-BWO", "MDP-BWO+R0", "BWO", "GA", "PSO", "GWO", "ACO"]:
            gm = g[g.method == meth].groupby("switch_ms")["U"]
            xs = sorted(gm.groups.keys())
            mu = [gm.get_group(x).mean() for x in xs]
            ax.plot(xs, mu, marker=MARK.get(meth, "x"), label=meth, ls="--" if meth == "MDP-BWO+R0" else "-")
        ax.set_xscale("symlog", linthresh=50); ax.set_xlabel("Rebinding handshake latency $\\delta_\\mathrm{sw}$ (ms)")
        ax.set_ylabel("Executed utility $U$"); ax.set_title(f"{DSNAME[ds]}, $m$=10, $n$=20", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "sensitivity_switch.pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, "sensitivity_switch.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)


def regime_table(df):
    st = paired_stats(df, "U"); ss = paired_stats(df, "solved")
    lines = [r"\begin{tabular}{llrrrrrrr}\toprule", r"$(m,n)$ & Metric & " + " & ".join(METHODS) + r"\\\midrule"]
    for (m, n) in [(10, 20), (20, 20)]:
        g = df[(df.m == m) & (df.n == n)]
        if g.empty:
            continue
        for metric, s_ in [("U", st), ("solved", ss)]:
            means = g.groupby("method")[metric].mean(); best = means[METHODS].idxmax(); cells = []
            for meth in METHODS:
                v = f"{means[meth]:.3f}"
                if meth == best: v = r"\textbf{" + v + "}"
                if meth != "MDP-BWO":
                    row = s_[(s_.m == m) & (s_.n == n) & (s_.method == meth)]
                    if len(row) and row.p_holm.values[0] < 0.05:
                        v += r"$^{+}$" if row["diff"].values[0] > 0 else r"$^{-}$"
                cells.append(v)
            lines.append(f"({m},{n}) & {'$U$' if metric=='U' else 'Solved'} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "regime.tex"), "w").write("\n".join(lines))


def forecast_table():
    from chorbwo import experiment as X
    from chorbwo.markov import forecast_error
    X.DATA_DIR = os.path.join(ROOT, "data")
    rows = []
    for ds in ["synthetic", "wsdream", "synthetic-regime"]:
        for seed in range(1000, 1010):
            inst, lm, qos, ev = X.build_case(ds, 10, 20, seed)
            fe = forecast_error(inst, lm, kinds=("static", "last", "markov", "oracle"), n_sessions=100, seed=seed)
            for kind, d in fe.items():
                for k, v in d.items():
                    rows.append(dict(dataset=ds, seed=seed, kind=kind, metric=k, rmse=v))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "summary_forecast.csv"), index=False)
    lines = [r"\begin{tabular}{llrrrr}\toprule", r"Data & Estimator & Latency & Reliability & Throughput & Energy\\\midrule"]
    for ds in ["synthetic", "wsdream", "synthetic-regime"]:
        for kind in ["static", "last", "markov", "oracle"]:
            g = d[(d.dataset == ds) & (d.kind == kind)].groupby("metric").rmse.mean()
            nm = {"synthetic": "Synthetic", "wsdream": "WS-DREAM", "synthetic-regime": "Synthetic (regime)"}[ds]
            kn = {"static": "historical average", "last": "last observed", "markov": "learned DTMC (proposed)", "oracle": "true DTMC (oracle)" if ds != "synthetic-regime" else "true calm-regime DTMC"}[kind]
            lines.append(f"{nm if kind=='static' else ''} & {kn} & {g['lat']:.3f} & {g['rel']:.3f} & {g['thr']:.3f} & " + (f"{g['eng']:.3f}" if ds != "wsdream" else "--") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "forecast.tex"), "w").write("\n".join(lines))


def timing_table(df):
    lines = [r"\begin{tabular}{llrrrrrrrr}\toprule", r"Data & $(m,n)$ & " + " & ".join(METHODS) + r" & MDP runtime\\\midrule"]
    for ds in ["syn", "wsd"]:
        for (m, n) in [(10, 10), (10, 20), (10, 30), (10, 50), (5, 20), (15, 20), (20, 20)]:
            g = df[(df.dataset == ds) & (df.m == m) & (df.n == n)]
            if g.empty: continue
            cells = [f"{g[g.method == meth].t_opt.mean():.2f}" for meth in METHODS]
            rt = g[g.method == "MDP-BWO"].t_runtime.mean()
            lines.append(f"{DSNAME[ds]} & ({m},{n}) & " + " & ".join(cells) + f" & {rt:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "timing.tex"), "w").write("\n".join(lines))
    # MILP status summary
    g = df[df.method == "MILP"]
    if "milp_status_code" in g.columns and g["milp_status_code"].notna().all():
        optimal = g["milp_status_code"].eq(0)            # scipy status 0 = optimal
    else:                                                # records written before the code was stored
        optimal = g["milp_status"].astype(str).str.startswith("Optimization terminated successfully")
    g = g.assign(_optimal=optimal.astype(float))
    s = g.groupby(["dataset", "m", "n"]).agg(time=("milp_time", "mean"), gap=("milp_gap", "mean"),
                                             optimal=("_optimal", "mean"))
    s.to_csv(os.path.join(RES, "summary_milp.csv"))
    lines = [r"\begin{tabular}{llrrr}\toprule", r"Data & $(m,n)$ & Solved to optimality & Mean time (s) & Mean MILP gap\\\midrule"]
    for (ds, m, n), r in s.iterrows():
        lines.append(f"{DSNAME[ds]} & ({m},{n}) & {100*r.optimal:.0f}\\% & {r.time:.1f} & {r.gap:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "milp_status.tex"), "w").write("\n".join(lines))


def optonly_table(df_main, df_opt):
    """Optimizer-only comparison: everyone with the same discounted Markov forecast and reactive runtime."""
    d = pd.concat([df_main[df_main.method == "MDP-BWO"], df_opt])
    ref = "MDP-BWO+R0"
    others = ["BWO+MF", "GA+MF", "PSO+MF", "GWO+MF", "ACO+MF"]
    lines = [r"\begin{tabular}{llrrrrrrr}\toprule", r"Data & $(m,n)$ & Metric & " + " & ".join([ref] + others) + r"\\\midrule"]
    for metric in ["fit_plan", "U"]:
        st = paired_stats(d, metric, ref=ref, methods=others)
        for ds in ["syn", "wsd"]:
            for (m, n) in [(10, 10), (10, 20), (10, 30), (10, 50), (5, 20), (15, 20), (20, 20)]:
                g = d[(d.dataset == ds) & (d.m == m) & (d.n == n)]
                if g.empty: continue
                means = g.groupby("method")[metric].mean()
                cells = []
                for meth in [ref] + others:
                    v = f"{means[meth]:.3f}"
                    if meth != ref:
                        row = st[(st.dataset == ds) & (st.m == m) & (st.n == n) & (st.method == meth)]
                        if len(row) and row.p_holm.values[0] < 0.05:
                            v += r"$^{+}$" if row["diff"].values[0] > 0 else r"$^{-}$"
                    cells.append(v)
                lines.append(f"{DSNAME[ds]} & ({m},{n}) & {'$F_{pen}$' if metric=='fit_plan' else '$U$'} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "optonly.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    if "all" in what or "main" in what:
        df = load("test_main.jsonl"); main_tables(df); scaling_figures(df); timing_table(df)
        hm = load_hist("test_main.jsonl")
        try:
            ho = load_hist("test_optonly.jsonl"); convergence_figure(hm, ho)
            optonly_table(df, load("test_optonly.jsonl"))
        except FileNotFoundError:
            print("optonly results missing")
    if "all" in what or "ablation" in what:
        try:
            ablation_table(load("test_ablation.jsonl"))
        except FileNotFoundError:
            print("ablation results missing")
    if "all" in what or "sens" in what:
        try:
            parts = [load("test_main.jsonl")]
            parts[0] = parts[0][parts[0].method.isin(["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]) & (parts[0].m == 10) & (parts[0].n == 20)].copy()
            if "switch_ms" not in parts[0]:
                parts[0]["switch_ms"] = 40.0
            parts[0]["switch_ms"] = parts[0]["switch_ms"].fillna(40.0)
            for sw in [0, 200, 1000]:
                parts.append(load(f"test_sens_sw{sw}.jsonl"))
            abl = load("test_ablation.jsonl"); r0 = abl[(abl.method == "A10 reactive runtime") & (abl.m == 10) & (abl.n == 20)].copy()
            r0["method"] = "MDP-BWO+R0"; parts.append(r0)
            sensitivity_figure(pd.concat(parts))
        except FileNotFoundError as e:
            print("sensitivity results missing", e)
    if "all" in what or "regime" in what:
        try:
            regime_table(load("test_regime.jsonl"))
        except FileNotFoundError:
            print("regime results missing")
    if "all" in what or "forecast" in what:
        forecast_table()
    print("done")
