#!/usr/bin/env python3
"""Publication figures with a GraphPad-Prism-like look.

Every panel has a coloured frame (a different colour per dataset), a light patterned background
(tinted fill + dotted grid), large labels, symbols with white edges, error bars with caps and a
shared boxed legend.  Outputs (paper/figures/): PDF (vector), PNG and TIFF at 600 dpi.
Figures: main_<metric> (2x2 or 1x2 panels), convergence, sensitivity_switch, ablation, forecast.
Data come from results/*.jsonl(.gz) and results/summary_forecast.csv only, so run
scripts/analyze.py first (it writes summary_forecast.csv).  PNG and TIFF copies are not committed.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze import load, load_hist, boot_ci, METHODS, MET, DSNAME  # noqa: E402

FIG = os.path.join(ROOT, "paper", "figures"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11.5, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.dpi": 110, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 1.4, "errorbar.capsize": 3,
    "mathtext.fontset": "dejavusans",
})
COLOR = {"MDP-BWO": "#1F4E9C", "BWO": "#E8771E", "GA": "#2E9E44", "PSO": "#D62728", "GWO": "#7B3FB5",
         "ACO": "#8C5A2B", "MILP": "#D63A9A", "MDP-BWO+R0": "#5AA3E0", "Greedy": "#7f7f7f"}
MARK = {"MDP-BWO": "o", "BWO": "s", "GA": "^", "PSO": "v", "GWO": "D", "ACO": "P", "MILP": "X", "MDP-BWO+R0": "o"}
BORDER = {"syn": "#1F4E9C", "wsd": "#1E7F5C", "synreg": "#8A4B08", "neutral": "#4A4A4A"}
BG = "#F5F7FB"


def style(ax, border, title, label):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(2.0); sp.set_color(border)
    ax.grid(True, which="major", color="#C7CFDA", linestyle=(0, (1, 2.5)), linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4.5, width=1.2, colors="#222222")
    ax.set_title(title, fontweight="bold", color=border, pad=9)
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def series(ax, xs, mu, lo, hi, name, emphasize=False):
    lw = 2.6 if emphasize else 1.7
    ms = 9 if emphasize else 7.5
    ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=COLOR[name], marker=MARK.get(name, "o"), ms=ms,
                mec="white", mew=1.2, lw=lw, ls="--" if name == "MDP-BWO+R0" else "-", label=name,
                capsize=3, capthick=1.2, elinewidth=1.2, zorder=4 if emphasize else 3, alpha=0.97)


def legend(fig, handles, labels, ncol, y=0.01):
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=True, fancybox=True,
                     framealpha=1.0, edgecolor="#8A8A8A", bbox_to_anchor=(0.5, y), handlelength=2.6,
                     columnspacing=1.4, borderpad=0.7)
    leg.get_frame().set_linewidth(1.2)
    return leg


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), metadata={"CreationDate": None})  # byte-stable PDFs
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel_data(df, metric, xvar, fixed, meth):
    g = df[(df[fixed[0]] == fixed[1]) & (df.method == meth)].groupby(xvar)[metric]
    xs = np.array(sorted(g.groups.keys()))
    mu = np.array([g.get_group(x).mean() for x in xs])
    ci = np.array([boot_ci(g.get_group(x).values) for x in xs])
    return xs, mu, ci[:, 0], ci[:, 1]


def scaling_figures(dm):
    labels = "abcd"
    for metric, (ylabel, _) in MET.items():
        if metric == "fit_markov":
            continue
        dsets = ["syn"] if metric in ("cost", "eng") else ["syn", "wsd"]
        nrow = len(dsets)
        fig, axes = plt.subplots(nrow, 2, figsize=(7.6, 3.6 * nrow + 0.9), squeeze=False)
        k = 0
        for i, ds in enumerate(dsets):
            d = dm[dm.dataset == ds]
            for j, (xvar, fixed, xlab) in enumerate([("n", ("m", 10), "Candidates per role $n$ ($m$ = 10)"),
                                                     ("m", ("n", 20), "Roles $m$ ($n$ = 20)")]):
                ax = axes[i, j]
                for meth in METHODS:
                    xs, mu, lo, hi = panel_data(d, metric, xvar, fixed, meth)
                    series(ax, xs, mu, lo, hi, meth, emphasize=(meth == "MDP-BWO"))
                ax.set_xticks(sorted(d[xvar].unique()))
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylabel if j == 0 else "")
                if metric == "t_opt":
                    ax.set_yscale("log")
                arrow = "\u2191 higher is better" if MET[metric][1] == "max" else "\u2193 lower is better"
                style(ax, BORDER[ds], f"{DSNAME[ds]}  \u2014  {arrow}", f"({labels[k]})")
                k += 1
        h, l = axes[0, 0].get_legend_handles_labels()
        legend(fig, h, l, ncol=7)
        fig.tight_layout(rect=(0, 0.075 if nrow == 2 else 0.14, 1, 1), h_pad=2.2, w_pad=2.0)
        save(fig, f"main_{metric}")


def convergence_figure():
    hist = load_hist("test_main.jsonl") + load_hist("test_optonly.jsonl")
    grid = np.arange(40, 3001, 40)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))
    for ax, ds, lab in zip(axes, ["syn", "wsd"], "ab"):
        for name in ["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]:
            key = name if name == "MDP-BWO" else name + "+MF"
            Y = np.array([np.interp(grid, [e for e, _ in h], [f for _, f in h]) for (d, m, n, s, meth, h) in hist
                          if d == ds and m == 10 and n == 20 and meth == key])
            mu, sd = Y.mean(0), Y.std(0)
            emph = name == "MDP-BWO"
            ax.plot(grid, mu, color=COLOR[name], lw=2.6 if emph else 1.7, marker=MARK[name], markevery=8,
                    ms=8 if emph else 6.5, mec="white", mew=1.1, label=name, zorder=4 if emph else 3)
            ax.fill_between(grid, mu - sd, mu + sd, color=COLOR[name], alpha=0.12, lw=0)
        ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best penalized fitness $F_{\\mathrm{pen}}$")
        style(ax, BORDER[ds], f"{DSNAME[ds]}, $m$ = 10, $n$ = 20", f"({lab})")
    h, l = axes[0].get_legend_handles_labels()
    legend(fig, h, l, ncol=6)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
    save(fig, "convergence")


def sensitivity_figure(dm):
    parts = [dm[dm.method.isin(["MDP-BWO", "BWO", "GA", "PSO", "GWO", "ACO"]) & (dm.m == 10) & (dm.n == 20)].copy()]
    if "switch_ms" not in parts[0]:
        parts[0]["switch_ms"] = 40.0
    parts[0]["switch_ms"] = parts[0]["switch_ms"].fillna(40.0)
    for sw in [0, 200, 1000]:
        parts.append(load(f"test_sens_sw{sw}.jsonl"))
    abl = load("test_ablation.jsonl")
    r0 = abl[(abl.method == "A10 reactive runtime") & (abl.m == 10) & (abl.n == 20)].copy(); r0["method"] = "MDP-BWO+R0"
    parts.append(r0)
    d = pd.concat(parts)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))
    levels = [0, 40, 200, 1000]; pos = np.arange(len(levels))
    for ax, ds, lab in zip(axes, ["syn", "wsd"], "ab"):
        g = d[d.dataset == ds]
        for meth in ["MDP-BWO", "MDP-BWO+R0", "BWO", "GA", "PSO", "GWO", "ACO"]:
            gm = g[g.method == meth].groupby("switch_ms")["U"]
            mu = np.array([gm.get_group(x).mean() for x in levels])
            ci = np.array([boot_ci(gm.get_group(x).values) for x in levels])
            series(ax, pos, mu, ci[:, 0], ci[:, 1], meth, emphasize=(meth == "MDP-BWO"))
        ax.set_xticks(pos); ax.set_xticklabels([str(v) for v in levels])
        ax.set_xlabel("Handshake latency per rebinding $\\delta_{\\mathrm{sw}}$ (ms)")
        ax.set_ylabel("Executed utility $U$")
        style(ax, BORDER[ds], f"{DSNAME[ds]}, $m$ = 10, $n$ = 20", f"({lab})")
    h, l = axes[0].get_legend_handles_labels()
    legend(fig, h, l, ncol=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
    save(fig, "sensitivity_switch")


def ablation_figure():
    abl = load("test_ablation.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))
    for ax, ds, lab in zip(axes, ["syn", "wsd"], "ab"):
        g = abl[abl.dataset == ds]
        piv = g.pivot_table(index=["m", "n", "seed"], columns="method", values="U")
        names = [n for n in sorted(piv.columns, key=lambda s: int(s.split()[0][1:])) if not n.startswith("A0")]
        diffs, los, his = [], [], []
        for nm in names:
            d = (piv[nm] - piv["A0 full"]).dropna().values
            diffs.append(d.mean()); lo, hi = boot_ci(d); los.append(lo); his.append(hi)
        diffs, los, his = map(np.array, (diffs, los, his))
        y = np.arange(len(names))[::-1]
        colors = ["#C0392B" if v < 0 else "#2E9E44" for v in diffs]
        ax.barh(y, diffs, color=colors, edgecolor="white", height=0.66, zorder=3)
        ax.errorbar(diffs, y, xerr=[diffs - los, his - diffs], fmt="none", ecolor="#222222", elinewidth=1.2, capsize=3, zorder=4)
        ax.axvline(0, color="#222222", lw=1.2, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([n.split(" ", 1)[1].replace("(g=1)", "(\u03b3=1)") for n in names])
        ax.set_xlabel("$\\Delta U$ vs. full method (mean, 95% CI)")
        style(ax, BORDER[ds], f"{DSNAME[ds]} \u2014 ablation", "")
        ax.text(-0.52, 1.06, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(w_pad=2.4)
    save(fig, "ablation")


def forecast_figure():
    d = pd.read_csv(os.path.join(ROOT, "results", "summary_forecast.csv"))
    kinds = ["static", "last", "markov", "oracle"]
    kn = {"static": "historical average", "last": "last observed", "markov": "learned DTMC (proposed)", "oracle": "true chain"}
    kc = {"static": "#8C8C8C", "last": "#E8771E", "markov": "#1F4E9C", "oracle": "#2E9E44"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
    for ax, ds, lab in zip(axes, ["synthetic", "wsdream"], "ab"):
        metrics = ["lat", "rel", "thr"] + (["eng"] if ds == "synthetic" else [])
        x = np.arange(len(metrics)); w = 0.2
        for i, k in enumerate(kinds):
            vals, cis = [], []
            for met in metrics:
                v = d[(d.dataset == ds) & (d.kind == k) & (d.metric == met)].rmse.values
                vals.append(v.mean()); cis.append(boot_ci(v))
            vals = np.array(vals); cis = np.array(cis)
            ax.bar(x + (i - 1.5) * w, vals, w, color=kc[k], edgecolor="white", label=kn[k], zorder=3)
            ax.errorbar(x + (i - 1.5) * w, vals, yerr=[vals - cis[:, 0], cis[:, 1] - vals], fmt="none", ecolor="#222", elinewidth=1.1, capsize=2.5, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels({"lat": "latency", "rel": "reliability", "thr": "throughput", "eng": "energy"}[m_] for m_ in metrics)
        ax.set_ylabel("Relative RMSE (lower is better)")
        style(ax, BORDER["syn" if ds == "synthetic" else "wsd"], "Synthetic" if ds == "synthetic" else "WS-DREAM", f"({lab})")
    h, l = axes[0].get_legend_handles_labels()
    legend(fig, h, l, ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
    save(fig, "forecast")


if __name__ == "__main__":
    dm = load("test_main.jsonl")
    scaling_figures(dm); convergence_figure(); sensitivity_figure(dm); ablation_figure(); forecast_figure()
    print("figures written to", FIG)
