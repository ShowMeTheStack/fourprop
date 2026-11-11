#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Figures for the paper from the clean-run summaries.

Dataviz method: categorical slots blue/orange/aqua in fixed order (validated for CVD), a single blue ramp for
ordinal/magnitude series, thin marks, recessive grid, text in text tokens, framed legends in empty regions,
direct labels only where they do not collide.
Run:  /usr/bin/python3 fourprop/scripts/make_figures.py results_run2 paper/figs
"""
import json, math, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results_run2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/figs"
os.makedirs(OUT, exist_ok=True)

TXT, TXT2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a897f", "#e6e5e0", "#ffffff"
BLUE, ORANGE, AQUA, LIGHTBLUE = "#2a78d6", "#eb6834", "#1baf7a", "#b7d3f6"
RAMP3 = ["#9ec5f4", "#5598e7", "#1c5cab"]          # orders 0, 1, 2 (sequential blue: light -> dark)
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.edgecolor": GRID, "axes.linewidth": 0.6, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "pdf.fonttype": 42, "font.family": "sans-serif"})
W = 3.45  # Elsevier 5p column width in inches


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=0.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5); ax.set_axisbelow(True)


def legend(ax, handles=None, labels=None, loc="best", **kw):
    opts = dict(loc=loc, frameon=True, framealpha=1.0, facecolor=SURF, edgecolor=GRID, borderpad=0.5, labelspacing=0.35, handlelength=1.8)
    opts.update(kw)
    lg = ax.legend(handles, labels, **opts) if handles else ax.legend(**opts)
    lg.get_frame().set_linewidth(0.5)
    return lg


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig); print("wrote", name)


def load(p):
    d = json.load(open(f"{R}/{p}"))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


p2 = load("phase2/phase2_rows.json"); gate = load("gate/gate_rows.json"); e3 = load("oracle/e3_rows.json")
e6 = load("power/e6_rows.json"); e8h = load("justify_hard/e8_rows.json"); e8r = load("justify_hardr/e8_rows.json"); e10 = load("timing/e10_rows.json")

ARITH = ("log2", "sin", "sqrt", "multiplier", "square")


# ---------------------------------------------------------------- 1. mean error versus order
def fig_error_vs_order():
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.6), gridspec_kw=dict(wspace=0.08), sharey=True)
    ks = [0, 1, 2]
    for ax, fam, title in ((axs[0], "aig", "ISCAS-85 AIGs"), (axs[1], "epfl", "EPFL suite")):
        style(ax, grid="y"); ax.set_yscale("log"); ax.set_title(title, fontsize=8, color=TXT2, pad=3)
        rows = [r for r in p2 if r["family"] == fam and r["regime"] == "half" and r.get("status") == "ok" and r.get("k2_mean") is not None and r["circuit"] != "c17"]
        for r in rows:
            ys = [max(r["k0_mean"], 1e-5), max(r["k1_mean"], 1e-5), max(r["k2_mean"], 1e-5)]
            bad = r["circuit"] in ARITH or r["circuit"] == "c6288"
            ax.plot(ks, ys, color=ORANGE if bad else BLUE, lw=1.0, alpha=0.9 if bad else 0.55, marker="o", ms=2.4, mec=SURF, mew=0.4, zorder=3 if bad else 2)
        ax.set_xticks(ks); ax.set_xticklabels(["0\n(COP)", "1\n(BAM)", "2"]); ax.set_xlim(-0.3, 2.6)
        ax.set_xlabel("order $k$", labelpad=1)
    axs[0].set_ylabel("mean absolute error, $p=\\frac{1}{2}$")
    axs[0].set_ylim(5e-6, 0.3)
    # direct labels for the stories
    lab = {"aig": {"c6288": "c6288", "c2670": "c2670"}, "epfl": {"bar": "bar, arbiter", "adder": "adder", "sqrt": "sqrt", "log2": "log2", "dec": "dec (exact)"}}
    for ax, fam in ((axs[0], "aig"), (axs[1], "epfl")):
        for r in p2:
            if r["family"] == fam and r["regime"] == "half" and r["circuit"] in lab[fam]:
                ax.text(2.08, max(r["k2_mean"], 1e-5), lab[fam][r["circuit"]], fontsize=5.8, va="center", ha="left", color=TXT2)
    h = [Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=2.4), Line2D([], [], color=ORANGE, lw=1.0, marker="o", ms=2.4)]
    legend(axs[0], h, ["control-type circuits", "multiplicative blocks"], loc="lower left", fontsize=5.6, handlelength=1.4)
    save(fig, "fig_error_vs_order")


# ---------------------------------------------------------------- 2. certificate informativeness
def fig_certified():
    rows = [r for r in gate if r["regime"] == "half" and r["status"] == "ok"]
    order = ["c17", "c432", "c499", "c880", "c1355", "c1908"]
    rows = sorted(rows, key=lambda r: order.index(r["circuit"]))
    fig, ax = plt.subplots(figsize=(W, W * 0.55)); style(ax, grid="y")
    x = np.arange(len(rows)); wdt = 0.26
    ax.bar(x - wdt, [r["cut_lt_half"] for r in rows], wdt * 0.92, color=ORANGE, zorder=3)
    ax.bar(x, [r["k1_cert_lt_half"] for r in rows], wdt * 0.92, color=AQUA, zorder=3)
    ax.bar(x + wdt, [r["k2_cert_lt_half"] for r in rows], wdt * 0.92, color=BLUE, zorder=3)
    for i, r in enumerate(rows):
        ax.text(i + wdt, r["k2_cert_lt_half"] + 0.015, f"{r['k2_cert_lt_half']:.2f}", ha="center", va="bottom", fontsize=5.4, color=TXT2)
    ax.set_xticks(x); ax.set_xticklabels([r["circuit"] for r in rows], family="monospace", fontsize=6.8)
    ax.set_ylabel("nets with width $<0.5$"); ax.set_ylim(0, 1.12)
    legend(ax, [Patch(color=ORANGE), Patch(color=AQUA), Patch(color=BLUE)], ["cutting algorithm", "order-1 certificate", "order-2 certificate"], loc="upper right", fontsize=5.8, handlelength=1.2)
    save(fig, "fig_certified")


# ---------------------------------------------------------------- 3. tail mass and errors by depth (cavlc, int2float)
def fig_depth():
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.52), gridspec_kw=dict(wspace=0.08), sharey=True)
    for ax, name in zip(axs, ("cavlc", "int2float")):
        style(ax, grid="y"); ax.set_yscale("log"); ax.set_title(name, fontsize=8, family="monospace", color=TXT2, pad=3)
        r = next(r for r in e3 if r["circuit"] == name and r["regime"] == "half")
        bins = r["by_depth"]; xs = np.arange(len(bins))
        labels = [f"{b['depth_min']}–{b['depth_max']}" if b["depth_min"] != b["depth_max"] else str(b["depth_min"]) for b in bins]
        eps = 1e-5
        m = lambda key: [b[key] if b[key] > 0 else np.nan for b in bins]
        ax.plot(xs, m("tau2"), color=ORANGE, lw=1.2, marker="o", ms=3, mec=SURF, mew=0.4, zorder=4)
        ax.plot(xs, m("prop_k2"), color=BLUE, lw=1.2, marker="s", ms=3, mec=SURF, mew=0.4, zorder=4)
        ax.plot(xs, m("oracle_k2"), color=AQUA, lw=1.2, marker="^", ms=3, mec=SURF, mew=0.4, zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.4); ax.set_xlabel("depth of the AND gate", labelpad=1)
    axs[0].set_ylabel("mean over gates"); axs[0].set_ylim(5e-5, 4)
    h = [Line2D([], [], color=ORANGE, lw=1.2, marker="o", ms=3), Line2D([], [], color=BLUE, lw=1.2, marker="s", ms=3), Line2D([], [], color=AQUA, lw=1.2, marker="^", ms=3)]
    legend(axs[0], h, ["tail mass above degree 2, $\\tau_2$", "propagated error, order 2", "oracle error, order 2"], loc="upper left", fontsize=5.4, handlelength=1.4)
    save(fig, "fig_depth")


# ---------------------------------------------------------------- 4. cost per gate versus number of inputs
def fig_cost():
    fig, ax = plt.subplots(figsize=(W, W * 0.6)); style(ax, grid="both"); ax.set_xscale("log"); ax.set_yscale("log")
    rows = [r for r in e10 if r.get("and")]
    def series(key, color, marker, label):
        pts = sorted((r["n"], 1e3 * r[key] / r["and"], r["circuit"]) for r in rows if r.get(key))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.9, marker=marker, ms=3.2, mec=SURF, mew=0.4, zorder=4, alpha=0.95)
        return pts
    g2 = series("k2_cuda0", BLUE, "o", "GPU, order 2")
    n2 = series("k2_numpy", ORANGE, "s", "NumPy, order 2")
    g1 = series("k1_cuda0", AQUA, "^", "GPU, order 1")
    ax.set_xlabel("primary inputs $n$"); ax.set_ylabel("milliseconds per AND gate")
    ax.set_xlim(4, 1300); ax.set_ylim(0.003, 3)
    ax.axvline(180, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(185, 0.0036, "NumPy/GPU\ncrossover", fontsize=5.6, color=TXT2, ha="left", va="bottom")
    h = [Line2D([], [], color=BLUE, lw=0.9, marker="o", ms=3.2), Line2D([], [], color=ORANGE, lw=0.9, marker="s", ms=3.2), Line2D([], [], color=AQUA, lw=0.9, marker="^", ms=3.2)]
    legend(ax, h, ["GPU, order 2", "NumPy (32 cores), order 2", "GPU, order 1"], loc="upper left", fontsize=5.8, handlelength=1.4)
    save(fig, "fig_cost")


# ---------------------------------------------------------------- 5. hard targets: dumbbell chart
def fig_hard():
    a = {r["circuit"]: r for r in e8h}; b = {r["circuit"]: r for r in e8r}
    cs = [c for c in a if a[c]["n_hard"] > 0]
    cs.sort(key=lambda c: b[c]["hard_mean_opt"] / b[c]["hard_mean_half"])
    fig, ax = plt.subplots(figsize=(W, W * 0.78)); style(ax, grid="x"); ax.set_xscale("log")
    ys = np.arange(len(cs))
    for i, c in enumerate(cs):
        x0, x1, x2 = a[c]["hard_mean_half"], a[c]["hard_mean_opt"], b[c]["hard_mean_opt"]
        ax.plot([x0, x2], [i, i], color=GRID, lw=1.6, zorder=2, solid_capstyle="round")
        ax.plot([x0], [i], marker="o", ms=4.2, color=MUTED, mec=SURF, mew=0.5, lw=0, zorder=3)
        ax.plot([x1], [i], marker="o", ms=4.2, color=AQUA, mec=SURF, mew=0.5, lw=0, zorder=4)
        ax.plot([x2], [i], marker="o", ms=4.2, color=BLUE, mec=SURF, mew=0.5, lw=0, zorder=5)
        g = x2 / x0
        ax.text(x2 * 1.35, i, f"{g:.1f}×" if g < 10 else f"{g:.0f}×", fontsize=5.8, va="center", ha="left", color=TXT2)
    ax.axvline(0.1, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=1); ax.text(0.1, len(cs) - 0.4, "0.1", fontsize=5.6, color=TXT2, ha="center", va="bottom")
    ax.set_yticks(ys); ax.set_yticklabels(cs, family="monospace", fontsize=6.6)
    ax.set_xlabel("mean success probability per pattern"); ax.set_xlim(2e-6, 1.5)
    h = [Line2D([], [], marker="o", color=MUTED, lw=0, ms=4), Line2D([], [], marker="o", color=AQUA, lw=0, ms=4), Line2D([], [], marker="o", color=BLUE, lw=0, ms=4)]
    legend(ax, h, ["at $p=\\frac{1}{2}$", "single ascent from $p=\\frac{1}{2}$", "best of 3 random starts"], loc="upper left", fontsize=5.8, handlelength=1.0)
    save(fig, "fig_hard")


# ---------------------------------------------------------------- 6. per-net switching-activity error by order and library
def fig_power():
    order = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.55), gridspec_kw=dict(wspace=0.08), sharey=True)
    for ax, lib, title in ((axs[0], "sky130", "sky130_fd_sc_hd"), (axs[1], "nangate45", "NanGate45")):
        style(ax, grid="y"); ax.set_yscale("log"); ax.set_title(title, fontsize=7.5, color=TXT2, pad=3)
        rows = {r["circuit"]: r for r in e6 if r["lib"] == lib and r["regime"] == "half" and r["status"] == "ok"}
        cs = [c for c in order if c in rows and c != "c17"]
        x = np.arange(len(cs)); wdt = 0.27
        for j, k in enumerate((0, 1, 2)):
            ax.bar(x + (j - 1) * wdt, [max(rows[c][f"k{k}_alpha_mean"], 1e-5) for c in cs], wdt * 0.92, color=RAMP3[j], zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(cs, family="monospace", fontsize=5.6, rotation=90)
    axs[0].set_ylabel("mean $|\\alpha-\\hat\\alpha|$ over nets"); axs[0].set_ylim(1e-4, 0.3)
    legend(axs[0], [Patch(color=RAMP3[0]), Patch(color=RAMP3[1]), Patch(color=RAMP3[2])], ["order 0 (COP)", "order 1 (BAM)", "order 2"], loc="upper left", fontsize=5.6, handlelength=1.2, ncol=1)
    save(fig, "fig_power")


# ================================================================ extension figures (results_ext/)
EXT = Path(os.environ.get("RESULTS_EXT", str(Path(R).parent / "results_ext")))
ORDER_ISCAS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]


def _ext(p):
    f = EXT / p
    if not f.exists():
        print("skip (missing)", f); return None
    d = json.load(open(f))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


LABEL_K = {"c6288": 1, "c432": 1, "c1355": 1.6, "sqrt": 1, "log2": 1, "square": 1, "priority": 1, "router": 1, "i2c": 1, "cavlc": 1, "int2float": 1, "ctrl": 1, "bar": 1, "dec": 1}


def fig_error_vs_k():
    rows = _ext("ablation/e15_rows.json")
    if rows is None: return
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.62), gridspec_kw=dict(wspace=0.08), sharey=True)
    for ax, fam, title in ((axs[0], "aig", "ISCAS-85 AIGs"), (axs[1], "epfl", "EPFL suite")):
        style(ax, grid="y"); ax.set_yscale("log"); ax.set_title(title, fontsize=8, color=TXT2, pad=3)
        circs = sorted({r["circuit"] for r in rows if r["family"] == fam and r["circuit"] != "c17"})
        for c in circs:
            pts = sorted((r["k"], r["mean_err"]) for r in rows if r["family"] == fam and r["circuit"] == c and r["regime"] == "half" and not r["raw"] and r.get("feasible") and r.get("status") == "ok")
            if not pts: continue
            bad = c in ARITH
            ax.plot([k for k, _ in pts], [max(e, 1e-6) for _, e in pts], color=ORANGE if bad else BLUE, lw=1.0, alpha=0.9 if bad else 0.55, marker="o", ms=2.4, mec=SURF, mew=0.4, zorder=3 if bad else 2)
            if c in LABEL_K:
                ax.text(pts[-1][0] + 0.12, max(pts[-1][1], 1e-6) * LABEL_K[c], c, fontsize=5.2, va="center", ha="left", color=TXT2)
        ax.set_xticks(range(6)); ax.set_xlim(-0.3, 5.9); ax.set_xlabel("order $k$", labelpad=1)
    axs[0].set_ylabel("mean absolute error, $p=\\frac{1}{2}$"); axs[0].set_ylim(5e-7, 0.3)
    h = [Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=2.4), Line2D([], [], color=ORANGE, lw=1.0, marker="o", ms=2.4)]
    legend(axs[0], h, ["control-type circuits", "multiplicative blocks"], loc="lower left", fontsize=5.6, handlelength=1.4)
    save(fig, "fig_error_vs_k")


def fig_cost_vs_k():
    rows = _ext("ablation/e15_rows.json")
    if rows is None: return
    fig, ax = plt.subplots(figsize=(W, W * 0.55)); style(ax, grid="y"); ax.set_yscale("log"); ends = []
    for c in ("c432", "c880", "c3540", "c6288", "cavlc", "int2float", "sin", "square"):
        pts = sorted((r["k"], r["runtime"]) for r in rows if r["circuit"] == c and r["regime"] == "half" and not r["raw"] and r.get("feasible") and r.get("status") == "ok" and r.get("runtime"))
        if not pts: continue
        ax.plot([k for k, _ in pts], [t for _, t in pts], color=ORANGE if c in ARITH else BLUE, lw=1.0, marker="o", ms=2.6, mec=SURF, mew=0.4, alpha=0.8)
        ends.append((pts[-1][0], pts[-1][1], c))
    # end labels: nudge apart in log space within each column so that none overlap (min gap 0.22 decades)
    for kx in sorted({e[0] for e in ends}):
        col = sorted([e for e in ends if e[0] == kx], key=lambda e: e[1]); ys = [math.log10(e[1]) for e in col]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < 0.22: ys[i] = ys[i - 1] + 0.22
        for (k, t, c), y in zip(col, ys):
            ax.text(k + 0.1, 10 ** y, c, fontsize=5.4, va="center", color=TXT2, family="monospace")
    ax.set_xticks(range(6)); ax.set_xlim(-0.2, 6.2); ax.set_xlabel("order $k$", labelpad=1); ax.set_ylabel("GPU time, s (median of 3)")
    save(fig, "fig_cost_vs_k")


def fig_families():
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.55), gridspec_kw=dict(wspace=0.08), sharey=True)
    fams = ["gate", "aig", "resyn2", "dc2"]; cols = [ORANGE, BLUE, AQUA, "#6250d6"]
    circs = [c for c in ORDER_ISCAS if c not in ("c17", "c6288")]
    for ax, k in ((axs[0], 1), (axs[1], 2)):
        style(ax, grid="y"); ax.set_yscale("log"); ax.set_title(f"order {k}", fontsize=8, color=TXT2, pad=3)
        x = np.arange(len(circs)); wdt = 0.2
        for j, fam in enumerate(fams):
            ys = []
            for c in circs:
                r = next((r for r in p2 if r["family"] == fam and r["circuit"] == c and r["regime"] == "half" and r.get("status") == "ok"), None)
                ys.append(max(r[f"k{k}_mean"], 1e-5) if r else np.nan)
            ax.bar(x + (j - 1.5) * wdt, ys, wdt * 0.92, color=cols[j], zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(circs, family="monospace", fontsize=5.6, rotation=90)
    axs[0].set_ylabel("mean absolute error, $p=\\frac{1}{2}$"); axs[0].set_ylim(1e-4, 0.3)
    legend(axs[1], [Patch(color=c) for c in cols], ["gate level", "raw AIG", "resyn2", "dc2"], loc="upper right", fontsize=5.4, handlelength=1.2, ncol=2, columnspacing=0.8)
    save(fig, "fig_families")


def fig_structure_seeds():
    d = json.load(open(f"{R}/phase2/phase2_rows.json")); struct = d["struct"]
    p2b = json.load(open(f"{R}/phase2b/phase2_rows.json"))["rows"]
    circs = [c for c in ORDER_ISCAS if c not in ("c17", "c499", "c1355")]
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.55), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]; style(ax, grid="y"); x = np.arange(len(circs)); wdt = 0.36
    s1 = [next((s["spread_mean"] for s in struct if s["circuit"] == c and s["regime"] == "half" and s["k"] == 1), np.nan) for c in circs]
    s2 = [next((s["spread_mean"] for s in struct if s["circuit"] == c and s["regime"] == "half" and s["k"] == 2), np.nan) for c in circs]
    ax.bar(x - wdt / 2, s1, wdt * 0.92, color=LIGHTBLUE, zorder=3); ax.bar(x + wdt / 2, s2, wdt * 0.92, color=BLUE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(circs, family="monospace", fontsize=5.4, rotation=90); ax.set_ylabel("spread across netlists")
    legend(ax, [Patch(color=LIGHTBLUE), Patch(color=BLUE)], ["order 1", "order 2"], loc="upper left", fontsize=5.6, handlelength=1.2)
    ax = axs[1]; style(ax, grid="y")
    circs2 = [c for c in ORDER_ISCAS if c != "c17"]
    import statistics as st, math
    m, se = [], []
    for c in circs2:
        rs = [r["k1_mean"] / r["k2_mean"] for r in p2b if r["circuit"] == c and r.get("status") == "ok" and r.get("k2_mean") and r["regime"] in ("rand2", "rand3")]
        m.append(st.mean(rs) if rs else np.nan); se.append(st.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else 0)
    ax.errorbar(np.arange(len(circs2)), m, yerr=se, fmt="o", color=BLUE, ms=3.2, mec=SURF, mew=0.4, ecolor=TXT2, elinewidth=0.8, capsize=2, zorder=3)
    ax.axhline(2.0, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=1); ax.axhline(1.0, color=GRID, lw=0.8, zorder=1)
    ax.set_xticks(np.arange(len(circs2))); ax.set_xticklabels(circs2, family="monospace", fontsize=5.4, rotation=90); ax.set_ylabel("error ratio order 1 / order 2")
    ax.set_ylim(0.5, 3.4)
    save(fig, "fig_structure_seeds")


def fig_width_cdf():
    files = sorted((EXT / "cv" / "widths").glob("widths__*__half.json")) if (EXT / "cv" / "widths").exists() else []
    if not files:
        print("skip fig_width_cdf (no per-net width files)"); return
    e11 = {(r["circuit"], r["regime"]): r for r in (_ext("pareto/e11_rows.json") or [])}
    fig, axs = plt.subplots(2, 3, figsize=(W, W * 0.72), gridspec_kw=dict(wspace=0.12, hspace=0.45), sharex=True, sharey=True)
    for ax, f in zip(axs.flat, files):
        d = json.load(open(f)); c = d["task"]["circuit"]; style(ax, grid="both"); ax.set_title(c, fontsize=7, family="monospace", color=TXT2, pad=2)
        P = np.array(d["truth_P"])
        for arr, col, lab in ((np.array(d["w_cut"]), ORANGE, "cutting"), (np.array(d["w_cert1"]), AQUA, "order-1 cert."), (np.array(d["w_cert2"]), BLUE, "order-2 cert.")):
            xs = np.sort(np.minimum(arr, 1.0)); ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=col, lw=1.0)
        r = e11.get((c, "half"))
        if r and r.get("k2_Nstar"):
            N = r["k2_Nstar"]; w = np.sort(2 * 1.96 * np.sqrt(np.maximum(P * (1 - P), 0) / N)); ax.step(w, np.arange(1, len(w) + 1) / len(w), where="post", color=MUTED, lw=1.0, ls=(0, (3, 2)))
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    for ax in axs[1]: ax.set_xlabel("interval width", labelpad=1)
    for ax in axs[:, 0]: ax.set_ylabel("fraction of nets")
    h = [Line2D([], [], color=ORANGE, lw=1), Line2D([], [], color=AQUA, lw=1), Line2D([], [], color=BLUE, lw=1), Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2)))]
    lg = fig.legend(h, ["cutting algorithm", "order-1 certificate", "order-2 certificate", "Monte Carlo 95% CI at $N^*(2)$"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=6.0,
                    frameon=True, framealpha=1.0, facecolor=SURF, edgecolor=GRID, handlelength=1.3, columnspacing=1.0, borderpad=0.4)
    lg.get_frame().set_linewidth(0.5); fig.subplots_adjust(top=0.80)
    save(fig, "fig_width_cdf")


def fig_pareto():
    rows = _ext("pareto/e11_rows.json")
    if rows is None: return
    sel = [("c2670", "aig"), ("c7552", "aig"), ("bar", "epfl"), ("multiplier", "epfl")]
    fig, axs = plt.subplots(2, 2, figsize=(W, W * 0.85), gridspec_kw=dict(wspace=0.1, hspace=0.42), sharex=True, sharey=True)
    NS = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]
    for ax, (c, fam) in zip(axs.flat, sel):
        style(ax, grid="both"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(c, fontsize=7.5, family="monospace", color=TXT2, pad=2)
        r = next((r for r in rows if r["circuit"] == c and r["regime"] == "half"), None)
        if r is None: continue
        ts = [r.get(f"mc_time_N{N}") for N in NS]; es = [r.get(f"mc_err_N{N}") for N in NS]; sd = [r.get(f"mc_err_sd_N{N}", 0) for N in NS]
        pts = [(t, e, s) for t, e, s in zip(ts, es, sd) if t and e]
        ax.errorbar([p[0] for p in pts], [max(p[1], 1e-6) for p in pts], yerr=[p[2] for p in pts], color=MUTED, lw=1.0, marker="o", ms=2.6, mec=SURF, mew=0.4, ecolor=MUTED, elinewidth=0.6, capsize=1.5, zorder=3)
        for k, col in ((0, "#9ec5f4"), (1, "#5598e7"), (2, "#1c5cab"), (3, "#0d366b"), (4, "#0d366b"), (5, "#0d366b")):
            if r.get(f"k{k}_time") and r.get(f"k{k}_err") is not None:
                ax.plot([r[f"k{k}_time"]], [max(r[f"k{k}_err"], 1e-6)], marker="s", ms=4.2, color=col, mec=SURF, mew=0.4, lw=0, zorder=5)
                ax.text(r[f"k{k}_time"] * 1.6, max(r[f"k{k}_err"], 1e-6), f"$k{{=}}{k}$", fontsize=5.2, va="center", ha="left", color=TXT2)
        if r.get("bdd_runtime"):
            ax.plot([r["bdd_runtime"]], [1.5e-6], marker="^", ms=4.5, color=AQUA, mec=SURF, mew=0.4, lw=0, zorder=5); ax.text(r["bdd_runtime"] * 1.4, 1.5e-6, "BDD (exact)", fontsize=5.2, va="center", color=TXT2)
        ax.set_xlim(1e-4, 3e3); ax.set_ylim(1e-6, 0.5)
    for ax in axs[1]: ax.set_xlabel("wall time, s", labelpad=1)
    for ax in axs[:, 0]: ax.set_ylabel("mean absolute error")
    h = [Line2D([], [], color=MUTED, lw=1, marker="o", ms=2.6), Line2D([], [], color="#1c5cab", marker="s", ms=4, lw=0), Line2D([], [], color=AQUA, marker="^", ms=4, lw=0)]
    lg = fig.legend(h, ["Monte Carlo, $N=10^3\\dots10^8$", "order $k$ (GPU)", "BDD (exact)"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=5.4,
                    frameon=True, framealpha=1.0, facecolor=SURF, edgecolor=GRID, handlelength=1.4, columnspacing=1.2, borderpad=0.4)
    lg.get_frame().set_linewidth(0.5); fig.subplots_adjust(top=0.86)
    save(fig, "fig_pareto")


def fig_breakeven():
    rows = _ext("pareto/e11_rows.json")
    if rows is None: return
    rs = [r for r in rows if r["regime"] == "half" and r.get("k2_Nstar")]
    rs.sort(key=lambda r: (r["class"] != "control", r["k2_Nstar"]))
    fig, ax = plt.subplots(figsize=(W, W * 0.5)); style(ax, grid="y"); ax.set_yscale("log")
    x = np.arange(len(rs))
    ax.bar(x, [r["k2_Nstar"] for r in rs], 0.75, color=[ORANGE if r["class"] == "arithmetic" else BLUE for r in rs], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([r["circuit"] for r in rs], family="monospace", fontsize=5.0, rotation=90)
    ax.set_ylabel("Monte Carlo patterns to\nmatch order 2, $N^*(2)$")
    legend(ax, [Patch(color=BLUE), Patch(color=ORANGE)], ["control-type circuits", "multiplicative blocks"], loc="upper left", fontsize=5.4, handlelength=1.2)
    save(fig, "fig_breakeven")


def fig_cv():
    rows = _ext("cv/e12b_rows.json")
    if rows is None: return
    keys = sorted({(r["family"], r["circuit"]) for r in rows}, key=lambda k: (k[0], ORDER_ISCAS.index(k[1]) if k[1] in ORDER_ISCAS else 99, k[1]))
    fig, ax = plt.subplots(figsize=(W, W * 0.55)); style(ax, grid="y"); ax.set_yscale("log")
    x = np.arange(len(keys)); med, lo, hi, pred, v2 = [], [], [], [], []
    for fam, c in keys:
        rs = [r for r in rows if r["family"] == fam and r["circuit"] == c and r["regime"] == "half" and r["N"] == 10 ** 6]
        if not rs: med.append(np.nan); lo.append(0); hi.append(0); pred.append(np.nan); v2.append(np.nan); continue
        m = np.mean([r["vrf1_median"] for r in rs]); med.append(m); lo.append(m - np.mean([r["vrf1_q1"] for r in rs])); hi.append(np.mean([r["vrf1_q3"] for r in rs]) - m)
        pred.append(np.mean([r["pred1_median"] for r in rs])); v2.append(np.mean([r["vrf2_median"] for r in rs if r.get("vrf2_median")]) if any(r.get("vrf2_median") for r in rs) else np.nan)
    ax.errorbar(x, med, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", color=BLUE, ms=3.2, mec=SURF, mew=0.4, ecolor=TXT2, elinewidth=0.7, capsize=1.5, zorder=4)
    ax.plot(x, pred, marker="_", ms=7, color=AQUA, lw=0, mew=1.4, zorder=3)
    ax.plot(x, v2, marker="s", ms=3.0, color=ORANGE, mec=SURF, mew=0.4, lw=0, zorder=4)
    print("fig_cv order-2 values:", [(c, None if np.isnan(v) else round(v, 1)) for (_, c), v in zip(keys, v2)])
    ax.axhline(1.0, color=GRID, lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in keys], family="monospace", fontsize=5.0, rotation=90); ax.set_ylabel("variance reduction factor")
    h = [Line2D([], [], color=BLUE, marker="o", ms=3.2, lw=0), Line2D([], [], color=AQUA, marker="_", ms=7, mew=1.4, lw=0), Line2D([], [], color=ORANGE, marker="s", ms=3, lw=0)]
    legend(ax, h, ["order 1, all nets (median, quartiles)", "predicted $1/(1-\\rho_1)$", "order 2, primary outputs"], loc="upper right", fontsize=5.2, handlelength=1.2)
    ax.set_ylim(0.4, 400)
    save(fig, "fig_cv")


def fig_openroad():
    rows = _ext("openroad/e13_rows.json")
    if rows is None: return
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.6), gridspec_kw=dict(wspace=0.08), sharey=True)
    series = [("k0", "#9ec5f4", "order 0"), ("k1", "#5598e7", "order 1"), ("k2", "#1c5cab", "order 2"), ("default", ORANGE, "OpenSTA default")]
    for ax, plat, title in ((axs[0], "nangate45", "NanGate45"), (axs[1], "sky130hd", "sky130hd")):
        style(ax, grid="y"); ax.set_title(title, fontsize=8, color=TXT2, pad=3)
        rs = [r for r in rows if r["platform"] == plat]; rs.sort(key=lambda r: (r["design"] not in ORDER_ISCAS, ORDER_ISCAS.index(r["design"]) if r["design"] in ORDER_ISCAS else 0, r["design"]))
        x = np.arange(len(rs))
        for j, (m, col, lab) in enumerate(series):
            ys = [100 * r.get(f"{m}_total_relerr", np.nan) if r.get(f"{m}_total_relerr") is not None else np.nan for r in rs]
            ax.plot(x + (j - 1.5) * 0.15, ys, marker="o", ms=2.8, color=col, mec=SURF, mew=0.4, lw=0, zorder=3)
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([r["design"] for r in rs], family="monospace", fontsize=4.8, rotation=90)
    axs[0].set_ylabel("total power error, %")
    legend(axs[1], [Line2D([], [], color=c, marker="o", ms=2.8, lw=0) for _, c, _ in series], [l for _, _, l in series], loc="upper right", fontsize=5.0, handlelength=1.0)
    save(fig, "fig_openroad")


def fig_errmodel():
    """(a) model M1 per net on three enumerable control circuits (order 2); (b) model M2 per circuit and order."""
    d = EXT / "errmodel" / "e14_rows.json"
    if not d.exists(): print("skip fig_errmodel"); return
    rows = json.load(open(d))["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * W, W * 0.66)); style(a1, grid="both"); style(a2, grid="both")
    # (a) M1: signed per-net error, measured against the first-order recursion, order 2, p = 1/2
    nets = EXT / "errmodel" / "m1_nets.json"
    if nets.exists():
        nn = json.load(open(nets)); cols = {"ctrl": BLUE, "int2float": ORANGE, "cavlc": AQUA}; h = []; lim = 0.0
        for c, col in cols.items():
            r = nn.get(f"{c}__k2")
            if r is None: continue
            m = np.array(r["meas"]); q = np.array(r["pred"]); lim = max(lim, np.abs(m).max(), np.abs(q).max())
            a1.plot(q, m, marker="o", ms=2.6, color=col, mec=SURF, mew=0.3, lw=0, alpha=0.8, zorder=3)
            h.append(Line2D([], [], color=col, marker="o", ms=3, lw=0, label=f"{c} (r = {r['corr']:.2f})"))
        lim = float(np.ceil(lim * 100) / 100)
        a1.plot([-lim, lim], [-lim, lim], color=GRID, lw=0.8, zorder=1)
        a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim); a1.set_aspect("equal")
        a1.set_xlabel("predicted signed error (model M1, order 2)"); a1.set_ylabel("measured signed error")
        legend(a1, h, [x.get_label() for x in h], loc="upper left", fontsize=5.2, handlelength=1.0)
        a1.text(0.02, 0.02, "(a)", transform=a1.transAxes, fontsize=7, color=TXT2)
    # (b) M2: mean error per circuit; rows where the engine is (nearly) exact (measured < 1e-4) carry no information
    mk = {1: "o", 2: "s", 3: "^"}
    for r in rows:
        if r["model"] != "m2" or r.get("status") != "ok" or r.get("mean_pred") is None or r["mean_meas"] < 1e-4: continue
        a2.plot([r["mean_pred"]], [r["mean_meas"]], marker=mk.get(r["k"], "o"), ms=3.4, color=ORANGE if r["class"] == "arithmetic" else BLUE, mec=SURF, mew=0.4, lw=0, alpha=0.85, zorder=3)
    lim = (1e-4, 0.5); a2.set_xscale("log"); a2.set_yscale("log")
    a2.plot(lim, lim, color=GRID, lw=0.8, zorder=1); a2.plot(lim, [l * 3 for l in lim], color=GRID, lw=0.6, ls=(0, (3, 2)), zorder=1); a2.plot(lim, [l / 3 for l in lim], color=GRID, lw=0.6, ls=(0, (3, 2)), zorder=1)
    a2.set_xlim(lim); a2.set_ylim(lim); a2.set_xlabel("predicted mean error (model M2)"); a2.set_ylabel("measured mean error")
    h = [Line2D([], [], color=BLUE, marker="o", ms=3, lw=0), Line2D([], [], color=ORANGE, marker="o", ms=3, lw=0)] + [Line2D([], [], color=TXT2, marker=mk[k], ms=3, lw=0) for k in (1, 2, 3)]
    legend(a2, h, ["other", "multiplicative", "order 1", "order 2", "order 3"], loc="upper left", fontsize=5.2, handlelength=1.0, ncol=2, columnspacing=0.8)
    a2.text(0.02, 0.02, "(b)", transform=a2.transAxes, fontsize=7, color=TXT2)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_errmodel")

if __name__ == "__main__":
    fig_error_vs_order(); fig_certified(); fig_depth(); fig_cost(); fig_hard(); fig_power()
    fig_families(); fig_structure_seeds()
    fig_error_vs_k(); fig_cost_vs_k(); fig_width_cdf(); fig_pareto(); fig_breakeven(); fig_cv(); fig_openroad(); fig_errmodel()
