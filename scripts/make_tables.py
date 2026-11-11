#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Generate LaTeX tables for the paper from the experiment JSON rows (no hand-entered numbers).

Usage: make_tables.py --algebra ~/rhiddhi/algebra --out ~/rhiddhi/algebra/paper/tables
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
    return str(v)


CLS = {"control": "other", "arithmetic": "mult."}


def table(path: Path, caption: str, label: str, header: list, rows: list, colspec: str | None = None):
    colspec = colspec or "l" * len(header)
    L = ["\\begin{table*}[!tbp]\\centering\\footnotesize", f"\\caption{{{caption}}}\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}\\toprule",
         " & ".join(header) + " \\\\\\midrule"]
    for r in rows:
        L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--algebra", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ext", default=None)
    a = ap.parse_args(); A = Path(a.algebra).expanduser(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    # Table 1: ISCAS-85 AIGs, orders 0/1/2 with projection, from proj_rows.json (k_clamp = BAM-style, k_norm = ours)
    if (A / "proj_rows.json").exists():
        rows = []
        for r in json.load(open(A / "proj_rows.json")):
            if r.get("status") != "ok":
                continue
            rows.append([r["circuit"], r["regime"], fmt(r.get("k1_clamp_mean")), fmt(r.get("k1_norm_mean")), fmt(r.get("k2_norm_mean")),
                         fmt(r.get("k1_clamp_max")), fmt(r.get("k2_norm_max")), fmt(r.get("k2_norm_rt"), 2)])
        table(out / "iscas_orders.tex", "ISCAS-85 AIGs: mean and maximum absolute error of the signal probability over all nets versus exact ground truth (BDD; Monte Carlo with $10^8$ patterns for c6288). Order 1 with clamping is BAM; ``norm'' is the feasibility projection.",
              "tab:iscas", ["circuit", "$p$", "k1 clamp", "k1 norm", "k2 norm", "k1 max", "k2 max", "k2 GPU s"], rows, "llrrrrrr")

    # Table 2: netlist families and EPFL
    if (A / "phase2_rows.json").exists():
        d = json.load(open(A / "phase2_rows.json"))
        def p2rows(pred):
            out = []
            for r in d["rows"]:
                if r.get("status") != "ok" or "k1_mean" not in r or r["regime"] != "half" or not pred(r):
                    continue
                out.append([r["family"], r["circuit"], str(r["n_inputs"]), str(r["n_nodes"]), str(r["depth"]),
                            fmt(r.get("k0_mean", "")), fmt(r["k1_mean"]), fmt(r.get("k2_mean", "--")), fmt(r.get("k1_max")), fmt(r.get("k2_max", "--")),
                            fmt(r.get("cert_lt_0.5", ""), 2)])
            return out
        hdr = ["family", "circuit", "$n$", "nodes", "depth", "k0", "k1", "k2", "k1 max", "k2 max", "cert$<$0.5"]
        table(out / "phase2_iscas.tex", "ISCAS-85 in four netlist forms at $p=\\tfrac12$ (projection on): mean absolute error at orders 0, 1, 2, maximum error at orders 1 and 2, and the fraction of nets whose certified order-1 interval is narrower than 0.5. Ground truth is exact (BDD) for the gate-level, resyn2 and dc2 forms except c6288; the raw-AIG rows of this run were scored against Monte Carlo with $10^8$ patterns (standard error at most $5\\times10^{-5}$), as was c6288.",
              "tab:phase2_iscas", hdr, p2rows(lambda r: r["family"] != "epfl"), "llrrrrrrrrr")
        table(out / "phase2_epfl.tex", "EPFL suite (AIGs) at $p=\\tfrac12$, projection on; columns as in Table~\\ref{tab:phase2_iscas}; ground truth by Monte Carlo with $10^8$ patterns.",
              "tab:phase2_epfl", hdr, p2rows(lambda r: r["family"] == "epfl"), "llrrrrrrrrr")
        srows = [[s["circuit"], s["regime"], str(s["k"]), str(len(s["variants"])), str(s["n_outputs"]), fmt(s["spread_mean"]), fmt(s["spread_max"])] for s in d["struct"]]
        table(out / "structure.tex", "Structure dependence at $p=\\tfrac12$: for each ISCAS-85 function, the spread (max$-$min) of the estimated output probabilities across its netlists (raw AIG, original gate level, resyn2, dc2), averaged over outputs and maximised over outputs.",
              "tab:structure", ["circuit", "$p$", "$k$", "netlists", "outputs", "mean spread", "max spread"], srows, "lllrrrr")

    # Table 3: certified vs cutting (gate experiment)
    if (A / "gate_rows.json").exists():
        rows = []
        for r in json.load(open(A / "gate_rows.json")):
            if r.get("status") != "ok":
                continue
            rows.append([r["circuit"], r["regime"], str(r.get("cut_stems", "")), fmt(r.get("cut_width_median")), fmt(r.get("cut_lt_half"), 2),
                         fmt(r.get("k1_cert_median")), fmt(r.get("k1_cert_lt_half"), 2), fmt(r.get("k2_cert_median")), fmt(r.get("k2_cert_lt_half"), 2), fmt(r.get("k2_tighter_frac"), 2)])
        table(out / "certified.tex", "Certified bounds versus the cutting algorithm on the six smaller ISCAS-85 AIGs: number of reconvergent stems, median interval width and fraction of nets with width below 0.5 for the cutting algorithm and for the order-1 and order-2 certificates, and the fraction of nets where the order-2 certificate is strictly tighter.",
              "tab:certified", ["circuit", "$p$", "stems", "cut med.", "cut$<$.5", "k1 med.", "k1$<$.5", "k2 med.", "k2$<$.5", "k2 tighter"], rows, "llrrrrrrrr")
    # ---------------------------------------------------------------- extension tables (results_ext/<job>/*rows.json), skipped if absent
    E = Path(a.ext).expanduser() if a.ext else A.parent / "results_ext"

    def ext(pth):
        f = E / pth
        if not f.exists():
            print("skip", f); return None
        d = json.load(open(f))
        return d["rows"] if isinstance(d, dict) and "rows" in d else d

    rows = ext("ablation/e15_rows.json")
    if rows:
        circs = sorted({(r["family"], r["circuit"]) for r in rows}, key=lambda x: (x[0] != "aig", x[1]))
        out_rows = []
        for fam, c in circs:
            cells = []
            for k in range(6):
                r = next((x for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["k"] == k and not x["raw"] and x.get("status") == "ok"), None)
                cells.append("--" if r is None or not r.get("feasible") else fmt(r["mean_err"]))
            n = next((x["n_inputs"] for x in rows if x["family"] == fam and x["circuit"] == c and "n_inputs" in x), "")
            t3 = next((x.get("runtime") for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["k"] == 3 and not x["raw"] and x.get("feasible")), None)
            out_rows.append([fam, c, str(n)] + cells + [fmt(t3, 2) if t3 else "--"])
        table(out / "ablation.tex", "Order ablation at $p=\\tfrac12$ (projection on): mean absolute error over all nets at orders 0 to 5 with the generic dense engine; ``--'' marks orders whose state would exceed 10\\,GB (order $k$ keeps $n^k$ coefficients per live net). Last column: GPU time of order 3.",
              "tab:ablation", ["family", "circuit", "$n$", "$k{=}0$", "$k{=}1$", "$k{=}2$", "$k{=}3$", "$k{=}4$", "$k{=}5$", "$t_3$ (s)"], out_rows, "llrrrrrrrr")

    rows = ext("pareto/e11_rows.json")
    if rows:
        out_rows = []
        for r in sorted(rows, key=lambda r: (r["family"] != "aig", r["circuit"])):
            if r["regime"] != "half": continue
            g = lambda k, d=3: ("--" if r.get(k) is None else (f"{r[k]:.{d}g}" if isinstance(r[k], float) else str(r[k])))
            out_rows.append([r["circuit"], CLS.get(r["class"], r["class"]), g("bdd_runtime", 2), g("k2_err"), g("k2_time", 2), ("--" if r.get("k2_Nstar") is None else f"{r['k2_Nstar']:.0f}"), g("k2_tMCstar", 2), g("k2_ratio", 2), g("mc_err_N10000"), g("mc_err_N1000000")])
        table(out / "breakeven.tex", "Time against accuracy at $p=\\tfrac12$: BDD time (exact), order-2 error and GPU time, the Monte Carlo sample count $N^*(2)$ whose mean error matches order 2 (log-log interpolation of a sweep over $10^3$ to $10^8$ patterns, three seeds), its wall time on the same GPU, and the ratio of the two times; Monte Carlo mean error at $10^4$ and $10^6$ patterns for reference. Class as in Table~\\ref{tab:cv}.",
              "tab:breakeven", ["circuit", "class", "$t_{\\mathrm{BDD}}$", "err $k{=}2$", "$t_2$", "$N^*(2)$", "$t_{\\mathrm{MC}}(N^*)$", "ratio", "MC $10^4$", "MC $10^6$"], out_rows, "llrrrrrrrr")

    rows = ext("cv/e12b_rows.json")
    if rows:
        import statistics as st
        out_rows = []
        for key in sorted({(r["family"], r["circuit"]) for r in rows}, key=lambda x: (x[0] != "aig", x[1])):
            rs = [r for r in rows if (r["family"], r["circuit"]) == key and r["regime"] == "half" and r["N"] == 10 ** 6]
            if not rs: continue
            m = lambda k: (st.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else None)
            f = lambda v, d=3: "--" if v is None else f"{v:.{d}g}"
            out_rows.append([key[1], CLS.get(rs[0]["class"], rs[0]["class"]), f(m("plain_err")), ("--" if m("cv1_err") is None or m("cv1_err") != m("cv1_err") else f(m("cv1_err"))), ("--" if m("vrf1_median") is None else f"{m('vrf1_median'):.2f}"), ("--" if m("pred1_median") is None else f"{m('pred1_median'):.2f}"), f(m("vrf2_median"), 2), f(m("t_cv") / max(m("t_sim"), 1e-9), 2) if m("t_sim") else "--"])
        table(out / "cv.tex", "Monte Carlo with Fourier control variates at $N=10^6$ patterns, $p=\\tfrac12$ (mean over three seeds): mean error of the plain and of the order-1 control-variate estimator over all nets, the measured variance-reduction factor (median over nets), its prediction $1/(1-\\rho_1)$ from the propagated state, the factor at order 2 on the primary outputs, and the cost of the variate relative to the circuit simulation. On c6288 the unprojected order-1 state is not finite on some nets, so the mean error of the control-variate estimator is undefined there and the factor is the median over the finite nets. Class: mult. = c6288, log2, multiplier, sin, sqrt, square; other = the remaining circuits.",
              "tab:cv", ["circuit", "class", "plain", "CV$_1$", "VRF$_1$", "pred.", "VRF$_2$ (out)", "overhead"], out_rows, "llrrrrrr")

    rows = ext("cv/e12a_rows.json")
    if rows:
        out_rows = [[r["circuit"], fmt(r["cut_median"], 2), fmt(r["cert1_median"], 2), fmt(r["cert2_median"], 2), fmt(r["cert2_narrower_N1000"], 2), fmt(r["cert2_narrower_N10000"], 2), fmt(r["cert2_narrower_N100000"], 2), fmt(r["cut_narrower_N10000"], 2)] for r in rows if r["regime"] == "half"]
        table(out / "cert_vs_mc.tex", "Certified intervals against Monte Carlo confidence intervals at $p=\\tfrac12$: median widths of the cutting algorithm and of the order-1 and order-2 certificates, and the fraction of nets on which the order-2 certificate (or the cutting interval) is narrower than the 95\\% Monte Carlo interval after $N$ patterns.",
              "tab:cert_vs_mc", ["circuit", "cut med.", "k1 med.", "k2 med.", "k2 $<$ MC $10^3$", "k2 $<$ MC $10^4$", "k2 $<$ MC $10^5$", "cut $<$ MC $10^4$"], out_rows, "lrrrrrrr")

    rows = ext("openroad/e13_rows.json")
    if rows:
        out_rows = []
        for r in sorted(rows, key=lambda r: (r["platform"], r["design"] not in ("c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"), r["design"])):
            f = lambda k: ("--" if r.get(k) is None else f"{100 * r[k]:+.2f}")
            out_rows.append([r["platform"], r["design"], str(r.get("n_instances", "")), str(r.get("truth", "")), f("k0_total_relerr"), f("k1_total_relerr"), f("k2_total_relerr"), f("k3_total_relerr"), f("default_total_relerr"), f("k0_switching_relerr"), f("k2_switching_relerr"), f("k0_pincap_relerr"), f("k2_pincap_relerr")])
        table(out / "openroad.tex", "Power reported by OpenSTA after placement, routing and parasitic extraction with the OpenROAD flow, with per-net switching activity annotated from the exact probabilities and from orders 0 to 3 (relative error of the total and of the switching component against the exact annotation, in \\%), the error of OpenSTA's own activity propagation from the input ports (``default''), and, for comparison, the error of the pin-capacitance-only estimate of Section~\\ref{sec:experiments}.",
              "tab:openroad", ["platform", "design", "cells", "truth", "$k0$", "$k1$", "$k2$", "$k3$", "default", "sw $k0$", "sw $k2$", "pin-cap $k0$", "pin-cap $k2$"], out_rows, "llrrrrrrrrrrr")

    # ---- tables from the other jobs (sibling directories of the flat dir): timing, justify_hard(r), phase2b
    RR = A.parent
    def sci(v):
        if v is None: return "--"
        if v == 0: return "0"
        if v >= 0.01: return f"{v:.3f}"
        e = int(math.floor(math.log10(v))); return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
    if (RR / "timing/e10_rows.json").exists():
        rows = json.load(open(RR / "timing/e10_rows.json")); rows = rows["rows"] if isinstance(rows, dict) else rows
        g = lambda v: "--" if v is None else f"{v:.3g}"
        out_rows = [[r["circuit"], str(r["n"]), str(r["and"]), str(r["depth"]), g(r.get("k1_cuda0")), g(r.get("k2_cuda0")), g(r.get("k1_numpy")), g(r.get("k2_numpy"))]
                    for r in sorted(rows, key=lambda r: (r["family"] != "aig", r["and"]))]
        table(out / "cost.tex", "Propagation time in seconds (median of three runs, feasibility projection on) on an otherwise idle machine: dense orders 1 and 2 on one RTX~3080 and with the 32-core NumPy backend (NumPy measured only for $n\\le 256$; order 2 skipped for voter, $n=1001$). The GPU engine launches one kernel per gate, so its time is $\\approx 0.07$\\,ms (order 1) and $0.23$\\,ms (order 2) per AND gate until $n$ is large; NumPy is faster below about 150 inputs and the GPU above about 200.",
              "tab:cost", ["circuit", "$n$", "AND", "depth", "GPU $k{=}1$", "GPU $k{=}2$", "NumPy $k{=}1$", "NumPy $k{=}2$"], out_rows, "lrrrrrrr")
    if (RR / "justify_hard/e8_rows.json").exists() and (RR / "justify_hardr/e8_rows.json").exists():
        ld = lambda f: (lambda d: d["rows"] if isinstance(d, dict) else d)(json.load(open(f)))
        h1 = {r["circuit"]: r for r in ld(RR / "justify_hard/e8_rows.json")}; h3 = {r["circuit"]: r for r in ld(RR / "justify_hardr/e8_rows.json")}
        cs = [c for c in h1 if h1[c]["n_hard"] > 0 and c in h3]
        cs.sort(key=lambda c: (h1[c]["family"] != "aig", int(c[1:]) if c[0] == "c" and c[1:].isdigit() else h1[c]["n_hard"], c))
        out_rows = [["ISCAS" if h1[c]["family"] == "aig" else "EPFL", c, str(h1[c]["n_hard"]), sci(h1[c]["hard_mean_half"]), sci(h1[c]["hard_mean_opt"]), sci(h3[c]["hard_mean_opt"]), f"{h3[c]['hard_frac_above_0.1']:.2f}"] for c in cs]
        table(out / "hard_targets.tex", "Probability-guided justification of rare internal nets. For each circuit the 20 rarest $P(\\text{net}=1)$ and 20 rarest $P(\\text{net}=0)$ nets are targets; ``hard'' are those with success probability below $10^{-2}$ per random pattern at $p=\\tfrac12$. Columns give the mean true success probability per pattern (Monte Carlo) at $p=\\tfrac12$, after order-1 gradient ascent from $p=\\tfrac12$ (single start), and after the best of three ascents from random starts $p_i\\sim U(0.3,0.7)$; ``$>0.1$'' is the fraction of hard targets whose success probability exceeds $10^{-1}$ after the restart variant. Circuits without hard targets are omitted.",
              "tab:hard_targets", ["Family", "Circuit", "hard", "at $p=\\frac12$", "single start", "3 restarts", "$>0.1$"], out_rows, "llrrrrr")
    if (RR / "phase2b/phase2_rows.json").exists():
        import statistics as st
        p2b = json.load(open(RR / "phase2b/phase2_rows.json"))["rows"]; ok = [x for x in p2b if x.get("status") == "ok" and x.get("k2_mean")]
        out_rows = []
        for c in sorted({x["circuit"] for x in ok} - {"c17"}, key=lambda c: int(c[1:])):
            rs = [x for x in ok if x["circuit"] == c and x["regime"] in ("rand2", "rand3")]; rat = [x["k1_mean"] / x["k2_mean"] for x in rs]
            out_rows.append([c, str(len(rs)), f"${st.mean(rat):.2f}\\pm{st.stdev(rat) / len(rat) ** 0.5:.2f}$"] + [f"{st.mean(x[f'k{k}_mean'] for x in rs):.4f}" for k in (0, 1, 2)])
        fam_rows = []
        for fam in ("aig", "gate", "resyn2", "dc2"):
            sel = [x for x in ok if x["family"] == fam and x["circuit"] not in ("c17", "c6288") and x["regime"] in ("rand2", "rand3")]
            fam_rows.append([f"all nine, {fam}", str(len(sel)), "--"] + [f"{st.mean(x[f'k{k}_mean'] for x in sel):.4f}" for k in (0, 1, 2)])
        fam_rows[0][0] = "\\midrule\n" + fam_rows[0][0]
        table(out / "seeds.tex", "Seed dependence. For each ISCAS-85 function, the ratio of the mean error at order 1 (BAM) to that at order 2, over its four netlists (raw AIG, gate level, resyn2, dc2) and two name-seeded random input distributions ($p_i\\sim U(0.1,0.9)$); mean $\\pm$ standard error over the eight runs, and the mean errors at orders 0, 1, 2 averaged over the same runs. The last four rows average the nine functions other than c6288 per netlist form.",
              "tab:seeds", ["circuit", "runs", "$k_1/k_2$", "mean $k_0$", "mean $k_1$", "mean $k_2$"], out_rows + fam_rows, "lrrrrr")

    print("tables written to", out)


if __name__ == "__main__":
    main()
