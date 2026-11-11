#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""E14B: an a-priori model of the propagated error, checked against measurement.

M1 (signed, first order; needs the exact coefficients, so the six enumerable circuits):
    AND: eps_H = P_G eps_F + P_F eps_G + eps0_H,   eps0_H = sum_{|S|<=k} f^(S) g^(S) - P_{fg}   (one-gate oracle error)
    NOT: eps = -eps_in;  OR = NOT(AND(NOT,NOT));  XOR: eps_F + eps_G - 2 eps_AND.
    Orders 1 and 2 (exact coefficients up to degree 2 from fourprop.exactgpu).
M2 (a priori; needs only the propagated state, so all 29 circuits):
    second moments:  E_H = P_G^2 E_F + P_F^2 E_G + c_k^2 tau_F tau_G,   predicted |eps_H| = sqrt(E_H),
    tau from the state (P(1-P) - sum of held squares), one constant c_k per order fitted on log scale.
Measured propagated errors come from the ablation job (same JOBBASE).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from ext_common import CIRCUITS, REGIMES, SMALL_EXACT, regime_p, load_truth, circuit_class  # noqa: E402

TIMEOUTS = {"m1": 3600, "m2": 1800}
KS_M1 = (1, 2); KS_M2 = (1, 2, 3)


def task_list(bench: Path):
    tasks = []
    for fam, c, rel in CIRCUITS:
        if not (bench / rel).exists():
            continue
        for reg in REGIMES:
            if c in SMALL_EXACT:
                tasks.append({"id": f"m1__{fam}__{c}__{reg}", "kind": "m1", "family": fam, "circuit": c, "regime": reg, "path": str(bench / rel)})
            tasks.append({"id": f"m2__{fam}__{c}__{reg}", "kind": "m2", "family": fam, "circuit": c, "regime": reg, "path": str(bench / rel)})
    return tasks


def _abl(root_abl: Path, fam, c, reg, k):
    f = root_abl / "results" / f"abl__{fam}__{c}__{reg}__k{k}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d if d.get("ok") and d.get("feasible") else None


def _propagate_signed(circ, P_in, eps0_and, eps_src):
    """M1 recursion. eps0_and[i] = oracle error of node i if it is an AND (signed, for the chained 2-input form); zero otherwise."""
    N = len(circ.nodes); eps = np.zeros(N)
    for i, nd in enumerate(circ.nodes):
        if nd.op in ("INPUT", "CONST0", "CONST1"):
            eps[i] = 0.0
        elif nd.op == "BUF":
            eps[i] = eps[nd.fanins[0]]
        elif nd.op == "NOT":
            eps[i] = -eps[nd.fanins[0]]
        else:
            f = nd.fanins[0]; e = eps[f]; Pf = P_in[f]
            for j in nd.fanins[1:]:
                g = j; eg = eps[g]; Pg = P_in[g]
                if nd.op == "AND":
                    e = Pg * e + Pf * eg + eps_src(i, f, g, "AND"); Pf = Pf * Pg
                elif nd.op == "OR":
                    e = (1 - Pg) * e + (1 - Pf) * eg + eps_src(i, f, g, "OR"); Pf = Pf + Pg - Pf * Pg
                else:
                    ea = Pg * e + Pf * eg + eps_src(i, f, g, "AND"); e = e + eg - 2 * ea; Pf = Pf + Pg - 2 * Pf * Pg
            eps[i] = e
        if nd.inv:
            eps[i] = -eps[i]
    return eps


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    circ = load_bench(task["path"]); n = circ.n_inputs; N = len(circ.nodes)
    p = regime_p(n, task["regime"], task["circuit"])
    truth, tkind, _ = load_truth(Path(extra["run2"]), task["family"], task["circuit"], task["regime"])
    res = {"n_inputs": n, "n_nodes": N, "truth": tkind, "load_start": os.getloadavg()[0]}
    abl_root = Path(extra["ablation"])
    if task["kind"] == "m1":
        from fourprop.exactgpu import exact_coeffs_deg2
        import math
        bb = max(10, min(20, int(math.log2(1.5e9 / (8 * max(1, N))))))
        Pex, A1, A2 = exact_coeffs_deg2(circ, p, device=extra.get("device", "cuda:0"), batch_bits=bb)
        Pex = np.asarray(Pex); A1 = np.asarray(A1); A2 = np.asarray(A2)
        # exact value of a product of two nets f, g up to degree k: sum of coefficient products; exact P_fg needs the pair
        # value — we take it from the exact probability of the gate output when the gate is a plain 2-input AND/OR of f, g
        # (true for AIGs and for the chained first pair); for chains the oracle error of later pairs is approximated by the
        # first-order rule, which the runner records as 'chain_approx'.
        out = {}
        # all benchmark circuits here are AIGs: every gate is a 2-input AND with an optional output inversion
        for i, nd in enumerate(circ.nodes):
            if nd.op not in ("INPUT", "CONST0", "CONST1", "BUF", "NOT", "AND") or (nd.op == "AND" and len(nd.fanins) != 2):
                res["m1"] = {"status": f"non-AIG gate {nd.op}/{len(nd.fanins)} at node {i}"}
                return res
        for k in KS_M1:
            def eps_src(i, f, g, op, k=k):
                v = Pex[f] * Pex[g] + (A1[f] @ A1[g] if k >= 1 else 0.0) + (0.5 * np.sum(A2[f] * A2[g]) if k >= 2 else 0.0)
                exact = 1.0 - Pex[i] if circ.nodes[i].inv else Pex[i]
                return float(v - exact)
            d = _abl(abl_root, task["family"], task["circuit"], task["regime"], k)
            if d is None:
                out[f"k{k}"] = {"status": "no ablation row"}; continue
            Pprop = np.array(d["P"]); meas = Pprop - truth
            pred = _propagate_signed(circ, Pex, None, eps_src)
            good = np.isfinite(meas) & np.isfinite(pred)
            m, q = meas[good], pred[good]
            corr = float(np.corrcoef(m, q)[0, 1]) if m.std() > 0 and q.std() > 0 else float("nan")
            slope = float(np.dot(m, q) / max(np.dot(q, q), 1e-18))
            out[f"k{k}"] = {"status": "ok", "corr": corr, "slope": slope, "mean_abs_meas": float(np.abs(m).mean()), "mean_abs_pred": float(np.abs(q).mean()),
                            "ratio_mean": float(np.abs(q).mean() / max(np.abs(m).mean(), 1e-18)), "r2": float(1 - np.sum((m - q) ** 2) / max(np.sum((m - m.mean()) ** 2), 1e-18))}
        res["m1"] = out
        return res
    # M2: per order, second-moment recursion with the state's tau; fit of c_k happens in the verdict (store the two sums)
    out = {}
    for k in KS_M2:
        d = _abl(abl_root, task["family"], task["circuit"], task["regime"], k)
        if d is None:
            out[f"k{k}"] = {"status": "infeasible"}; continue
        Pp = np.clip(np.array(d["P"]), 0, 1); tau = np.maximum(np.array(d["tau"]), 0.0); meas = np.abs(Pp - truth)
        # two recursions: A (source terms only, with c=1) and B (no source) are linear in c^2: E = c^2 * S where S is the
        # source-only recursion, because inputs have E = 0. So E_H(c) = c^2 S_H and predicted |eps| = c * sqrt(S).
        S = np.zeros(N)
        for i, nd in enumerate(circ.nodes):
            if nd.op in ("INPUT", "CONST0", "CONST1"):
                S[i] = 0.0
            elif nd.op in ("BUF", "NOT"):
                S[i] = S[nd.fanins[0]]
            else:
                f = nd.fanins[0]; s = S[f]; Pf = Pp[f]; tf = tau[f]
                for j in nd.fanins[1:]:
                    g = j; Pg = Pp[g]; tg = tau[g]
                    if nd.op == "AND":
                        s = Pg ** 2 * s + Pf ** 2 * S[g] + tf * tg; Pf = Pf * Pg
                    elif nd.op == "OR":
                        s = (1 - Pg) ** 2 * s + (1 - Pf) ** 2 * S[g] + tf * tg; Pf = Pf + Pg - Pf * Pg
                    else:
                        sa = Pg ** 2 * s + Pf ** 2 * S[g] + tf * tg; s = s + S[g] + 4 * sa; Pf = Pf + Pg - 2 * Pf * Pg
                S[i] = s
        pred_unit = np.sqrt(np.maximum(S, 0))
        good = (pred_unit > 0) & (meas > 0)
        out[f"k{k}"] = {"status": "ok", "mean_meas": float(meas.mean()), "mean_pred_unit": float(pred_unit.mean()),
                        "log_ratio_mean": float(np.mean(np.log(meas[good]) - np.log(pred_unit[good]))) if good.any() else None,
                        "corr_log": float(np.corrcoef(np.log(meas[good]), np.log(pred_unit[good]))[0, 1]) if good.sum() > 2 else None, "n_good": int(good.sum())}
    res["m2"] = out
    return res


def write_verdict(root: Path) -> str:
    rows = []
    for f in sorted((root / "results").glob("*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            continue
        t = d["task"]
        for kind in ("m1", "m2"):
            if kind in d:
                for kk, v in d[kind].items():
                    rows.append({"model": kind, "family": t["family"], "circuit": t["circuit"], "regime": t["regime"], "class": circuit_class(t["circuit"]), "k": int(kk[1:]), **v})
    # fit c_k for M2 per order (geometric mean of meas/pred over circuits, p = 1/2 and rand1 both)
    fits = {}
    for k in KS_M2:
        lr = [r["log_ratio_mean"] for r in rows if r["model"] == "m2" and r["k"] == k and r.get("status") == "ok" and r.get("log_ratio_mean") is not None]
        fits[k] = float(np.exp(np.mean(lr))) if lr else None
    for r in rows:
        if r["model"] == "m2" and r.get("status") == "ok" and fits.get(r["k"]):
            r["c_k"] = fits[r["k"]]; r["mean_pred"] = r["mean_pred_unit"] * fits[r["k"]]; r["ratio"] = r["mean_pred"] / max(r["mean_meas"], 1e-18)
    L = ["# E14B: error model\n", f"Fitted constants c_k (M2): {fits}\n", "## M1 (signed first-order recursion with exact oracle errors), per net\n",
         "| circuit | p | k | corr | slope | R2 | mean|meas| | mean|pred| |", "|---|" * 8]
    for r in sorted([r for r in rows if r["model"] == "m1" and r.get("status") == "ok"], key=lambda r: (r["circuit"], r["regime"], r["k"])):
        L.append(f"| {r['circuit']} | {r['regime']} | {r['k']} | {r['corr']:.3f} | {r['slope']:.3f} | {r['r2']:.3f} | {r['mean_abs_meas']:.3g} | {r['mean_abs_pred']:.3g} |")
    L += ["\n## M2 (a priori, state only), per circuit: measured vs predicted mean error (c_k fitted globally)\n", "| family | circuit | p | class | k | measured | predicted | ratio | corr(log) per net |", "|---|" * 9]
    for r in sorted([r for r in rows if r["model"] == "m2" and r.get("status") == "ok"], key=lambda r: (r["family"], r["circuit"], r["regime"], r["k"])):
        L.append(f"| {r['family']} | {r['circuit']} | {r['regime']} | {r['class']} | {r['k']} | {r['mean_meas']:.3g} | {r.get('mean_pred', float('nan')):.3g} | {r.get('ratio', float('nan')):.2f} | {'' if r.get('corr_log') is None else f'{r['corr_log']:.2f}'} |")
    txt = "\n".join(L) + "\n"; (root / "E14_RESULTS.md").write_text(txt); json.dump({"rows": rows, "fits": fits}, open(root / "e14_rows.json", "w"), indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--run2", default="/extra/fourprop/run2"); ap.add_argument("--ablation", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); root = Path(a.root)
    return run_tasks(root, task_list(Path(a.bench)), "errmodel_runner", "run_task", {"run2": a.run2, "ablation": a.ablation, "device": a.device}, TIMEOUTS, workers=a.workers,
                     serial_kinds=("m1",), kind_order={"m2": 0, "m1": 1}, on_done=lambda r: print(write_verdict(root)))


if __name__ == "__main__":
    sys.exit(main())
