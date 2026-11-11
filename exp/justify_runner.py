#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Probability-guided justification on ISCAS-85 AIGs (+ EPFL control blocks).

For every primary output and target value v in {0,1}: run gradient ascent on the order-1 estimate from p=½,
then measure the TRUE success probability per random pattern at the optimised p (GPU Monte Carlo) and at p=½,
and whether the certified order-1 bound proves P > 0 (satisfiability certificate) at the optimised p.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402

CIRCUITS = [("aig", c, f"iscas85/{c}.bench") for c in ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]] + \
           [("epfl", c, f"epfl_bench/{c}.bench") for c in ["ctrl", "cavlc", "int2float", "router", "i2c", "priority"]]
TIMEOUTS = {"justify": 4 * 3600}


def task_list(bench: Path, targets: str = "outputs"):
    tag = "" if targets == "outputs" else f"__{targets}"
    return [{"id": f"justify__{fam}__{c}{tag}", "kind": "justify", "family": fam, "circuit": c, "path": str(bench / rel), "targets": targets}
            for fam, c, rel in CIRCUITS if (bench / rel).exists()]


def run_task(task: dict, extra: dict) -> dict:
    import torch
    from fourprop.circuit import load_bench
    from fourprop.justify import ascend
    from fourprop.dense import propagate_dense
    from fourprop.mc_torch import monte_carlo_torch
    circ = load_bench(task["path"]); n = circ.n_inputs
    p_half = np.full(n, 0.5)
    P_half_mc, _ = monte_carlo_torch(circ, p_half, n_samples=20_000_000, batch=4_000_000, seed=1, device="cuda:0")
    rows = []
    t0 = time.time()
    mode = task.get("targets", "outputs")
    rng = np.random.default_rng(11)
    if mode in ("hard", "hard_rand"):
        # the 20 internal nets with the smallest nonzero P(=1) and the 20 with the smallest nonzero P(=0) at p=½
        cand = [i for i, nd in enumerate(circ.nodes) if nd.op not in ("INPUT", "CONST0", "CONST1", "BUF")]
        lo1 = sorted((i for i in cand if P_half_mc[i] > 0), key=lambda i: P_half_mc[i])[:20]
        lo0 = sorted((i for i in cand if P_half_mc[i] < 1), key=lambda i: 1 - P_half_mc[i])[:20]
        pairs = [(i, 1) for i in lo1] + [(i, 0) for i in lo0]
    else:
        pairs = [(out, v) for out in circ.outputs for v in (0, 1)]
    for out, v in pairs:
        if True:
            t1 = time.time()
            if mode == "hard_rand":   # best of 3 random starts (p=½ is a saddle for parity-like targets)
                best = None
                for _ in range(3):
                    p0 = rng.uniform(0.3, 0.7, size=n)
                    cand, tr = ascend(circ, out, v, p0, steps=20, eta=0.25, backend="cuda:0")
                    if best is None or tr[-1] > best[1][-1]:
                        best = (cand, tr)
                p_opt, trace = best
            else:
                p_opt, trace = ascend(circ, out, v, p_half, steps=20, eta=0.25, backend="cuda:0")
            t_asc = time.time() - t1
            Pm, _ = monte_carlo_torch(circ, p_opt, n_samples=20_000_000, batch=4_000_000, seed=2, device="cuda:0")
            true_opt = float(Pm[out] if v == 1 else 1 - Pm[out]); true_half = float(P_half_mc[out] if v == 1 else 1 - P_half_mc[out])
            # certificate at p_opt (raw order-1 certified run)
            Pc, _, _, E, _ = propagate_dense(circ, p_opt, k=1, backend="cuda:0", certify=True)
            est = Pc[out] if v == 1 else 1 - Pc[out]
            rows.append({"output": circ.nodes[out].name, "value": v, "true_half": true_half, "true_opt": true_opt, "est_opt": float(est),
                         "cert": bool(est - E[out] > 0), "cert_lower": float(est - E[out]), "steps": len(trace), "time": t_asc})
    rows_arr = rows
    th = np.array([r["true_half"] for r in rows]); to = np.array([r["true_opt"] for r in rows])
    gain = (to + 1e-9) / (th + 1e-9)
    hard = th < 0.01
    summ = {"n_targets": len(rows), "median_gain": float(np.median(gain)), "mean_true_half": float(th.mean()), "mean_true_opt": float(to.mean()),
            "frac_improved": float(np.mean(to > th + 1e-6)), "frac_worse": float(np.mean(to < th - 1e-6)),
            "n_hard": int(hard.sum()), "hard_mean_half": float(th[hard].mean()) if hard.any() else None, "hard_mean_opt": float(to[hard].mean()) if hard.any() else None,
            "hard_frac_above_0.1": float(np.mean(to[hard] > 0.1)) if hard.any() else None,
            "frac_certified": float(np.mean([r["cert"] for r in rows])), "frac_unsat_like": float(np.mean(th == 0)),
            "total_time": time.time() - t0, "n_inputs": n, "n_outputs": len(circ.outputs)}
    return {"summary": summ, "rows": rows_arr}


def write_verdict(root: Path) -> str:
    rows = []
    for f in sorted((root / "results").glob("justify__*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            rows.append({"circuit": f.stem, "error": d.get("error", "?")[:60]}); continue
        rows.append({"family": d["task"]["family"], "circuit": d["task"]["circuit"], **d["summary"]})
    cols = ["family", "circuit", "n_inputs", "n_outputs", "n_targets", "mean_true_half", "mean_true_opt", "median_gain", "frac_improved", "frac_worse",
            "n_hard", "hard_mean_half", "hard_mean_opt", "hard_frac_above_0.1", "frac_certified", "total_time"]
    L = ["# Probability-guided justification (order-1 gradient ascent from p=½; true success probability per pattern by MC)\n",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols) + " |")
    txt = "\n".join(L) + "\n"; (root / "E8_RESULTS.md").write_text(txt); json.dump(rows, open(root / "e8_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--targets", default="outputs")
    a = ap.parse_args()
    return run_tasks(Path(a.root), task_list(Path(a.bench), a.targets), "justify_runner", "run_task", {"bench": a.bench, "root": a.root}, TIMEOUTS,
                     workers=1, serial_kinds=("justify",), on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
