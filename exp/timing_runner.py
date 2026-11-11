#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Timings of dense orders 1 and 2 (GPU, and NumPy for n <= 64), 3 repeats, median and min.
The wrapper only starts this when the 1-minute load average is below a threshold, so numbers are measured on a quiet box."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402

CIRCUITS = [("aig", c, f"iscas85/{c}.bench") for c in ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]] + \
           [("epfl", c, f"epfl_bench/{c}.bench") for c in ["adder", "bar", "div", "log2", "max", "multiplier", "sin", "sqrt", "square", "arbiter", "cavlc", "ctrl", "dec", "i2c", "int2float", "priority", "router", "voter"]]
TIMEOUTS = {"timing": 2 * 3600}
REPEATS = 3


def task_list(bench: Path):
    return [{"id": f"timing__{fam}__{c}", "kind": "timing", "family": fam, "circuit": c, "path": str(bench / rel)} for fam, c, rel in CIRCUITS if (bench / rel).exists()]


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    from fourprop.dense import propagate_dense
    circ = load_bench(task["path"]); n = circ.n_inputs; p = np.full(n, 0.5)
    res = {"n_inputs": n, "n_nodes": len(circ.nodes), "n_and": sum(1 for nd in circ.nodes if nd.op == "AND"), "depth": circ.depth(),
           "load_start": os.getloadavg()[0], "times": {}}
    for k in (0, 1, 2):
        if k == 2 and n > 512:
            continue
        for be in ("cuda:0", "numpy"):
            if be == "numpy" and n > 256:   # measure the GPU/NumPy crossover
                continue
            propagate_dense(circ, p, k=k, backend=be, project="norm")   # warm-up
            ts = []
            for _ in range(REPEATS):
                _, rt, _ = propagate_dense(circ, p, k=k, backend=be, project="norm"); ts.append(rt)
            res["times"][f"k{k}_{be.replace(':', '')}"] = {"median": float(np.median(ts)), "min": float(min(ts)), "all": ts}
    res["load_end"] = os.getloadavg()[0]
    return res


def write_verdict(root: Path) -> str:
    rows = []
    for f in sorted((root / "results").glob("timing__*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            continue
        t = d["times"]; r = {"family": d["task"]["family"], "circuit": d["task"]["circuit"], "n": d["n_inputs"], "nodes": d["n_nodes"], "and": d["n_and"], "depth": d["depth"],
                             "load": f"{d['load_start']:.1f}-{d['load_end']:.1f}"}
        for key in ("k0_cuda0", "k1_cuda0", "k2_cuda0", "k1_numpy", "k2_numpy"):
            if key in t:
                r[key] = t[key]["median"]
        if "k1_cuda0" in t and "k2_cuda0" in t:
            r["ratio_cuda"] = t["k2_cuda0"]["median"] / t["k1_cuda0"]["median"]
        rows.append(r)
    # cost model without intercept: t ≈ a·#AND + b·#AND·n^k  (a = per-gate launch overhead, b = per-gate arithmetic)
    X = np.array([[r["and"], r["and"] * r["n"] ** 2] for r in rows if "k2_cuda0" in r]); y = np.array([r["k2_cuda0"] for r in rows if "k2_cuda0" in r])
    fit = np.linalg.lstsq(X, y, rcond=None)[0] if len(y) >= 2 else [np.nan, np.nan]
    X1 = np.array([[r["and"], r["and"] * r["n"]] for r in rows if "k1_cuda0" in r]); y1 = np.array([r["k1_cuda0"] for r in rows if "k1_cuda0" in r])
    fit1 = np.linalg.lstsq(X1, y1, rcond=None)[0] if len(y1) >= 2 else [np.nan, np.nan]
    cols = ["family", "circuit", "n", "nodes", "and", "depth", "load", "k0_cuda0", "k1_cuda0", "k2_cuda0", "ratio_cuda", "k1_numpy", "k2_numpy"]
    loads = [max(float(x) for x in r["load"].split("-")) for r in rows]
    maxload = max(loads) if loads else float("nan")
    # load_start of the first task tells whether the box was otherwise idle; later load values are raised by our own multithreaded NumPy runs
    first_load = float(rows[0]["load"].split("-")[0]) if rows else float("nan")
    title = (f"timings on an otherwise idle box (load {first_load:.1f} at start; later per-task loads include our own NumPy BLAS threads)"
             if first_load < float(os.environ.get("MAXLOAD", "6")) else f"DIRTY timings (load {first_load:.0f} at start; not for the paper)")
    L = [f"# {title} (median of 3, seconds; projection on)\n", f"Cost-model fit: t_k2(GPU) ≈ {fit[0]:.3g}·#AND + {fit[1]:.3g}·(#AND·n²);  t_k1(GPU) ≈ {fit1[0]:.3g}·#AND + {fit1[1]:.3g}·(#AND·n)\n",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols) + " |")
    txt = "\n".join(L) + "\n"; (root / "E10_RESULTS.md").write_text(txt); json.dump(rows, open(root / "e10_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    a = ap.parse_args()
    return run_tasks(Path(a.root), task_list(Path(a.bench)), "timing_runner", "run_task", {"bench": a.bench, "root": a.root}, TIMEOUTS,
                     workers=1, serial_kinds=("timing",), on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
