#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Fast order-2 experiment: all ISCAS-85 circuits, dense GPU/CPU engines, exact or MC ground truth.

Usage: fast_runner.py --root /extra/fourprop/fast --bench /extra/fourprop/bench/iscas85 --gate /extra/fourprop/gate
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402

CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
REGIMES = {"half": None, "rand1": 1}
TIMEOUTS = {"bdd": 2 * 3600, "mc": 3600, "dense": 4 * 3600, "sparse": 4 * 3600}


def regime_p(n: int, regime: str, circuit: str) -> np.ndarray:
    if REGIMES[regime] is None:
        return np.full(n, 0.5)
    seed = REGIMES[regime] * 1000 + sum(map(ord, circuit))
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=n)


def task_list(bench: Path):
    tasks = []
    for c in CIRCUITS:
        if not (bench / f"{c}.bench").exists():
            continue
        for r in REGIMES:
            tasks.append({"id": f"bdd__{c}__{r}", "kind": "bdd", "circuit": c, "regime": r})
            tasks.append({"id": f"mc__{c}__{r}", "kind": "mc", "circuit": c, "regime": r})
            for k in (1, 2):
                for be in ("cuda:0", "numpy"):
                    tasks.append({"id": f"dense__{c}__{r}__k{k}__{be.replace(':', '')}", "kind": "dense", "circuit": c, "regime": r, "k": k, "backend": be})
            tasks.append({"id": f"sparse__{c}__{r}__k1", "kind": "sparse", "circuit": c, "regime": r, "k": 1})
    return tasks


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    circ = load_bench(str(Path(extra["bench"]) / f"{task['circuit']}.bench"))
    p = regime_p(circ.n_inputs, task["regime"], task["circuit"])
    res = {"n_inputs": circ.n_inputs, "n_nodes": len(circ.nodes), "depth": circ.depth(), "max_cone": max(circ.cone_sizes())}
    if task["kind"] == "bdd":
        from fourprop.bdd import bdd_probabilities
        (Path(extra["root"]) / "logs").mkdir(parents=True, exist_ok=True)
        with open(Path(extra["root"]) / "logs" / "bdd.log", "a") as log:
            P = bdd_probabilities(circ, p, time_limit=TIMEOUTS["bdd"] - 120, log=log)
        res["P"] = P.tolist()
    elif task["kind"] == "mc":
        from fourprop.mc_torch import monte_carlo_torch
        P, se = monte_carlo_torch(circ, p, n_samples=100_000_000, batch=4_000_000, seed=7, device="cuda:1")
        res["P"] = P.tolist(); res["se"] = se.tolist(); res["device"] = "cuda:1"
    elif task["kind"] == "dense":
        from fourprop.dense import propagate_dense
        # warm-up run (CUDA context / allocator), then timed run
        propagate_dense(circ, p, k=task["k"], backend=task["backend"])
        P, rt, _ = propagate_dense(circ, p, k=task["k"], backend=task["backend"])
        res["P"] = P.tolist(); res["runtime_propagate"] = rt
    elif task["kind"] == "sparse":
        from fourprop.fourier import propagate
        t = time.time(); out = propagate(circ, p, k=task["k"], clamp=False)
        res["runtime_propagate"] = time.time() - t; res["P"] = [r.P for r in out]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--gate", default=None, help="reuse bdd/mc results from the gate experiment")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root); (root / "results").mkdir(parents=True, exist_ok=True)
    if a.gate:
        for f in Path(a.gate).glob("results/*.json"):
            if f.name.startswith(("bdd__", "mc__")) and not (root / "results" / f.name).exists():
                d = json.load(open(f))
                if d.get("ok"):
                    shutil.copy(f, root / "results" / f.name)
    from fast_verdict import write_verdict
    return run_tasks(root, task_list(Path(a.bench)), "fast_runner", "run_task", {"bench": a.bench, "root": str(root)},
                     TIMEOUTS, a.workers, serial_kinds=("mc", "dense"), kind_order={"mc": 0, "dense": 1, "sparse": 2, "bdd": 3},
                     on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
