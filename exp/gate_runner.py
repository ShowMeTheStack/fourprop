#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Resumable runner for the gate experiment.

Every task writes results/<task_id>.json atomically; tasks whose file exists are skipped, so the
runner can be killed or the machine rebooted at any time and simply re-invoked.

Usage: gate_runner.py --root /extra/fourprop/gate --bench /extra/fourprop/bench/iscas85 [--workers 12]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

from fourprop.circuit import load_bench  # noqa: E402

GATE_CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908"]
REGIMES = {"half": None, "rand1": 1}
ORDERS = {"c17": [0, 1, 2, 3], "default": [0, 1, 2]}
TIMEOUTS = {"bdd": 2 * 3600, "mc": 2 * 3600, "cut": 600, "fp": 10 * 3600}


def regime_p(n: int, regime: str, circuit: str) -> np.ndarray:
    if REGIMES[regime] is None:
        return np.full(n, 0.5)
    seed = REGIMES[regime] * 1000 + sum(map(ord, circuit))
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=n)


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def task_list(bench_dir: Path):
    tasks = []
    for c in GATE_CIRCUITS:
        if not (bench_dir / f"{c}.bench").exists():
            continue
        for r in REGIMES:
            tasks.append({"id": f"bdd__{c}__{r}", "kind": "bdd", "circuit": c, "regime": r})
            tasks.append({"id": f"mc__{c}__{r}", "kind": "mc", "circuit": c, "regime": r})
            for pol in ("reconvergent", "all"):
                tasks.append({"id": f"cut__{c}__{r}__{pol}", "kind": "cut", "circuit": c, "regime": r, "policy": pol})
            for k in ORDERS.get(c, ORDERS["default"]):
                tasks.append({"id": f"fp__{c}__{r}__k{k}", "kind": "fp", "circuit": c, "regime": r, "k": k})
    return tasks


def run_task(task: dict, bench_dir: str, out_path: str, log_path: str) -> None:
    t0 = time.time()
    res = {"task": task, "ok": False}
    try:
        circ = load_bench(str(Path(bench_dir) / f"{task['circuit']}.bench"))
        p = regime_p(circ.n_inputs, task["regime"], task["circuit"])
        res["n_inputs"] = circ.n_inputs; res["n_nodes"] = len(circ.nodes); res["n_gates"] = circ.n_gates
        res["depth"] = circ.depth()
        if task["kind"] == "bdd":
            from fourprop.bdd import bdd_probabilities
            with open(log_path, "a") as log:
                P = bdd_probabilities(circ, p, time_limit=TIMEOUTS["bdd"] - 60, log=log)
            res["P"] = P.tolist()
        elif task["kind"] == "mc":
            try:
                from fourprop.mc_torch import monte_carlo_torch
                P, se = monte_carlo_torch(circ, p, n_samples=100_000_000, batch=4_000_000, seed=7, device="cuda")
                res["device"] = "cuda"
            except Exception as e:  # no GPU: CPU fallback with fewer samples
                from fourprop.mc import monte_carlo
                P, se = monte_carlo(circ, p, n_samples=5_000_000, seed=7)
                res["device"] = f"cpu ({type(e).__name__})"
            res["P"] = P.tolist(); res["se"] = se.tolist()
        elif task["kind"] == "cut":
            from fourprop.cutting import cutting_bounds, reconvergent_stems
            iv = cutting_bounds(circ, p, policy=task["policy"])
            res["lo"] = iv[:, 0].tolist(); res["hi"] = iv[:, 1].tolist()
            res["n_reconvergent_stems"] = len(reconvergent_stems(circ))
        elif task["kind"] == "fp":
            from fourprop.fourier import propagate, tail_mass, variance_explained
            k = task["k"]
            t1 = time.time()
            out = propagate(circ, p, k=k, certify=True, clamp=False)
            res["runtime_propagate"] = time.time() - t1
            res["P"] = [r.P for r in out]
            res["delta"] = [r.delta for r in out]
            res["e"] = [r.e for r in out]
            res["Delta"] = [r.Delta for r in out]
            res["tail"] = [tail_mass(r.state) for r in out]
            res["var_explained"] = [variance_explained(r.state) for r in out]
            res["n_coeffs"] = [len(r.state) for r in out]
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    res["runtime"] = time.time() - t0
    atomic_write(Path(out_path), res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.root); (root / "results").mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    log_path = root / "logs" / "gate_runner.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    if (root / "DONE").exists():
        log("DONE marker present; nothing to do"); return 0
    tasks = task_list(Path(args.bench))
    if not tasks:
        log("no benchmark files found; refusing to run"); return 1
    pending = [t for t in tasks if not (root / "results" / f"{t['id']}.json").exists()]
    log(f"{len(tasks)} tasks, {len(pending)} pending")
    # GPU/MC tasks first, serially (one CUDA context); everything else in worker processes
    order = sorted(pending, key=lambda t: {"mc": 0, "cut": 1, "bdd": 2, "fp": 3}[t["kind"]])
    ctx = mp.get_context("spawn")
    running: list = []
    heavy_running = 0
    while order or running:
        # reap
        still = []
        for proc, t, t0 in running:
            if proc.is_alive():
                if time.time() - t0 > TIMEOUTS[t["kind"]]:
                    proc.terminate(); proc.join(5)
                    atomic_write(root / "results" / f"{t['id']}.json",
                                 {"task": t, "ok": False, "error": "timeout", "runtime": time.time() - t0})
                    log(f"TIMEOUT {t['id']}")
                else:
                    still.append((proc, t, t0))
            else:
                log(f"finished {t['id']} (exit {proc.exitcode}, {time.time()-t0:.0f}s)")
                if proc.exitcode != 0 and not (root / "results" / f"{t['id']}.json").exists():
                    atomic_write(root / "results" / f"{t['id']}.json",
                                 {"task": t, "ok": False, "error": f"worker exit {proc.exitcode}", "runtime": time.time() - t0})
        running = still
        while order and len(running) < args.workers:
            t = order.pop(0)
            if t["kind"] == "mc" and any(r[1]["kind"] == "mc" for r in running):
                order.insert(0, t); break  # one GPU task at a time
            out = root / "results" / f"{t['id']}.json"
            proc = ctx.Process(target=run_task, args=(t, args.bench, str(out), str(log_path)), daemon=False)
            proc.start(); running.append((proc, t, time.time())); log(f"started {t['id']} pid {proc.pid}")
        time.sleep(5)
    log("all tasks finished; computing verdict")
    from gate_verdict import write_verdict
    write_verdict(root)
    (root / "DONE").touch()
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
