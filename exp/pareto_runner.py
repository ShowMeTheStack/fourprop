#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Time against accuracy. Monte Carlo N-sweep (timed end to end, 3 seeds, both GPUs) + harvest of the propagation
rows from the ablation job (same JOBBASE) and of the BDD / MC-10^8 runtimes of the clean run (read-only).

Verdict: per circuit/regime the Pareto rows and the break-even sample count N*(k) for k = 1..5 (log-log interpolation).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from ext_common import CIRCUITS, REGIMES, regime_p, load_truth, errors, circuit_class  # noqa: E402

NS = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]
SEEDS = [1, 2, 3]
TIMEOUTS = {"mc0": 3 * 3600, "mc1": 3 * 3600, "harvest": 600}


def task_list(bench: Path):
    tasks = []; i = 0
    for fam, c, rel in CIRCUITS:
        if not (bench / rel).exists():
            continue
        for reg in REGIMES:
            tasks.append({"id": f"harvest__{fam}__{c}__{reg}", "kind": "harvest", "family": fam, "circuit": c, "regime": reg, "path": str(bench / rel)})
            for N in NS:
                for s in SEEDS:
                    kind = "mc0" if i % 2 == 0 else "mc1"; i += 1
                    tasks.append({"id": f"mc__{fam}__{c}__{reg}__N{N}__s{s}", "kind": kind, "family": fam, "circuit": c, "regime": reg, "N": N, "seed": s, "path": str(bench / rel)})
    return tasks


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    circ = load_bench(task["path"]); n = circ.n_inputs
    p = regime_p(n, task["regime"], task["circuit"])
    truth, tkind, truth_rt = load_truth(Path(extra["run2"]), task["family"], task["circuit"], task["regime"])
    res = {"n_inputs": n, "n_nodes": len(circ.nodes), "truth": tkind, "truth_runtime": truth_rt, "load_start": os.getloadavg()[0]}
    if task["kind"] == "harvest":
        run2 = Path(extra["run2"]); res["bdd_runtime"] = None; res["mc8_runtime"] = None
        for f in (run2 / "fast" / "results" / f"bdd__{task['circuit']}__{task['regime']}.json",):
            if f.exists():
                d = json.load(open(f))
                if d.get("ok"):
                    res["bdd_runtime"] = float(d["runtime"])
        for f in (run2 / "phase2" / "results" / f"mc__{task['family']}__{task['circuit']}__{task['regime']}.json", run2 / "fast" / "results" / f"mc__{task['circuit']}__{task['regime']}.json"):
            if f.exists():
                d = json.load(open(f))
                if d.get("ok"):
                    res["mc8_runtime"] = float(d["runtime"]); break
        return res
    from fourprop.mc_torch import monte_carlo_torch
    import torch
    dev = "cuda:0" if task["kind"] == "mc0" else "cuda:1"
    monte_carlo_torch(circ, p, n_samples=min(10_000, task["N"]), seed=99, device=dev)   # warm-up (kernels, allocator)
    torch.cuda.synchronize(dev); t0 = time.time()
    P, se = monte_carlo_torch(circ, p, n_samples=task["N"], seed=task["seed"], device=dev)
    torch.cuda.synchronize(dev); rt = time.time() - t0
    res.update({"N": task["N"], "seed": task["seed"], "device": dev, "runtime": rt, "mc_time": rt}); res.update(errors(P, truth))
    res["se_mean"] = float(np.mean(se))
    return res


def _interp_break_even(Ns, errs, target):
    """Smallest N on the (log-log interpolated) MC error curve with error <= target; None if never."""
    pts = sorted(zip(Ns, errs))
    for (N0, e0), (N1, e1) in zip(pts, pts[1:]):
        if e0 <= target:
            return N0
        if e1 <= target < e0 and e1 > 0 and e0 > 0:
            t = (math.log(e0) - math.log(target)) / (math.log(e0) - math.log(e1))
            return 10 ** (math.log10(N0) + t * (math.log10(N1) - math.log10(N0)))
    return pts[-1][0] if pts and pts[-1][1] <= target else None


def write_verdict(root: Path, abl_root: Path, e10_path: Path = None) -> str:
    """e10_path: timing rows (dense.py engine, p = 1/2). Where available they replace the generic engine's
    times for orders 0-2, because the generic order-k engine is 5-20x slower than the specialised dense engine at k <= 2;
    the generic times are kept as k{k}_time_generic."""
    e10 = {}
    if e10_path and Path(e10_path).exists():
        for r in json.load(open(e10_path)):
            e10[r["circuit"]] = r
    mc = {}; harvest = {}
    for f in sorted((root / "results").glob("*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            continue
        t = d["task"]; key = (t["family"], t["circuit"], t["regime"])
        if t["kind"] == "harvest":
            harvest[key] = d
        else:
            mc.setdefault(key, {}).setdefault(t["N"], []).append(d)
    abl = {}
    for f in sorted((abl_root / "results").glob("abl__*.json")):
        d = json.load(open(f))
        if d.get("ok") and d.get("feasible") and not d["task"]["raw"]:
            t = d["task"]; abl.setdefault((t["family"], t["circuit"], t["regime"]), {})[t["k"]] = d
    rows = []
    for key in sorted(set(mc) | set(abl)):
        fam, c, reg = key
        row = {"family": fam, "circuit": c, "regime": reg, "class": circuit_class(c)}
        h = harvest.get(key, {}); row["bdd_runtime"] = h.get("bdd_runtime"); row["mc8_runtime"] = h.get("mc8_runtime"); row["truth"] = h.get("truth")
        Ns = sorted(mc.get(key, {})); errs = []
        for N in Ns:
            rs = mc[key][N]; e = float(np.mean([r["mean_err"] for r in rs])); row[f"mc_err_N{N}"] = e; row[f"mc_err_sd_N{N}"] = float(np.std([r["mean_err"] for r in rs]))
            row[f"mc_time_N{N}"] = float(np.median([r.get("mc_time", r["runtime"]) for r in rs])); errs.append(e)
        for k, d in sorted(abl.get(key, {}).items()):
            row[f"k{k}_err"] = d["mean_err"]; row[f"k{k}_time"] = d["runtime"]; row[f"k{k}_time_generic"] = d["runtime"]
            if reg == "half" and k <= 2 and c in e10 and e10[c].get(f"k{k}_cuda0"):
                row[f"k{k}_time"] = e10[c][f"k{k}_cuda0"]; row[f"k{k}_time_source"] = "dense engine"
            if Ns:
                Nb = _interp_break_even(Ns, errs, d["mean_err"]); row[f"k{k}_Nstar"] = Nb
                if Nb is not None:
                    # time at N*: linear interpolation of time in N (time is linear in N for MC)
                    tN = np.interp(Nb, Ns, [row[f"mc_time_N{N}"] for N in Ns]); row[f"k{k}_tMCstar"] = float(tN); row[f"k{k}_ratio"] = float(tN / row[f"k{k}_time"]) if row[f"k{k}_time"] > 0 else None
        rows.append(row)
    L = ["# Time against accuracy (Monte Carlo sweep, 3 seeds, both GPUs; propagation from the ablation job; BDD/MC8 runtimes from the clean run)\n",
         "| family | circuit | p | class | truth | t_BDD | err k1 | t k1 | N*(1) | err k2 | t k2 | N*(2) | t_MC(N*2) | ratio(2) | err k3 | N*(3) | MC err 1e4 | MC err 1e6 | t_MC 1e6 |", "|---|" * 19]
    for r in rows:
        f = lambda key, d=3: ("" if r.get(key) is None else (f"{r[key]:.{d}g}" if isinstance(r[key], float) else str(r[key])))
        L.append("| " + " | ".join([r["family"], r["circuit"], r["regime"], r["class"], str(r.get("truth")), f("bdd_runtime"), f("k1_err"), f("k1_time"), f("k1_Nstar"), f("k2_err"), f("k2_time"), f("k2_Nstar"), f("k2_tMCstar"), f("k2_ratio"), f("k3_err"), f("k3_Nstar"), f("mc_err_N10000"), f("mc_err_N1000000"), f("mc_time_N1000000")]) + " |")
    txt = "\n".join(L) + "\n"; (root / "E11_RESULTS.md").write_text(txt); json.dump(rows, open(root / "e11_rows.json", "w"), indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--run2", default="/extra/fourprop/run2"); ap.add_argument("--ablation", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); root = Path(a.root)
    tasks = task_list(Path(a.bench))
    # rebalance: the two GPUs differ in speed, so re-split the still-pending MC tasks alternately at every (re)start
    i = 0
    for t in tasks:
        if t["kind"] in ("mc0", "mc1") and not (root / "results" / f"{t['id']}.json").exists():
            t["kind"] = "mc0" if i % 2 == 0 else "mc1"; i += 1
    return run_tasks(root, tasks, "pareto_runner", "run_task", {"run2": a.run2}, TIMEOUTS, workers=a.workers,
                     serial_kinds=("mc0", "mc1"), kind_order={"harvest": 0, "mc0": 1, "mc1": 1}, on_done=lambda r: print(write_verdict(root, Path(a.ablation), Path(a.run2) / "timing" / "e10_rows.json")))


if __name__ == "__main__":
    sys.exit(main())
