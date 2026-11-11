#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Order ablation k = 0 … KMAX on all 29 circuits with the generic dense engine (fourprop.dense_k).

One task per (circuit, regime, k[, raw]); feasibility decided from the state-memory estimate before running; infeasible
tasks are recorded as rows with feasible=false (never silently skipped). Each feasible task stores P, rho_k and tau_k per
net (used by the error model and the Pareto plot), mean/max error against the clean-run truth, median-of-3 runtime.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from ext_common import CIRCUITS, REGIMES, regime_p, load_truth, errors, circuit_class  # noqa: E402

KMAX = 5
TIMEOUTS = {"abl": 2 * 3600}
BUDGET = 10 * 1024 ** 3   # bytes of GPU state (float64)
REPEATS = 3


def task_list(bench: Path):
    tasks = []
    for fam, c, rel in CIRCUITS:
        if not (bench / rel).exists():
            continue
        for reg in REGIMES:
            for k in range(KMAX + 1):
                tasks.append({"id": f"abl__{fam}__{c}__{reg}__k{k}", "kind": "abl", "family": fam, "circuit": c, "regime": reg, "k": k, "raw": False, "path": str(bench / rel)})
            if fam == "aig":   # unprojected order 3 on the ISCAS AIGs: does divergence persist at higher order?
                tasks.append({"id": f"abl__{fam}__{c}__{reg}__k3raw", "kind": "abl", "family": fam, "circuit": c, "regime": reg, "k": 3, "raw": True, "path": str(bench / rel)})
    return tasks


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    from fourprop.dense_k import propagate_dense_k, state_bytes, max_live_nets
    circ = load_bench(task["path"]); n = circ.n_inputs; k = task["k"]
    p = regime_p(n, task["regime"], task["circuit"])
    truth, tkind, _ = load_truth(Path(extra["run2"]), task["family"], task["circuit"], task["regime"])
    live = max_live_nets(circ); need = state_bytes(n, k) * live
    res = {"n_inputs": n, "n_nodes": len(circ.nodes), "depth": circ.depth(), "k": k, "raw": task["raw"], "truth": tkind,
           "live_nets": live, "state_bytes": state_bytes(n, k), "est_bytes": need, "feasible": need <= BUDGET, "load_start": os.getloadavg()[0]}
    if not res["feasible"]:
        res["reason"] = f"estimated state memory {need/1e9:.1f} GB > {BUDGET/1e9:.0f} GB"
        return res
    dev = extra.get("device", "cuda:0")
    proj = None if task["raw"] else "norm"
    P, rt, _, diag = propagate_dense_k(circ, p, k=k, backend=dev, project=proj)   # warm-up + first measurement
    times = [rt]
    for _ in range(REPEATS - 1):
        _, rt, _, _ = propagate_dense_k(circ, p, k=k, backend=dev, project=proj); times.append(rt)
    finite = bool(np.all(np.isfinite(P)))
    res.update({"runtime": float(np.median(times)), "runtime_all": times, "finite": finite, "nonfinite": int((~np.isfinite(P)).sum())})
    Pf = np.where(np.isfinite(P), np.clip(P, 0, 1), 0.5)
    res.update(errors(Pf, truth))
    res["rho_mean"] = float(np.nanmean(diag["rho"])); res["tau_mean"] = float(np.nanmean(diag["tau"]))
    res["P"] = [float(x) for x in P]; res["rho"] = [float(x) for x in diag["rho"]]; res["tau"] = [float(x) for x in diag["tau"]]
    return res


def write_verdict(root: Path) -> str:
    rows = []
    for f in sorted((root / "results").glob("abl__*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            rows.append({"family": d["task"]["family"], "circuit": d["task"]["circuit"], "regime": d["task"]["regime"], "k": d["task"]["k"], "raw": d["task"]["raw"], "status": d.get("error", "failed")}); continue
        r = {kk: d[kk] for kk in ("n_inputs", "n_nodes", "depth", "k", "raw", "truth", "live_nets", "est_bytes", "feasible")}
        r.update({"family": d["task"]["family"], "circuit": d["task"]["circuit"], "regime": d["task"]["regime"], "class": circuit_class(d["task"]["circuit"]), "status": "ok"})
        if d["feasible"]:
            r.update({kk: d[kk] for kk in ("runtime", "finite", "nonfinite", "mean_err", "max_err", "p95_err", "rmse", "rho_mean", "tau_mean")})
            if d.get("runtime_all"):
                r["runtime"] = float(np.median(d["runtime_all"]))   # the worker wrapper used to overwrite runtime with the task wall time
        else:
            r["reason"] = d.get("reason", "")
        rows.append(r)
    rows.sort(key=lambda r: (r["family"], r["circuit"], r["regime"], r["raw"], r["k"]))
    # table: circuit × k (mean error, p = 1/2, projected)
    circs = sorted({(r["family"], r["circuit"]) for r in rows}, key=lambda x: (x[0], x[1]))
    L = [f"# Order ablation k = 0..{KMAX} (projection on; mean absolute error vs clean-run truth; '--' = infeasible under {BUDGET/1e9:.0f} GB)\n",
         "| family | circuit | n | class | " + " | ".join(f"k={k}" for k in range(KMAX + 1)) + " | t(k=2) s | t(k=3) s | k3 raw finite |", "|---|---|---|---|" + "---|" * (KMAX + 1) + "---|---|---|"]
    for fam, c in circs:
        cells = []
        for k in range(KMAX + 1):
            r = next((x for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["k"] == k and not x["raw"] and x.get("status") == "ok"), None)
            cells.append("--" if r is None or not r.get("feasible") else f"{r['mean_err']:.4g}")
        t2 = next((x.get("runtime") for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["k"] == 2 and not x["raw"] and x.get("feasible")), None)
        t3 = next((x.get("runtime") for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["k"] == 3 and not x["raw"] and x.get("feasible")), None)
        raw = next((x.get("finite") for x in rows if x["family"] == fam and x["circuit"] == c and x["regime"] == "half" and x["raw"] and x.get("feasible")), None)
        n = next((x["n_inputs"] for x in rows if x["family"] == fam and x["circuit"] == c and "n_inputs" in x), "")
        L.append(f"| {fam} | {c} | {n} | {circuit_class(c)} | " + " | ".join(cells) + f" | {'' if t2 is None else f'{t2:.3g}'} | {'' if t3 is None else f'{t3:.3g}'} | {'' if raw is None else raw} |")
    txt = "\n".join(L) + "\n"; (root / "E15_RESULTS.md").write_text(txt); json.dump(rows, open(root / "e15_rows.json", "w"), indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--run2", default="/extra/fourprop/run2"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(); root = Path(a.root)
    tasks = task_list(Path(a.bench))
    rc = run_tasks(root, tasks, "ablation_runner", "run_task", {"run2": a.run2, "device": a.device}, TIMEOUTS, workers=a.workers,
                   serial_kinds=("abl",), kind_order={"abl": 0}, on_done=lambda r: print(write_verdict(root)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
