#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Oracle (one-gate, exact inputs) truncation error versus propagated error, and Fourier tail mass by depth.

Circuits: AIGs with <= 24 inputs (c17, EPFL ctrl/dec/cavlc/int2float/sin) and synthetic AND/NOT circuits.
"""
from __future__ import annotations
import argparse, math, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from fast_runner import regime_p  # noqa: E402

REAL = [("aig", "c17", "iscas85/c17.bench"), ("epfl", "ctrl", "epfl_bench/ctrl.bench"), ("epfl", "dec", "epfl_bench/dec.bench"),
        ("epfl", "cavlc", "epfl_bench/cavlc.bench"), ("epfl", "int2float", "epfl_bench/int2float.bench"), ("epfl", "sin", "epfl_bench/sin.bench")]
SYNTH = [(n, g, loc, s) for n in (16,) for g in (200, 800) for loc in (None, 24) for s in range(3)]
TIMEOUTS = {"oracle": 3 * 3600}


def task_list(bench: Path):
    tasks = []
    for fam, c, rel in REAL:
        if (bench / rel).exists():
            for r in ("half", "rand1"):
                tasks.append({"id": f"oracle__{fam}__{c}__{r}", "kind": "oracle", "family": fam, "circuit": c, "path": str(bench / rel), "regime": r})
    for n, g, loc, s in SYNTH:
        c = f"synth_n{n}_g{g}_l{loc or 0}_s{s}"
        tasks.append({"id": f"oracle__synth__{c}__half", "kind": "oracle", "family": "synth", "circuit": c, "regime": "half",
                      "synth": {"n": n, "g": g, "loc": loc, "seed": s}})
    return tasks


def load_circuit(task):
    if task["family"] == "synth":
        from fourprop.synth import random_circuit
        sp = task["synth"]
        return random_circuit(sp["n"], sp["g"], seed=sp["seed"], ops=("AND", "NOT"), max_fanin=2, locality=sp["loc"], name=task["circuit"])
    from fourprop.circuit import load_bench
    return load_bench(task["path"])


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.exactgpu import exact_coeffs_deg2
    from fourprop.dense import propagate_dense
    circ = load_circuit(task)
    p = regime_p(circ.n_inputs, task["regime"], task["circuit"])
    dev = extra.get("device")
    (Path(extra["root"]) / "logs").mkdir(exist_ok=True)
    with open(Path(extra["root"]) / "logs" / "exact.log", "a") as log:
        # batch so that the (batch x nodes) float64 value table stays under ~1.5 GB (sin: 8984 nodes blew a 70 GB allocation at 2^20)
        bb = max(10, min(20, int(math.log2(1.5e9 / (8 * max(1, len(circ.nodes)))))))
        P, A1, A2 = exact_coeffs_deg2(circ, p, device=dev, batch_bits=bb, log=log)
    N = len(circ.nodes)
    tau0 = P - P * P; tau1 = tau0 - (A1 * A1).sum(1); tau2 = tau1 - 0.5 * (A2 * A2).sum((1, 2))
    # depth per node
    depth = np.zeros(N, dtype=int)
    for i, nd in enumerate(circ.nodes):
        if nd.fanins:
            depth[i] = 1 + max(depth[j] for j in nd.fanins)
    # oracle one-gate errors at AND nodes
    and_nodes = [i for i, nd in enumerate(circ.nodes) if nd.op == "AND" and len(nd.fanins) == 2]
    orc = {k: np.full(N, np.nan) for k in (0, 1, 2)}
    for i in and_nodes:
        f, g = circ.nodes[i].fanins
        v0 = P[f] * P[g]; v1 = v0 + A1[f] @ A1[g]; v2 = v1 + 0.5 * (A2[f] * A2[g]).sum()
        for k, v in ((0, v0), (1, v1), (2, v2)):
            if circ.nodes[i].inv:
                v = 1 - v
            orc[k][i] = abs(P[i] - v)
    res = {"n_inputs": circ.n_inputs, "n_nodes": N, "depth": int(depth.max()), "n_and": len(and_nodes), "device": dev}
    prop = {}
    for k in (0, 1, 2):
        for mode in ("norm", "raw"):
            kw = dict(k=k, backend=dev or "numpy")
            if mode == "norm":
                kw["project"] = "norm"
            Pk, rt, _ = propagate_dense(circ, p, **kw)
            prop[(k, mode)] = np.abs(np.clip(Pk, 0, 1) - P)
    am = np.array(and_nodes)
    summ = {}
    for k in (0, 1, 2):
        summ[f"oracle_k{k}_mean"] = float(np.nanmean(orc[k][am])); summ[f"oracle_k{k}_max"] = float(np.nanmax(orc[k][am]))
        for mode in ("norm", "raw"):
            e = prop[(k, mode)]
            summ[f"prop_{mode}_k{k}_mean"] = float(e[am].mean()); summ[f"prop_{mode}_k{k}_max"] = float(e[am].max())
            summ[f"prop_{mode}_k{k}_all_mean"] = float(e.mean())
        summ[f"tau{k}_mean"] = float([tau0, tau1, tau2][k].mean())
    # by depth quartiles (AND nodes)
    qs = np.quantile(depth[am], [0.25, 0.5, 0.75]); bins = np.digitize(depth[am], qs)
    bydepth = []
    for b in range(4):
        sel = am[bins == b]
        if len(sel) == 0:
            continue
        row = {"bin": b, "depth_min": int(depth[sel].min()), "depth_max": int(depth[sel].max()), "n": int(len(sel))}
        for k in (0, 1, 2):
            row[f"oracle_k{k}"] = float(np.nanmean(orc[k][sel])); row[f"prop_k{k}"] = float(prop[(k, "norm")][sel].mean())
            row[f"tau{k}"] = float([tau0, tau1, tau2][k][sel].mean())
        bydepth.append(row)
    res["summary"] = summ; res["by_depth"] = bydepth
    res["per_node"] = {"depth": depth.tolist(), "tau1": tau1.astype(np.float32).tolist(), "tau2": tau2.astype(np.float32).tolist(),
                       "oracle_k1": np.nan_to_num(orc[1], nan=-1).astype(np.float32).tolist(), "oracle_k2": np.nan_to_num(orc[2], nan=-1).astype(np.float32).tolist(),
                       "prop_k1": prop[(1, "norm")].astype(np.float32).tolist(), "prop_k2": prop[(2, "norm")].astype(np.float32).tolist()}
    return res


def write_verdict(root: Path) -> str:
    rows = []
    for f in sorted((root / "results").glob("oracle__*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            rows.append({"id": f.stem, "error": d.get("error", "?")[:60]}); continue
        s = d["summary"]; t = d["task"]
        rows.append({"family": t["family"], "circuit": t["circuit"], "regime": t["regime"], "n_in": d["n_inputs"], "nodes": d["n_nodes"], "depth": d["depth"],
                     **{k: s[k] for k in s}, "by_depth": d["by_depth"]})
    cols = ["family", "circuit", "regime", "n_in", "nodes", "depth", "tau1_mean", "tau2_mean", "oracle_k1_mean", "prop_norm_k1_mean", "oracle_k2_mean",
            "prop_norm_k2_mean", "prop_raw_k2_mean", "oracle_k2_max", "prop_norm_k2_max"]
    L = ["# Oracle vs propagated truncation error (AND nodes)\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols) + " |")
    L.append("\n## By depth quartile (mean over AND nodes in the bin)\n")
    L.append("| circuit | regime | bin | depth range | n | tau1 | tau2 | oracle k1 | prop k1 | oracle k2 | prop k2 |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for b in r.get("by_depth", []):
            L.append(f"| {r['circuit']} | {r['regime']} | {b['bin']} | {b['depth_min']}-{b['depth_max']} | {b['n']} | {b['tau1']:.4g} | {b['tau2']:.4g} | {b['oracle_k1']:.4g} | {b['prop_k1']:.4g} | {b['oracle_k2']:.4g} | {b['prop_k2']:.4g} |")
    txt = "\n".join(L) + "\n"; (root / "E3_RESULTS.md").write_text(txt)
    json.dump(rows, open(root / "e3_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); root = Path(a.root)
    return run_tasks(root, task_list(Path(a.bench)), "oracle_runner", "run_task", {"bench": a.bench, "root": str(root), "device": a.device},
                     TIMEOUTS, workers=1, serial_kinds=("oracle",), on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
