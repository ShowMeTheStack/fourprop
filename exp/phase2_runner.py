#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Benchmark experiment: ISCAS-85 (AIG, original gate-level, resyn2, dc2) and EPFL; orders 0/1/2 with
projection, certified order-1 bounds, MC (and BDD where feasible) ground truth."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402

ISCAS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
EPFL = ["adder", "bar", "div", "log2", "max", "multiplier", "sin", "sqrt", "square", "arbiter", "cavlc", "ctrl", "dec",
        "i2c", "int2float", "priority", "router", "voter"]
REGIMES = {"half": None, "rand1": 1}
NAMED_REGIMES = {"rand2": 2, "rand3": 3}   # p drawn per (circuit, input NAME): comparable across netlists
TIMEOUTS = {"bdd": 3600, "mc": 3 * 3600, "dense": 3 * 3600}
K2_MAX_INPUTS = 512


def regime_p(n, regime, circuit, names=None):
    if regime in NAMED_REGIMES:
        assert names is not None and len(names) == n
        seed = NAMED_REGIMES[regime]
        return np.array([np.random.default_rng(seed * 100003 + sum((i + 1) * ord(ch) for i, ch in enumerate(f"{circuit}/{nm}"))).uniform(0.1, 0.9) for nm in names])
    if REGIMES[regime] is None:
        return np.full(n, 0.5)
    return np.random.default_rng(REGIMES[regime] * 1000 + sum(map(ord, circuit))).uniform(0.1, 0.9, size=n)


def circuits(bench: Path):
    out = []  # (family, name, path)
    for c in ISCAS:
        out.append(("aig", c, bench / "iscas85" / f"{c}.bench"))
        out.append(("gate", c, bench / "iscas85_src" / "ISCAS85" / c / f"{c}.v"))
        if c != "c17":
            out.append(("resyn2", c, bench / "iscas85_var" / f"{c}_resyn2.bench"))
            out.append(("dc2", c, bench / "iscas85_var" / f"{c}_dc2.bench"))
    for e in EPFL:
        out.append(("epfl", e, bench / "epfl_bench" / f"{e}.bench"))
    return [x for x in out if x[2].exists()]


def load(path: Path):
    if path.suffix == ".v":
        from fourprop.verilog import load_verilog
        return load_verilog(str(path))
    from fourprop.circuit import load_bench
    return load_bench(str(path))


def task_list(bench: Path, regimes=None, families=None):
    tasks = []
    regimes = regimes or list(REGIMES)
    for fam, c, path in circuits(bench):
        if families and fam not in families:
            continue
        n_in = sum(1 for line in open(path) if line.startswith("INPUT(")) if path.suffix == ".bench" else 0
        for r in regimes:
            base = {"family": fam, "circuit": c, "path": str(path), "regime": r}
            tasks.append({"id": f"mc__{fam}__{c}__{r}", "kind": "mc", **base})
            if fam in ("gate", "resyn2", "dc2") and c != "c6288":
                tasks.append({"id": f"bdd__{fam}__{c}__{r}", "kind": "bdd", **base})
            for k in (0, 1, 2):
                if k == 2 and n_in > K2_MAX_INPUTS:
                    continue
                tasks.append({"id": f"dense__{fam}__{c}__{r}__k{k}", "kind": "dense", "k": k, "mode": "norm", **base})
            tasks.append({"id": f"cert__{fam}__{c}__{r}__k1", "kind": "dense", "k": 1, "mode": "cert", **base})
    return tasks


def run_task(task: dict, extra: dict) -> dict:
    circ = load(Path(task["path"]))
    p = regime_p(circ.n_inputs, task["regime"], task["circuit"], names=[circ.nodes[i].name for i in circ.inputs])
    res = {"n_inputs": circ.n_inputs, "n_nodes": len(circ.nodes), "depth": circ.depth(),
           "outputs": [circ.nodes[i].name for i in circ.outputs], "output_ids": circ.outputs}
    if task["kind"] == "mc":
        from fourprop.mc_torch import monte_carlo_torch
        P, se = monte_carlo_torch(circ, p, n_samples=100_000_000, batch=4_000_000, seed=7, device="cuda:1")
        res["P"] = P.tolist(); res["se"] = se.tolist()
    elif task["kind"] == "bdd":
        from fourprop.bdd import bdd_probabilities
        (Path(extra["root"]) / "logs").mkdir(exist_ok=True)
        with open(Path(extra["root"]) / "logs" / "bdd.log", "a") as log:
            res["P"] = bdd_probabilities(circ, p, time_limit=TIMEOUTS["bdd"] - 120, log=log).tolist()
    else:
        from fourprop.dense import propagate_dense
        if task["mode"] == "cert":
            propagate_dense(circ, p, k=1, backend="cuda:0", certify=True)
            P, rt, _, E, D = propagate_dense(circ, p, k=1, backend="cuda:0", certify=True)
            res["P"] = P.tolist(); res["e"] = E.tolist(); res["delta"] = D.tolist(); res["runtime_propagate"] = rt
        else:
            kw = dict(k=task["k"], backend="cuda:0", project="norm")
            propagate_dense(circ, p, **kw)
            P, rt, _ = propagate_dense(circ, p, **kw)
            res["P"] = P.tolist(); res["runtime_propagate"] = rt
    return res


def write_verdict(root: Path) -> str:
    R = root / "results"

    def get(tid):
        f = R / f"{tid}.json"
        if not f.exists():
            return None
        d = json.load(open(f)); return d if d.get("ok") else None

    rows = []
    for fam, c, path in circuits(Path(root / "bench_link").resolve()) if (root / "bench_link").exists() else []:
        pass
    ids = sorted({f.name.split("__")[1] + "__" + f.name.split("__")[2] for f in R.glob("dense__*.json")})
    for key in ids:
        fam, c = key.split("__")
        for r in ("half", "rand1", "rand2", "rand3"):
            t = get(f"bdd__{fam}__{c}__{r}") or get(f"mc__{fam}__{c}__{r}")
            if not t:
                rows.append({"family": fam, "circuit": c, "regime": r, "status": "no truth"}); continue
            ex = np.array(t["P"]); tol = 4 * np.array(t["se"]) if "se" in t else 0.0
            row = {"family": fam, "circuit": c, "regime": r, "truth": "bdd" if get(f"bdd__{fam}__{c}__{r}") else "mc",
                                        "n_inputs": t["n_inputs"], "n_nodes": t["n_nodes"], "depth": t["depth"], "status": "ok"}
            for k in (0, 1, 2):
                d = get(f"dense__{fam}__{c}__{r}__k{k}")
                if not d:
                    continue
                err = np.abs(np.clip(np.array(d["P"]), 0, 1) - ex)
                row[f"k{k}_mean"] = float(err.mean()); row[f"k{k}_p95"] = float(np.quantile(err, .95)); row[f"k{k}_max"] = float(err.max()); row[f"k{k}_rt"] = float(d["runtime_propagate"])
                oerr = err[t["output_ids"]]; row[f"k{k}_out_mean"] = float(oerr.mean())
            cd = get(f"cert__{fam}__{c}__{r}__k1")
            if cd:
                e = np.array(cd["e"]); P = np.array(cd["P"]); w = np.minimum(P + e, 1) - np.maximum(P - e, 0)
                row["cert_sound"] = bool(np.all(np.abs(P - ex) <= e + tol + 1e-9)); row["cert_median_w"] = float(np.median(w))
                row["cert_lt_0.1"] = float(np.mean(w < 0.1)); row["cert_lt_0.5"] = float(np.mean(w < 0.5))
            rows.append(row)
    # structure dependence: same ISCAS function, four netlists; compare output estimates by name
    struct = []
    for c in ISCAS:
        for r in ("half", "rand2", "rand3"):  # rand1 draws p by input position (not comparable across netlists); rand2/3 are name-seeded
            for k in (1, 2):
                ests = {}
                for fam in ("aig", "gate", "resyn2", "dc2"):
                    d = get(f"dense__{fam}__{c}__{r}__k{k}")
                    if d:
                        ests[fam] = dict(zip(d["outputs"], np.clip(np.array(d["P"])[d["output_ids"]], 0, 1)))
                if len(ests) < 2:
                    continue
                names = set.intersection(*(set(v) for v in ests.values()))
                if not names:
                    continue
                M = np.array([[ests[f][nm] for nm in sorted(names)] for f in ests])
                spread = M.max(0) - M.min(0)
                t = get(f"mc__aig__{c}__{r}")
                struct.append({"circuit": c, "regime": r, "k": k, "variants": list(ests), "n_outputs": len(names),
                               "spread_mean": float(spread.mean()), "spread_max": float(spread.max())})
    cols = ["family", "circuit", "regime", "truth", "n_inputs", "n_nodes", "depth", "k0_mean", "k1_mean", "k2_mean", "k1_max", "k2_max",
            "k1_out_mean", "k2_out_mean", "k1_rt", "k2_rt", "cert_sound", "cert_median_w", "cert_lt_0.1", "cert_lt_0.5", "status"]
    L = ["# Benchmark results\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        L.append("| " + " | ".join(f"{row.get(x, ''):.4g}" if isinstance(row.get(x, ""), float) else str(row.get(x, "")) for x in cols) + " |")
    L.append("\n## Structure dependence (max−min of output estimates across netlists of the same function)\n")
    L.append("| circuit | regime | k | variants | n_outputs | spread_mean | spread_max |"); L.append("|---|---|---|---|---|---|---|")
    for s_ in struct:
        L.append(f"| {s_['circuit']} | {s_['regime']} | {s_['k']} | {','.join(s_['variants'])} | {s_['n_outputs']} | {s_['spread_mean']:.4g} | {s_['spread_max']:.4g} |")
    txt = "\n".join(L) + "\n"; (root / "PHASE2_RESULTS.md").write_text(txt)
    json.dump({"rows": rows, "struct": struct}, open(root / "phase2_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--regimes", default=None); ap.add_argument("--families", default=None)
    a = ap.parse_args(); root = Path(a.root)
    regs = a.regimes.split(",") if a.regimes else None; fams = a.families.split(",") if a.families else None
    return run_tasks(root, task_list(Path(a.bench), regs, fams), "phase2_runner", "run_task", {"bench": a.bench, "root": str(root)},
                     TIMEOUTS, a.workers, serial_kinds=("mc", "dense"), kind_order={"dense": 0, "mc": 1, "bdd": 2}, on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
