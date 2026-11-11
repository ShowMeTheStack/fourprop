#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Total dynamic power error with real cell libraries.

For each ISCAS-85 gate-level netlist and library (sky130_fd_sc_hd, NanGate45): map with Yosys/ABC, expand cell
functions, estimate per-net switching activity alpha = 2P(1-P) at orders 0/1/2 (projection on), and compare
total sum_net alpha*C_load against exact (BDD) or Monte Carlo truth.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from fast_runner import regime_p  # noqa: E402

ISCAS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
LIBS = {"sky130": "liberty/sky130_fd_sc_hd__tt_025C_1v80.lib", "nangate45": "liberty/NangateOpenCellLibrary_typical.lib"}
TIMEOUTS = {"map": 1800, "mc": 3600, "bdd": 3600, "dense": 3600}


def task_list(bench: Path):
    tasks = []
    for c in ISCAS:
        src = bench / "iscas85_src" / "ISCAS85" / c / f"{c}.v"
        if not src.exists():
            continue
        for lib in LIBS:
            base = {"circuit": c, "lib": lib, "src": str(src), "libfile": str(bench / LIBS[lib])}
            tasks.append({"id": f"map__{c}__{lib}", "kind": "map", **base})
            for r in ("half", "rand1"):
                tasks.append({"id": f"mc__{c}__{lib}__{r}", "kind": "mc", "regime": r, **base})
                if c != "c6288":
                    tasks.append({"id": f"bdd__{c}__{lib}__{r}", "kind": "bdd", "regime": r, **base})
                for k in (0, 1, 2):
                    tasks.append({"id": f"dense__{c}__{lib}__{r}__k{k}", "kind": "dense", "regime": r, "k": k, **base})
    return tasks


def mapped_path(root: Path, task) -> Path:
    return root / "mapped" / f"{task['circuit']}__{task['lib']}.v"


def load_mapped(root: Path, task):
    from fourprop.liberty import read_liberty, load_mapped_verilog
    cells = read_liberty(task["libfile"])
    return load_mapped_verilog(str(mapped_path(root, task)), cells, name=task["circuit"])


def run_task(task: dict, extra: dict) -> dict:
    root = Path(extra["root"])
    if task["kind"] == "map":
        (root / "mapped").mkdir(exist_ok=True)
        out = mapped_path(root, task)
        script = (f"read_verilog {task['src']}; hierarchy -auto-top; proc; flatten; techmap; opt; "
                  f"abc -liberty {task['libfile']}; opt_clean; write_verilog -noattr {out}")
        r = subprocess.run(["yosys", "-q", "-p", script], capture_output=True, text=True, timeout=TIMEOUTS["map"] - 60)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError("yosys failed: " + r.stderr[-500:])
        circ, nets = load_mapped(root, task)
        from collections import Counter
        return {"n_inputs": circ.n_inputs, "n_nodes": len(circ.nodes), "n_nets": len(nets), "depth": circ.depth(),
                "total_cap": float(sum(n.load_cap for n in nets))}
    # wait for the mapped netlist (the scheduler orders map first)
    if not mapped_path(root, task).exists():
        raise RuntimeError("mapped netlist missing")
    circ, nets = load_mapped(root, task)
    p = regime_p(circ.n_inputs, task["regime"], task["circuit"])
    net_nodes = np.array([n.node for n in nets]); caps = np.array([n.load_cap for n in nets])
    res = {"n_inputs": circ.n_inputs, "n_nodes": len(circ.nodes), "n_nets": len(nets)}
    if task["kind"] == "mc":
        from fourprop.mc_torch import monte_carlo_torch
        P, se = monte_carlo_torch(circ, p, n_samples=100_000_000, batch=4_000_000, seed=7, device="cuda:1")
    elif task["kind"] == "bdd":
        from fourprop.bdd import bdd_probabilities
        (root / "logs").mkdir(exist_ok=True)
        with open(root / "logs" / "bdd.log", "a") as log:
            P = bdd_probabilities(circ, p, time_limit=TIMEOUTS["bdd"] - 120, log=log)
    else:
        from fourprop.dense import propagate_dense
        kw = dict(k=task["k"], backend="cuda:0", project="norm")
        propagate_dense(circ, p, **kw)
        P, rt, _ = propagate_dense(circ, p, **kw); res["runtime_propagate"] = rt
    Pn = np.clip(np.asarray(P)[net_nodes], 0, 1); alpha = 2 * Pn * (1 - Pn)
    res["P_nets"] = Pn.tolist(); res["alpha_nets"] = alpha.tolist(); res["caps"] = caps.tolist()
    res["power"] = float((alpha * caps).sum())
    return res


def write_verdict(root: Path) -> str:
    R = root / "results"

    def get(t):
        f = R / f"{t}.json"
        if f.exists():
            d = json.load(open(f)); return d if d.get("ok") else None

    rows = []
    for c in ISCAS:
        for lib in LIBS:
            m = get(f"map__{c}__{lib}")
            for r in ("half", "rand1"):
                t = get(f"bdd__{c}__{lib}__{r}") or get(f"mc__{c}__{lib}__{r}")
                if not t or not m:
                    rows.append({"circuit": c, "lib": lib, "regime": r, "status": "missing"}); continue
                row = {"circuit": c, "lib": lib, "regime": r, "truth": "bdd" if get(f"bdd__{c}__{lib}__{r}") else "mc", "n_nets": m["n_nets"],
                       "n_nodes": m["n_nodes"], "depth": m["depth"], "power_true": t["power"], "status": "ok"}
                at = np.array(t["alpha_nets"])
                for k in (0, 1, 2):
                    d = get(f"dense__{c}__{lib}__{r}__k{k}")
                    if not d:
                        continue
                    row[f"k{k}_power_relerr"] = float((d["power"] - t["power"]) / t["power"])
                    row[f"k{k}_alpha_mean"] = float(np.mean(np.abs(np.array(d["alpha_nets"]) - at)))
                    row[f"k{k}_alpha_max"] = float(np.max(np.abs(np.array(d["alpha_nets"]) - at)))
                rows.append(row)
    cols = ["circuit", "lib", "regime", "truth", "n_nets", "n_nodes", "depth", "k0_power_relerr", "k1_power_relerr", "k2_power_relerr",
            "k0_alpha_mean", "k1_alpha_mean", "k2_alpha_mean", "k1_alpha_max", "k2_alpha_max", "status"]
    L = ["# Total dynamic power (sum alpha*C) relative error and per-net switching-activity error\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(x, ''):.4g}" if isinstance(r.get(x, ""), float) else str(r.get(x, "")) for x in cols) + " |")
    txt = "\n".join(L) + "\n"; (root / "E6_RESULTS.md").write_text(txt); json.dump(rows, open(root / "e6_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    return run_tasks(Path(a.root), task_list(Path(a.bench)), "power_runner", "run_task", {"bench": a.bench, "root": a.root}, TIMEOUTS, a.workers,
                     serial_kinds=("mc", "dense"), kind_order={"map": 0, "dense": 1, "mc": 2, "bdd": 3}, on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
