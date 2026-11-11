#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Projection experiment: dense orders 1,2 on GPU with projection off (clamp only), "norm", "box"; all ISCAS-85."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from fast_runner import regime_p, CIRCUITS, REGIMES  # noqa: E402

PROJ = ["clamp", "norm", "box"]
TIMEOUTS = {"dense": 3600}


def task_list(bench: Path):
    return [{"id": f"dense__{c}__{r}__k{k}__{pj}", "kind": "dense", "circuit": c, "regime": r, "k": k, "project": pj}
            for c in CIRCUITS if (bench / f"{c}.bench").exists() for r in REGIMES for k in (1, 2) for pj in PROJ]


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    from fourprop.dense import propagate_dense
    circ = load_bench(str(Path(extra["bench"]) / f"{task['circuit']}.bench"))
    p = regime_p(circ.n_inputs, task["regime"], task["circuit"])
    kw = dict(k=task["k"], backend="cuda:0")
    if task["project"] == "clamp":
        kw["clamp"] = True
    else:
        kw["project"] = task["project"]
    propagate_dense(circ, p, **kw)
    P, rt, _ = propagate_dense(circ, p, **kw)
    return {"P": P.tolist(), "runtime_propagate": rt, "n_nodes": len(circ.nodes), "depth": circ.depth()}


def write_verdict(root: Path) -> str:
    rows = []
    for c in CIRCUITS:
        for r in REGIMES:
            t = None
            for kind in ("bdd", "mc"):
                f = root / "results" / f"{kind}__{c}__{r}.json"
                if f.exists():
                    d = json.load(open(f))
                    if d.get("ok"):
                        t = (kind, np.array(d["P"])); break
            if t is None:
                rows.append({"circuit": c, "regime": r, "status": "no truth"}); continue
            row = {"circuit": c, "regime": r, "truth": t[0], "status": "ok"}
            for k in (1, 2):
                for pj in PROJ:
                    f = root / "results" / f"dense__{c}__{r}__k{k}__{pj}.json"
                    if not f.exists():
                        continue
                    d = json.load(open(f))
                    if not d.get("ok"):
                        row[f"k{k}_{pj}_error"] = d.get("error", "?")[:40]; continue
                    P = np.array(d["P"], dtype=float); fin = np.isfinite(P)
                    err = np.abs(np.clip(np.where(fin, P, 0.5), 0, 1) - t[1])
                    row[f"k{k}_{pj}_mean"] = float(err.mean()); row[f"k{k}_{pj}_max"] = float(err.max())
                    row[f"k{k}_{pj}_nonfinite"] = int((~fin).sum()); row[f"k{k}_{pj}_rt"] = float(d["runtime_propagate"])
            rows.append(row)
    cols = ["circuit", "regime", "truth"] + [f"k{k}_{pj}_{m}" for k in (1, 2) for pj in PROJ for m in ("mean", "max")] + \
           ["k1_clamp_nonfinite", "k2_clamp_nonfinite", "k2_norm_rt", "status"]
    L = ["# Projection experiment\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols) + " |")
    ok = [r for r in rows if "k2_norm_mean" in r and "k1_clamp_mean" in r]
    wins = sum(r["k2_norm_mean"] < r["k1_clamp_mean"] for r in ok)
    L.append(f"\nOrder-2 'norm' beats order-1 clamp (BAM) on {wins}/{len(ok)} circuit-regimes.")
    L.append("Order-2 'norm' mean-error ratio vs order-1 clamp: " + ", ".join(f"{r['circuit']}/{r['regime']} {r['k1_clamp_mean']/max(r['k2_norm_mean'],1e-12):.2f}" for r in ok))
    txt = "\n".join(L) + "\n"; (root / "PROJ_VERDICT.md").write_text(txt); json.dump(rows, open(root / "proj_rows.json", "w"), indent=1)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    root = Path(a.root); (root / "results").mkdir(parents=True, exist_ok=True)
    for f in Path(a.truth).glob("results/*.json"):
        if f.name.startswith(("bdd__", "mc__")) and not (root / "results" / f.name).exists():
            shutil.copy(f, root / "results" / f.name)
    return run_tasks(root, task_list(Path(a.bench)), "proj_runner", "run_task", {"bench": a.bench, "root": str(root)},
                     TIMEOUTS, workers=1, serial_kinds=("dense",), on_done=write_verdict)


if __name__ == "__main__":
    sys.exit(main())
