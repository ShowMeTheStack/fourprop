#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Verdict for the fast order-2 experiment: criterion B with dense runtimes."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
REGIMES = ["half", "rand1"]


def load(root, tid):
    f = root / "results" / f"{tid}.json"
    return json.load(open(f)) if f.exists() else None


def write_verdict(root: Path) -> str:
    rows = []
    for c in CIRCUITS:
        for r in REGIMES:
            truth = load(root, f"bdd__{c}__{r}"); src = "bdd"
            if not truth or not truth.get("ok"):
                truth = load(root, f"mc__{c}__{r}"); src = "mc"
            if not truth or not truth.get("ok"):
                rows.append({"circuit": c, "regime": r, "status": "no truth"}); continue
            ex = np.array(truth["P"]); row = {"circuit": c, "regime": r, "truth": src, "n_nodes": len(ex), "status": "ok"}
            if src == "mc":
                row["mc_max_se"] = float(np.max(truth["se"]))
            for k in (1, 2):
                for be in ("cuda0", "numpy"):
                    d = load(root, f"dense__{c}__{r}__k{k}__{be}")
                    if not d:
                        continue
                    if not d.get("ok"):
                        row[f"k{k}_{be}_error"] = d.get("error", "?")[:50]; continue
                    P = np.clip(np.array(d["P"]), 0, 1); err = np.abs(P - ex)
                    row[f"k{k}_mean"] = float(err.mean()); row[f"k{k}_p95"] = float(np.quantile(err, .95)); row[f"k{k}_max"] = float(err.max())
                    row[f"k{k}_rt_{be}"] = float(d["runtime_propagate"])
            s = load(root, f"sparse__{c}__{r}__k1")
            if s and s.get("ok"):
                row["k1_rt_sparse"] = float(s["runtime_propagate"])
            if "k1_rt_cuda0" in row and "k2_rt_cuda0" in row:
                row["ratio_cuda"] = row["k2_rt_cuda0"] / max(row["k1_rt_cuda0"], 1e-6)
            if "k1_rt_numpy" in row and "k2_rt_numpy" in row:
                row["ratio_numpy"] = row["k2_rt_numpy"] / max(row["k1_rt_numpy"], 1e-6)
            if "k1_mean" in row and "k2_mean" in row:
                row["err_ratio"] = row["k1_mean"] / max(row["k2_mean"], 1e-12)
                best_ratio = min(row.get("ratio_cuda", 1e9), row.get("ratio_numpy", 1e9))
                row["critB"] = bool(row["k2_mean"] <= 0.5 * row["k1_mean"] and best_ratio <= 10)
            rows.append(row)
    ok = [r for r in rows if "critB" in r]
    passed = sum(r["critB"] for r in ok)
    go = len(ok) > 0 and passed * 2 >= len(ok)
    cols = ["circuit", "regime", "truth", "n_nodes", "k1_mean", "k2_mean", "err_ratio", "k1_p95", "k2_p95", "k1_max", "k2_max",
            "k1_rt_cuda0", "k2_rt_cuda0", "ratio_cuda", "k1_rt_numpy", "k2_rt_numpy", "ratio_numpy", "k1_rt_sparse", "critB", "status"]
    L = [f"# Fast order-2 criterion: **{'met' if go else 'not met'}**\n",
         f"Criterion B (order-2 mean error <= 0.5 x order-1 and dense runtime ratio <= 10): {passed}/{len(ok)} circuit-regimes.\n",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols) + " |")
    errs = [(r["circuit"], r["regime"], k, r[k]) for r in rows for k in r if k.endswith("_error")]
    if errs:
        L.append("\nTask errors:"); L += [f"- {e}" for e in errs]
    txt = "\n".join(L) + "\n"
    (root / "FAST_VERDICT.md").write_text(txt); json.dump(rows, open(root / "fast_rows.json", "w"), indent=1)
    return txt


if __name__ == "__main__":
    print(write_verdict(Path(sys.argv[1])))
