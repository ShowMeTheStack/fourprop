# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Shared helpers for the extension experiments (pareto, cv, errmodel, ablation).

Truth for a circuit/regime comes from the clean-run artefacts (read-only): the BDD of the fast job for ISCAS AIGs,
else the Monte Carlo 10^8 (seed 0) of the phase2 job.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from fast_runner import regime_p  # noqa: E402

CIRCUITS = [("aig", c, f"iscas85/{c}.bench") for c in ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]] + \
           [("epfl", c, f"epfl_bench/{c}.bench") for c in ["adder", "bar", "div", "log2", "max", "multiplier", "sin", "sqrt", "square", "arbiter", "cavlc", "ctrl", "dec", "i2c", "int2float", "priority", "router", "voter"]]
ARITH = {"c6288", "log2", "sin", "sqrt", "multiplier", "square"}
REGIMES = ("half", "rand1")
SMALL_EXACT = ["c17", "ctrl", "dec", "cavlc", "int2float", "sin"]   # n <= 24: exact coefficients by enumeration (oracle job)


def circuit_class(c: str) -> str:
    return "arithmetic" if c in ARITH else "control"


def load_truth(run2: Path, family: str, circuit: str, regime: str):
    """(P array over nodes, kind, runtime) from the clean run; raises FileNotFoundError if absent."""
    cands = []
    if family == "aig":
        cands.append(run2 / "fast" / "results" / f"bdd__{circuit}__{regime}.json")
    cands.append(run2 / "phase2" / "results" / f"mc__{family}__{circuit}__{regime}.json")
    if family == "aig":
        cands.append(run2 / "fast" / "results" / f"mc__{circuit}__{regime}.json")
    for f in cands:
        if f.exists():
            d = json.load(open(f))
            if d.get("ok") and "P" in d:
                return np.asarray(d["P"], dtype=np.float64), d["task"]["kind"], float(d.get("runtime", float("nan")))
    raise FileNotFoundError(f"no truth for {family}/{circuit}/{regime} under {run2}")


def errors(P: np.ndarray, truth: np.ndarray) -> dict:
    e = np.abs(P - truth)
    return {"mean_err": float(e.mean()), "max_err": float(e.max()), "p95_err": float(np.quantile(e, 0.95)), "rmse": float(np.sqrt((e * e).mean()))}
