# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Exact signal probabilities via BDDs (dd.cudd if available, else dd.autoref)."""
from __future__ import annotations
import sys
import time
from typing import Optional
import numpy as np
from .circuit import Circuit


def _backend():
    try:
        from dd import cudd as mod
        return mod, "cudd"
    except Exception:
        from dd import autoref as mod
        return mod, "autoref"


def bdd_probabilities(circ: Circuit, p, time_limit: Optional[float] = None, max_nodes: Optional[int] = None,
                      log=None) -> np.ndarray:
    """Exact P(node=1) for every node. Raises TimeoutError / MemoryError if limits are exceeded."""
    mod, name = _backend()
    p = np.asarray(p, dtype=float)
    bdd = mod.BDD()
    n = circ.n_inputs
    names = [f"v{i}" for i in range(n)]
    bdd.declare(*names)
    if name == "cudd":
        bdd.configure(reordering=True)
    else:
        try:
            bdd.configure(reordering=True)
        except Exception:
            pass
    idx = circ.input_index()
    t0 = time.time()
    funcs = [None] * len(circ.nodes)
    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            u = bdd.var(names[idx[i]])
        elif nd.op == "CONST0":
            u = bdd.false
        elif nd.op == "CONST1":
            u = bdd.true
        elif nd.op == "BUF":
            u = funcs[nd.fanins[0]]
        elif nd.op == "NOT":
            u = ~funcs[nd.fanins[0]]
        else:
            u = funcs[nd.fanins[0]]
            for j in nd.fanins[1:]:
                w = funcs[j]
                u = (u & w) if nd.op == "AND" else (u | w) if nd.op == "OR" else bdd.apply("xor", u, w)
        if nd.inv:
            u = ~u
        funcs[i] = u
        if time_limit is not None and time.time() - t0 > time_limit:
            raise TimeoutError(f"BDD construction exceeded {time_limit}s at node {i}/{len(circ.nodes)}")
        if max_nodes is not None and len(bdd) > max_nodes:
            raise MemoryError(f"BDD size {len(bdd)} exceeded {max_nodes} at node {i}")
        if log and i % 200 == 0:
            print(f"[bdd {circ.name}] node {i}/{len(circ.nodes)} size={len(bdd)} t={time.time()-t0:.0f}s", file=log, flush=True)

    pv = {names[k]: float(p[k]) for k in range(n)}
    memo = {}
    sys.setrecursionlimit(max(10000, 4 * n + 1000))

    def prob(u) -> float:
        if u == bdd.true:
            return 1.0
        if u == bdd.false:
            return 0.0
        if u.negated:
            return 1.0 - prob(~u)
        key = int(u)
        v = memo.get(key)
        if v is not None:
            return v
        q = pv[u.var]
        v = (1 - q) * prob(u.low) + q * prob(u.high)
        memo[key] = v
        return v

    return np.array([prob(u) for u in funcs])
