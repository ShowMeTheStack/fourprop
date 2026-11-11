# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Cutting-algorithm interval bounds on signal probabilities.

Savir, Ditlow, Bardell (IEEE TC 1984) cut the branches of reconvergent fan-out stems, assign the
cut lines the interval [0,1], and propagate intervals through the gates. Markowsky (IEEE TC 1987)
proved the resulting intervals are correct.

Two cut policies:
  "all"          cut every branch of every stem with fan-out >= 2  (provably sound by a direct
                 conditioning argument; the loosest).
  "reconvergent" cut every branch of every stem whose branches reconverge downstream (SDB/Markowsky).
"""
from __future__ import annotations
from itertools import product as iproduct
from typing import List, Tuple
import numpy as np
from .circuit import Circuit


def _fanout_lists(circ: Circuit):
    fan = [[] for _ in circ.nodes]
    for gid, nd in enumerate(circ.nodes):
        for pos, j in enumerate(nd.fanins):
            fan[j].append((gid, pos))
    return fan


def reconvergent_stems(circ: Circuit) -> List[int]:
    """Node ids whose fan-out branches reconverge (two branches reach a common gate)."""
    N = len(circ.nodes)
    fan = _fanout_lists(circ)
    reach = [0] * N  # bitset of nodes reachable from node (excluding itself)
    for i in range(N - 1, -1, -1):
        r = 0
        for gid, _ in fan[i]:
            r |= (1 << gid) | reach[gid]
        reach[i] = r
    stems = []
    for i in range(N):
        cons = fan[i]
        if len(cons) < 2:
            continue
        cones = [(1 << gid) | reach[gid] for gid, _ in cons]
        rec = False
        for a in range(len(cones)):
            for b in range(a + 1, len(cones)):
                if cones[a] & cones[b]:
                    rec = True
                    break
            if rec:
                break
        if rec:
            stems.append(i)
    return stems


def _xor_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    vals = [x + y - 2 * x * y for x in a for y in b]
    return (min(vals), max(vals))


def cutting_bounds(circ: Circuit, p, policy: str = "reconvergent") -> np.ndarray:
    """Return array (n_nodes, 2) of [lo, hi] interval bounds on P(node = 1)."""
    p = np.asarray(p, dtype=float)
    fan = _fanout_lists(circ)
    if policy == "all":
        cut = {i for i in range(len(circ.nodes)) if len(fan[i]) >= 2}
    elif policy == "reconvergent":
        cut = set(reconvergent_stems(circ))
    else:
        raise ValueError(policy)
    idx = circ.input_index()
    iv = np.zeros((len(circ.nodes), 2))
    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            iv[i] = (p[idx[i]], p[idx[i]])
            continue
        if nd.op == "CONST0":
            iv[i] = (0.0, 0.0); continue
        if nd.op == "CONST1":
            iv[i] = (1.0, 1.0); continue
        ins = [(0.0, 1.0) if j in cut else (iv[j, 0], iv[j, 1]) for j in nd.fanins]
        if nd.op in ("NOT", "BUF"):
            lo, hi = ins[0]
            if nd.op == "NOT":
                lo, hi = 1 - hi, 1 - lo
        elif nd.op == "AND":
            lo = float(np.prod([a for a, _ in ins])); hi = float(np.prod([b for _, b in ins]))
        elif nd.op == "OR":
            lo = 1 - float(np.prod([1 - a for a, _ in ins])); hi = 1 - float(np.prod([1 - b for _, b in ins]))
        elif nd.op == "XOR":
            lo, hi = ins[0]
            for b in ins[1:]:
                lo, hi = _xor_interval((lo, hi), b)
        else:
            raise ValueError(nd.op)
        if nd.inv:
            lo, hi = 1 - hi, 1 - lo
        iv[i] = (lo, hi)
    return iv
