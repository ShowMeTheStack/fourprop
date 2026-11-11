# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Exact ground truth by enumeration (n ≤ ~22) and exact p-biased Fourier coefficients."""
from __future__ import annotations
from itertools import combinations
from typing import Dict, Tuple, List
import numpy as np

from .circuit import Circuit, evaluate


def all_patterns(n: int) -> np.ndarray:
    if n > 24:
        raise ValueError("too many inputs for enumeration")
    N = 1 << n
    ar = np.arange(N, dtype=np.int64)
    return ((ar[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def pattern_weights(x: np.ndarray, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    logw = x * np.log(p)[None, :] + (~x) * np.log(1 - p)[None, :]
    return np.exp(logw.sum(axis=1))


def exact_probabilities(circ: Circuit, p) -> np.ndarray:
    """Exact P(node = 1) for every node under independent inputs with probabilities p."""
    x = all_patterns(circ.n_inputs)
    w = pattern_weights(x, p)
    vals = evaluate(circ, x)
    return w @ vals.astype(float)


def exact_coefficients(circ: Circuit, p, k: int, nodes: List[int] | None = None) -> List[Dict[Tuple[int, ...], float]]:
    """Exact p-biased coefficients f̂(T) for |T| ≤ k, for the given nodes (default: all)."""
    p = np.asarray(p, dtype=float)
    n = circ.n_inputs
    x = all_patterns(n)
    w = pattern_weights(x, p)
    vals = evaluate(circ, x).astype(float)
    sigma = np.sqrt(p * (1 - p))
    phi = (x.astype(float) - p[None, :]) / sigma[None, :]
    if nodes is None:
        nodes = list(range(len(circ.nodes)))
    out: List[Dict[Tuple[int, ...], float]] = []
    basis = {}
    for r in range(0, k + 1):
        for T in combinations(range(n), r):
            b = w.copy()
            for i in T:
                b *= phi[:, i]
            basis[T] = b
    for nid in nodes:
        f = vals[:, nid]
        out.append({T: float(b @ f) for T, b in basis.items()})
    return out
