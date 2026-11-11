#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Tightness of the per-gate Cauchy-Schwarz bound (Theorem 'Per-gate error at order k').

Random search over pairs of Boolean functions f, g on n inputs under random product distributions p:
ratio = |P(f AND g) - order-k value from the exact coefficients of f and g| / sqrt(tau_k(f) tau_k(g)).
Reports the maximum ratio found per order (1 = bound attained). Writes results_run2/bound_tightness.json.
"""
import itertools, json, sys
import numpy as np
from pathlib import Path

def coeffs(f, p):
    """Exact p-biased Fourier coefficients of f (truth table over {0,1}^n, lexicographic x) -> dict T -> coefficient."""
    n = int(np.log2(len(f)))
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    sigma = np.sqrt(p * (1 - p))
    w = np.prod(np.where(X == 1, p, 1 - p), axis=1)
    phi = (X - p) / sigma
    out = {}
    for k in range(n + 1):
        for T in itertools.combinations(range(n), k):
            phiT = np.prod(phi[:, list(T)], axis=1) if T else np.ones(len(f))
            out[T] = float(np.sum(w * f * phiT))
    return out

def main(pairs=3000, nmax=6, seed=0, out=None):
    rng = np.random.default_rng(seed)
    best = {1: (0.0, None), 2: (0.0, None)}
    ratios = {1: [], 2: []}
    for t in range(pairs):
        n = int(rng.integers(3, nmax + 1))
        p = rng.uniform(0.1, 0.9, size=n)
        f = rng.integers(0, 2, size=2 ** n).astype(float); g = rng.integers(0, 2, size=2 ** n).astype(float)
        cf, cg = coeffs(f, p), coeffs(g, p)
        Pf, Pg = cf[()], cg[()]
        X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
        w = np.prod(np.where(X == 1, p, 1 - p), axis=1)
        Ph = float(np.sum(w * f * g))
        for k in (1, 2):
            est = sum(cf[T] * cg[T] for T in cf if len(T) <= k)
            tf = Pf - sum(cf[T] ** 2 for T in cf if len(T) <= k); tg = Pg - sum(cg[T] ** 2 for T in cg if len(T) <= k)
            bound = np.sqrt(max(tf, 0) * max(tg, 0))
            if bound > 1e-9:
                r = abs(Ph - est) / bound
                ratios[k].append(r)
                if r > best[k][0]:
                    best[k] = (r, {"n": n, "p": p.round(3).tolist(), "err": abs(Ph - est), "bound": float(bound)})
    res = {"pairs": pairs, "nmax": nmax, "seed": seed,
           "max_ratio": {k: best[k][0] for k in best}, "argmax": {k: best[k][1] for k in best},
           "mean_ratio": {k: float(np.mean(ratios[k])) for k in ratios}, "median_ratio": {k: float(np.median(ratios[k])) for k in ratios}}
    print(json.dumps(res, indent=1))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out=out)
