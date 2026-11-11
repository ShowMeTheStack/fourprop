# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Probability-guided justification: choose input probabilities that maximise the estimated probability
that a target net takes a target value, by projected gradient ascent on the order-1 estimate."""
from __future__ import annotations
import numpy as np
from .circuit import Circuit
from .dense import propagate_dense


def ascend(circ: Circuit, target: int, value: int, p0, steps: int = 20, eta: float = 0.25, lo: float = 0.05,
           hi: float = 0.95, backend: str = "numpy", stop_at: float = 0.99):
    """Returns (p_final, trace) where trace is a list of estimated P(target = value) per step."""
    p = np.array(p0, dtype=np.float64); trace = []
    for _ in range(steps):
        P, _, kept = propagate_dense(circ, p, k=1, backend=backend, project="norm", keep_states=True)
        Pt, a, _ = kept[target]
        est = Pt if value == 1 else 1 - Pt
        trace.append(float(est))
        if est >= stop_at:
            break
        sigma = np.sqrt(p * (1 - p))
        grad = np.asarray(a) / sigma                       # ∂P̂/∂p_i proxy (exact for the true function)
        if value == 0:
            grad = -grad
        p = np.clip(p + eta * grad, lo, hi)
    return p, trace
