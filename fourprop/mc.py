# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Monte Carlo signal probabilities (NumPy; a torch/GPU version lives in mc_torch.py)."""
from __future__ import annotations
import numpy as np
from .circuit import Circuit, evaluate


def monte_carlo(circ: Circuit, p, n_samples: int = 1_000_000, batch: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    acc = np.zeros(len(circ.nodes))
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = rng.random((b, circ.n_inputs)) < p[None, :]
        acc += evaluate(circ, x).sum(axis=0)
        done += b
    P = acc / n_samples
    se = np.sqrt(P * (1 - P) / n_samples)
    return P, se
