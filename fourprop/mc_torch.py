# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""GPU Monte Carlo signal probabilities with PyTorch (bool tensors, refcounted node values)."""
from __future__ import annotations
import numpy as np
from .circuit import Circuit


def monte_carlo_torch(circ: Circuit, p, n_samples: int = 100_000_000, batch: int = 4_000_000,
                      seed: int = 0, device: str = "cuda"):
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    p_t = torch.tensor(np.asarray(p, dtype=np.float32), device=device)
    N = len(circ.nodes)
    counts = torch.zeros(N, dtype=torch.int64, device=device)
    idx = circ.input_index()
    refs = [0] * N
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        # generate in column chunks so the float scratch tensor stays small (b × 32 floats)
        x = torch.empty((b, circ.n_inputs), dtype=torch.bool, device=device)
        for c0 in range(0, circ.n_inputs, 32):
            c1 = min(c0 + 32, circ.n_inputs)
            x[:, c0:c1] = torch.rand((b, c1 - c0), generator=g, device=device) < p_t[None, c0:c1]
        vals = [None] * N
        left = list(refs)
        for i, nd in enumerate(circ.nodes):
            if nd.op == "INPUT":
                v = x[:, idx[i]]
            elif nd.op == "CONST0":
                v = torch.zeros(b, dtype=torch.bool, device=device)
            elif nd.op == "CONST1":
                v = torch.ones(b, dtype=torch.bool, device=device)
            elif nd.op == "BUF":
                v = vals[nd.fanins[0]]
            elif nd.op == "NOT":
                v = ~vals[nd.fanins[0]]
            else:
                v = vals[nd.fanins[0]].clone()
                for j in nd.fanins[1:]:
                    w = vals[j]
                    if nd.op == "AND":
                        v &= w
                    elif nd.op == "OR":
                        v |= w
                    else:
                        v ^= w
            if nd.inv:
                v = ~v
            vals[i] = v
            counts[i] += v.sum()
            for j in nd.fanins:
                left[j] -= 1
                if left[j] == 0:
                    vals[j] = None
        done += b
    P = (counts.double() / n_samples).cpu().numpy()
    se = np.sqrt(P * (1 - P) / n_samples)
    return P, se
