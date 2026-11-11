# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Exact signal probabilities and p-biased Fourier coefficients up to degree 2 by full enumeration,
batched on a GPU (torch) or in NumPy. Feasible up to about 24 inputs."""
from __future__ import annotations
from typing import Optional
import numpy as np
from .circuit import Circuit


def exact_coeffs_deg2(circ: Circuit, p, device: Optional[str] = None, batch_bits: int = 20, log=None):
    """Returns (P, A1, A2) with P[node], A1[node, i] = f̂({i}), A2[node, i, j] = f̂({i,j}) (symmetric, zero diag)."""
    p = np.asarray(p, dtype=np.float64); n = circ.n_inputs; N = len(circ.nodes)
    if n > 26:
        raise ValueError("too many inputs for enumeration")
    sigma = np.sqrt(p * (1 - p))
    use_torch = device is not None
    if use_torch:
        import torch
        dev = device
        P = torch.zeros(N, dtype=torch.float64, device=dev)
        A1 = torch.zeros(N, n, dtype=torch.float64, device=dev)
        A2 = torch.zeros(N, n, n, dtype=torch.float64, device=dev)
        pv = torch.tensor(p, device=dev); sv = torch.tensor(sigma, device=dev)
    else:
        P = np.zeros(N); A1 = np.zeros((N, n)); A2 = np.zeros((N, n, n))
    idx = circ.input_index()
    refs = [0] * N
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    total = 1 << n
    B = min(total, 1 << batch_bits)
    for start in range(0, total, B):
        ar = np.arange(start, min(start + B, total), dtype=np.int64)
        x = ((ar[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
        logw = x * np.log(p)[None, :] + (~x) * np.log(1 - p)[None, :]
        w = np.exp(logw.sum(axis=1))
        phi = (x.astype(np.float64) - p[None, :]) / sigma[None, :]
        if use_torch:
            xt = torch.tensor(x, device=dev); wt = torch.tensor(w, device=dev)
            phit = torch.tensor(phi, device=dev)
            wphi = phit * wt[:, None]                       # (B, n)
            # pairwise basis weighted: (B, n, n) is too big; use per-node contraction instead
            vals = [None] * N; left = list(refs)
            V = torch.zeros(len(ar), N, dtype=torch.float64, device=dev)
            for i, nd in enumerate(circ.nodes):
                if nd.op == "INPUT":
                    v = xt[:, idx[i]]
                elif nd.op == "CONST0":
                    v = torch.zeros(len(ar), dtype=torch.bool, device=dev)
                elif nd.op == "CONST1":
                    v = torch.ones(len(ar), dtype=torch.bool, device=dev)
                elif nd.op == "BUF":
                    v = vals[nd.fanins[0]]
                elif nd.op == "NOT":
                    v = ~vals[nd.fanins[0]]
                else:
                    v = vals[nd.fanins[0]].clone()
                    for j in nd.fanins[1:]:
                        if nd.op == "AND":
                            v &= vals[j]
                        elif nd.op == "OR":
                            v |= vals[j]
                        else:
                            v ^= vals[j]
                if nd.inv:
                    v = ~v
                vals[i] = v; V[:, i] = v
                for j in nd.fanins:
                    left[j] -= 1
                    if left[j] == 0:
                        vals[j] = None
            P += wt @ V                                     # (N,)
            A1 += (wphi.T @ V).T                            # (N, n)
            # A2[node, i, j] = Σ_b w_b φ_i φ_j V_b: contract in chunks of nodes
            for c0 in range(0, N, 512):
                Vc = V[:, c0:c0 + 512]                      # (B, m)
                A2[c0:c0 + 512] += torch.einsum("bm,bi,bj->mij", Vc * wt[:, None], phit, phit)
            if log:
                print(f"[exact {circ.name}] {min(start+B,total)}/{total}", file=log, flush=True)
        else:
            from .circuit import evaluate
            V = evaluate(circ, x).astype(np.float64)
            P += w @ V
            A1 += ((phi * w[:, None]).T @ V).T
            A2 += np.einsum("bm,bi,bj->mij", V * w[:, None], phi, phi)
    if use_torch:
        P = P.cpu().numpy(); A1 = A1.cpu().numpy(); A2 = A2.cpu().numpy()
    for i in range(n):
        A2[:, i, i] = 0.0
    return P, A1, A2
