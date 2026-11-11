# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Monte Carlo with Fourier control variates (Proposition "control variates").

P̂_cv = ĥ(∅) + (1/N) Σ_j ( f(x_j) − f̃(x_j) ),  f̃ the held degree-≤k polynomial of the net, x_j ~ μ_p.
Unbiased; Var = ‖f − f̃‖²₂ / N ≤ δ_h² / N (δ_h the certified radius).

Order 1 for every net is one GEMM per batch: f̃ = P + Φ·aᵀ with Φ[j,i] = (x_j,i − p_i)/σ_i.
Order 2 for a selected set of nets adds ½ φᵀ A φ per pattern (einsum), which costs n² per net per pattern.
"""
from __future__ import annotations
import time
from typing import List, Optional
import numpy as np
from .circuit import Circuit


def simulate_batch(circ: Circuit, x, torch, keep: Optional[np.ndarray] = None):
    """Evaluate all nodes on the bool tensor x (b × n); returns bool matrix (b × N) (or b × |keep|)."""
    N = len(circ.nodes); b = x.shape[0]; idx = circ.input_index()
    vals = [None] * N
    refs = [0] * N
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    out = torch.empty((b, N if keep is None else len(keep)), dtype=torch.bool, device=x.device)
    pos = {int(i): j for j, i in enumerate(keep)} if keep is not None else None
    left = list(refs)
    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            v = x[:, idx[i]]
        elif nd.op == "CONST0":
            v = torch.zeros(b, dtype=torch.bool, device=x.device)
        elif nd.op == "CONST1":
            v = torch.ones(b, dtype=torch.bool, device=x.device)
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
        if keep is None:
            out[:, i] = v
        elif i in pos:
            out[:, pos[i]] = v
        for j in nd.fanins:
            left[j] -= 1
            if left[j] == 0:
                vals[j] = None
    return out


def cv_monte_carlo(circ: Circuit, p, P1: np.ndarray, A1: np.ndarray, n_samples: int, seed: int = 1, device: str = "cuda:0",
                   batch: Optional[int] = None, A2: Optional[np.ndarray] = None, nets2: Optional[np.ndarray] = None):
    """Plain and control-variate Monte Carlo in one pass.

    P1[node], A1[node, i]: order-1 propagated state (ĥ(∅), ĥ({i})). A2[t, i, j] (optional): order-2 matrices for the
    nodes listed in nets2. Returns dict with plain/cv1 estimates (per node), cv2 estimates (per nets2), the empirical
    variances of f and of the residuals, and timings.
    """
    import torch
    t_all = time.time()
    n = circ.n_inputs; N = len(circ.nodes)
    p = np.asarray(p, dtype=np.float64); sigma = np.sqrt(p * (1 - p))
    if batch is None:
        # three float32 (b × N) tensors live at once (values, held polynomial, residual): keep each under ~400 MB
        batch = int(max(2_000, min(400_000, 4.0e8 // max(N, 1))))   # ~1.6 GB per (b x N) float32 tensor; 3 live -> < 5 GB
    g = torch.Generator(device=device); g.manual_seed(seed)
    p32 = torch.tensor(p.astype(np.float32), device=device); s32 = torch.tensor(sigma.astype(np.float32), device=device)
    A1t = torch.tensor(np.asarray(A1, dtype=np.float32), device=device)            # N × n
    P1t = torch.tensor(np.asarray(P1, dtype=np.float32), device=device)            # N
    use2 = A2 is not None and nets2 is not None and len(nets2) > 0
    if use2:
        A2t = torch.tensor(np.asarray(A2, dtype=np.float32), device=device)        # T × n × n
        a2t = A1t[torch.tensor(np.asarray(nets2), device=device)]                  # T × n
        P2t = P1t[torch.tensor(np.asarray(nets2), device=device)]
        nets2_t = torch.tensor(np.asarray(nets2), device=device)
    sum_f = torch.zeros(N, dtype=torch.float64, device=device); sum_f2 = torch.zeros(N, dtype=torch.float64, device=device)
    sum_r1 = torch.zeros(N, dtype=torch.float64, device=device); sum_r1sq = torch.zeros(N, dtype=torch.float64, device=device)
    if use2:
        T = len(nets2)
        sum_r2 = torch.zeros(T, dtype=torch.float64, device=device); sum_r2sq = torch.zeros(T, dtype=torch.float64, device=device)
    done = 0; t_sim = 0.0; t_cv = 0.0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = torch.rand((b, n), generator=g, device=device) < p32[None, :]
        t0 = time.time(); V = simulate_batch(circ, x, torch)                       # b × N bool
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_sim += time.time() - t0; t0 = time.time()
        Vf = V.float()
        sum_f += Vf.sum(0).double(); sum_f2 += Vf.sum(0).double()                   # f ∈ {0,1}: f² = f
        Phi = (x.float() - p32[None, :]) / s32[None, :]                             # b × n
        ft1 = P1t[None, :] + Phi @ A1t.T                                            # b × N
        r1 = Vf - ft1
        sum_r1 += r1.sum(0).double(); sum_r1sq += (r1 * r1).sum(0).double()
        if use2:
            # ½ φᵀ A φ per pattern per selected net, in sub-batches so the b_sub × T × n intermediate stays under ~512 MB
            T = A2t.shape[0]; sub = max(1, int(5e8 // max(T * n * 4, 1)))
            q = torch.empty((b, T), dtype=torch.float32, device=Phi.device)
            for s0 in range(0, b, sub):
                Ps = Phi[s0:s0 + sub]                                                # b_sub × n
                Q = torch.einsum("bi,tij->btj", Ps, A2t)                             # b_sub × T × n
                q[s0:s0 + sub] = 0.5 * (Q * Ps[:, None, :]).sum(-1)
            r2 = Vf[:, nets2_t] - (P2t[None, :] + Phi @ a2t.T + q)
            sum_r2 += r2.sum(0).double(); sum_r2sq += (r2 * r2).sum(0).double()
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_cv += time.time() - t0
        done += b
    Nn = float(n_samples)
    plain = (sum_f / Nn).cpu().numpy()
    var_f = (sum_f2 / Nn - (sum_f / Nn) ** 2).cpu().numpy()
    mean_r1 = (sum_r1 / Nn); var_r1 = (sum_r1sq / Nn - mean_r1 ** 2).cpu().numpy()
    cv1 = (P1t.double() + mean_r1).cpu().numpy()
    out = {"plain": plain, "cv1": cv1, "var_f": var_f, "var_r1": var_r1, "t_sim": t_sim, "t_cv": t_cv, "t_total": time.time() - t_all, "batch": batch}
    if use2:
        mean_r2 = sum_r2 / Nn; var_r2 = (sum_r2sq / Nn - mean_r2 ** 2).cpu().numpy()
        out["cv2"] = (P2t.double() + mean_r2).cpu().numpy(); out["var_r2"] = var_r2; out["nets2"] = np.asarray(nets2)
    return out
