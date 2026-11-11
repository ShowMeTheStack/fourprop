# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Dense (vectorised) order-1 and order-2 propagation; NumPy or PyTorch (GPU) backend.

Order-2 state per net: P (scalar), a (n-vector, a_i = ĥ({i})), A (n×n symmetric, zero diagonal,
A_ij = ĥ({i,j})). From Theorem A with c_i = (1−2p_i)/σ_i, for h = f·g:

  P_H = P_F P_G + a_F·a_G + ½ ⟨A_F, A_G⟩
  a_H = P_F a_G + P_G a_F + c∘a_F∘a_G + A_F a_G + A_G a_F + c∘rowsum(A_F∘A_G)
  A_H = P_F A_G + P_G A_F + a_F a_Gᵀ + a_G a_Fᵀ + A_G∘(u1ᵀ+1uᵀ) + A_F∘(v1ᵀ+1vᵀ)
        + (c cᵀ)∘A_F∘A_G + A_F A_G + A_G A_F,     u = c∘a_F, v = c∘a_G, diagonal zeroed.
Order 1 keeps only P and a (drop every A term). NOT: (1−P, −a, −A).
"""
from __future__ import annotations
import time
from typing import List, Optional
import numpy as np
from .circuit import Circuit


class _NP:
    name = "numpy"

    def __init__(self):
        self.np = np

    def array(self, x):
        return np.asarray(x, dtype=np.float64)

    def zeros(self, *shape):
        return np.zeros(shape, dtype=np.float64)

    def unit(self, n, i):
        v = np.zeros(n); v[i] = 1.0; return v

    def outer(self, a, b):
        return np.outer(a, b)

    def zero_diag(self, A):
        np.fill_diagonal(A, 0.0); return A

    def to_numpy(self, x):
        return np.asarray(x)

    def sync(self):
        pass


class _Torch:
    name = "torch"

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = device

    def array(self, x):
        return self.torch.tensor(np.asarray(x, dtype=np.float64), device=self.device)

    def zeros(self, *shape):
        return self.torch.zeros(shape, dtype=self.torch.float64, device=self.device)

    def unit(self, n, i):
        v = self.zeros(n); v[i] = 1.0; return v

    def outer(self, a, b):
        return self.torch.outer(a, b)

    def zero_diag(self, A):
        return A.fill_diagonal_(0.0)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


def _backend(backend: str):
    if backend == "numpy":
        return _NP()
    if backend.startswith("cuda") or backend == "cpu":
        return _Torch(backend)
    raise ValueError(backend)


def propagate_dense(circ: Circuit, p, k: int = 2, backend: str = "numpy", clamp: bool = False,
                    keep_states: bool = False, project: Optional[str] = None, certify: bool = False):
    """Order-1/2 dense propagation. Returns (P: np.ndarray over nodes, runtime seconds, states or None).

    project: None (raw), "norm" (clip P to [0,1] and scale a, A so that ‖a‖² + ½‖A‖² ≤ P(1−P)),
             "box" (additionally |a_i| ≤ σ_i and |A_ij| ≤ 2σ_iσ_j). Projection implies clamp.
    """
    if k not in (0, 1, 2):
        raise ValueError("dense engine supports k = 0, 1 or 2")
    if certify and k == 2:
        raise ValueError("certified bounds in the dense engine are implemented for k = 0 and 1")
    B = _backend(backend)
    p = np.asarray(p, dtype=np.float64)
    n = circ.n_inputs
    sigma = np.sqrt(p * (1 - p))
    c = B.array((1 - 2 * p) / sigma)
    C2 = B.outer(c, c) if k == 2 else None
    sig = B.array(sigma)
    S2 = 2.0 * B.outer(sig, sig) if k == 2 else None

    def proj(st):
        P, a, A = st
        P = min(max(P, 0.0), 1.0)
        if a is None:
            return (P, None, None)
        if project == "box":
            a = a.clip(-sig, sig) if B.name == "numpy" else B.torch.maximum(B.torch.minimum(a, sig), -sig)
            if A is not None:
                A = A.clip(-S2, S2) if B.name == "numpy" else B.torch.maximum(B.torch.minimum(A, S2), -S2)
        var = P * (1.0 - P)
        rest = float((a * a).sum()) + (0.5 * float((A * A).sum()) if A is not None else 0.0)
        if rest > var:
            s_ = (var / rest) ** 0.5 if rest > 0 else 0.0
            a = a * s_
            if A is not None:
                A = A * s_
        return (P, a, A)
    idx = circ.input_index()
    N = len(circ.nodes)
    refs = [0] * N
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    left = list(refs)
    states: List[Optional[tuple]] = [None] * N
    kept = [None] * N if keep_states else None
    Pout = np.zeros(N)
    pvec = B.array(p); sigma_t = B.array(sigma)
    # certified radii (Theorem C, value-damped): per node (delta = ‖h − h̃‖₂ bound, e = |P_H − P̂_H| bound)
    cert: List[Optional[tuple]] = [None] * N
    Eout = np.zeros(N); Dout = np.zeros(N)

    def sup_affine(st):
        P, a, _ = st
        if a is None:
            return abs(P)
        u1 = a * (1.0 - pvec) / sigma_t; u0 = -a * pvec / sigma_t
        mx = np.maximum(u1, u0) if B.name == "numpy" else B.torch.maximum(u1, u0)
        mn = np.minimum(u1, u0) if B.name == "numpy" else B.torch.minimum(u1, u0)
        return max(abs(P + float(mx.sum())), abs(P + float(mn.sum())))

    def l2(st):
        P, a, _ = st
        return (P * P + (float((a * a).sum()) if a is not None else 0.0)) ** 0.5

    def rho2(f, g):
        # L2 norm of the dropped degree-2 part of f̃·g̃ at order 1 (zero at order 0)
        if k == 0 or f[1] is None or g[1] is None:
            return 0.0
        af, ag = f[1], g[1]
        nf = float((af * af).sum()); ng = float((ag * ag).sum()); dot = float((af * ag).sum())
        diag = float((af * af * ag * ag).sum())
        return max(nf * ng + dot * dot - 2.0 * diag, 0.0) ** 0.5

    def cert_and(f, g, cf, cg, h):
        df, ef = cf; dg, eg = cg
        r = rho2(f, g)
        best_d = min(df + sup_affine(f) * dg + r, dg + sup_affine(g) * df + r)
        PG = min(1.0, max(0.0, g[0]) + eg) ** 0.5; PF = min(1.0, max(0.0, f[0]) + ef) ** 0.5
        e = min(best_d, df * PG + l2(f) * dg, dg * PF + l2(g) * df)
        return (best_d, e)

    B.sync(); t0 = time.time()

    def AND(f, g):
        Pf, af, Af = f; Pg, ag, Ag = g
        if k == 0:
            return (Pf * Pg, None, None)
        P = Pf * Pg + float((af * ag).sum())
        a = Pf * ag + Pg * af + c * af * ag
        if k == 1:
            return (P, a, None)
        P += 0.5 * float((Af * Ag).sum())
        a = a + Af @ ag + Ag @ af + c * (Af * Ag).sum(1)
        u = c * af; v = c * ag
        A = (Pf * Ag + Pg * Af + B.outer(af, ag) + B.outer(ag, af)
             + Ag * (u[:, None] + u[None, :]) + Af * (v[:, None] + v[None, :])
             + C2 * Af * Ag + Af @ Ag + Ag @ Af)
        B.zero_diag(A)
        return (P, a, A)

    def NOT(f):
        Pf, af, Af = f
        return (1.0 - Pf, None if af is None else -af, None if Af is None else -Af)

    def XOR(f, g):
        h = AND(f, g)
        Pf, af, Af = f; Pg, ag, Ag = g; Ph, ah, Ah = h
        return (Pf + Pg - 2 * Ph, None if af is None else af + ag - 2 * ah, None if Af is None else Af + Ag - 2 * Ah)

    def OR(f, g):
        return NOT(AND(NOT(f), NOT(g)))

    ops = {"AND": AND, "OR": OR, "XOR": XOR}
    for i, nd in enumerate(circ.nodes):
        cr = (0.0, 0.0)
        if nd.op == "INPUT":
            j = idx[i]
            st = (float(p[j]), B.unit(n, j) * float(sigma[j]) if k >= 1 else None, B.zeros(n, n) if k == 2 else None)
            if k == 0:
                cr = (float(sigma[j]), 0.0)
        elif nd.op == "CONST0":
            st = (0.0, B.zeros(n) if k >= 1 else None, B.zeros(n, n) if k == 2 else None)
        elif nd.op == "CONST1":
            st = (1.0, B.zeros(n) if k >= 1 else None, B.zeros(n, n) if k == 2 else None)
        elif nd.op == "BUF":
            st = states[nd.fanins[0]]; cr = cert[nd.fanins[0]] or cr
        elif nd.op == "NOT":
            st = NOT(states[nd.fanins[0]]); cr = cert[nd.fanins[0]] or cr
        else:
            fn = ops[nd.op]
            st = states[nd.fanins[0]]; cr = cert[nd.fanins[0]] or (0.0, 0.0)
            for j in nd.fanins[1:]:
                g = states[j]; cg = cert[j] or (0.0, 0.0)
                if certify:
                    if nd.op == "AND":
                        cr = cert_and(st, g, cr, cg, None)
                    elif nd.op == "OR":
                        cr = cert_and(NOT(st), NOT(g), cr, cg, None)
                    else:  # XOR = f + g − 2fg
                        ca = cert_and(st, g, cr, cg, None)
                        cr = (cr[0] + cg[0] + 2 * ca[0], cr[1] + cg[1] + 2 * ca[1])
                st = fn(st, g)
        if nd.inv:
            st = NOT(st)
        if project and nd.op not in ("INPUT", "CONST0", "CONST1", "BUF"):
            st = proj(st)
        elif clamp:
            st = (min(max(st[0], 0.0), 1.0), st[1], st[2])
        states[i] = st; cert[i] = cr
        Pout[i] = st[0]; Dout[i] = cr[0]; Eout[i] = cr[1]
        if keep_states:
            kept[i] = (st[0], B.to_numpy(st[1]), None if st[2] is None else B.to_numpy(st[2]))
        for j in nd.fanins:
            left[j] -= 1
            if left[j] == 0:
                states[j] = None
    B.sync()
    if certify:
        return Pout, time.time() - t0, kept, Eout, Dout
    return Pout, time.time() - t0, kept
