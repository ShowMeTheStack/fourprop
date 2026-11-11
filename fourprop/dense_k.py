# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Generic dense order-k propagation (k = 0 … 5 and beyond, memory permitting); NumPy or PyTorch backend.

State of a net: a list T[0..k]; T[d] is the full symmetric rank-d tensor over the n primary inputs with
T[d][i1,…,id] = ĥ({i1,…,id}) for distinct indices and 0 wherever an index repeats; T[0] is the scalar ĥ(∅) = P.
This is the natural generalisation of dense.py's (P, a, A).

Product rule (Theorem "Product rule"), for h = f·g and |T| = d:
    ĥ(T) = Σ_{S,U : SΔU ⊆ T ⊆ S∪U} f̂(S) ĝ(U) c_{T∖(SΔU)},   c_i = (1 − 2p_i)/σ_i.
Write S = X∪O, U = Y∪O with O = S∩U, T = X∪Y∪W with W ⊆ O, L = O∖W (contracted). With a=|X|, b=|Y|, w=|W|,
e=|L| (so |S| = a+w+e, |U| = b+w+e, d = a+b+w) every (a,b,w,e) gives one tensor contraction
    E[x…, y…, w…] = Σ_{l…} F_{a+w+e}[x…, w…, l…] · G_{b+w+e}[y…, w…, l…] · Π_j c[w_j]
and the set-level sum is  H_d += d!/(a! b! w! e!) · Sym(E), where Sym averages over the d! permutations of the
output indices (ordered tuples over-count each set partition a!b!w! times, ordered l-tuples over-count L e! times).
Entries with a repeated output index are zeroed afterwards (x–y coincidences are not killed by the inputs' zeros).
For k ≤ 2 this reproduces dense.py term by term (checked numerically in tests/test_dense_k.py).

NOT: T[0] ← 1 − T[0], T[d] ← −T[d]. OR via NOT(AND(NOT f, NOT g)); XOR = f + g − 2 f·g.
Projection "norm": clip P to [0,1]; if Σ_{d≥1} ‖T[d]‖²/d! > P(1−P) scale all T[d], d ≥ 1, by the square root of the ratio.
"""
from __future__ import annotations
import itertools, math, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from .circuit import Circuit

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def terms_for(k: int) -> Dict[int, List[Tuple[int, int, int, int, float]]]:
    """For each output degree d ≤ k the list of (a, b, w, e, coefficient) with a+w+e ≤ k, b+w+e ≤ k, d = a+b+w."""
    out: Dict[int, List[Tuple[int, int, int, int, float]]] = {d: [] for d in range(k + 1)}
    for a in range(k + 1):
        for b in range(k + 1):
            for w in range(k + 1):
                for e in range(k + 1):
                    if a + w + e > k or b + w + e > k:
                        continue
                    d = a + b + w
                    if d > k:
                        continue
                    coef = math.factorial(d) / (math.factorial(a) * math.factorial(b) * math.factorial(w) * math.factorial(e))
                    out[d].append((a, b, w, e, coef))
    return out


class _Backend:
    def __init__(self, backend: str, dtype: str = "float64"):
        self.name = backend
        if backend == "numpy":
            self.np = np; self.torch = None; self.dtype = np.float64 if dtype == "float64" else np.float32
        else:
            import torch
            self.torch = torch; self.device = backend
            self.dtype = torch.float64 if dtype == "float64" else torch.float32

    def zeros(self, shape):
        if self.torch is None:
            return np.zeros(shape, dtype=self.dtype)
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def array(self, x):
        if self.torch is None:
            return np.asarray(x, dtype=self.dtype)
        return self.torch.tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.device)

    def einsum(self, sub, *ops):
        if self.torch is None:
            return np.einsum(sub, *ops, optimize=True)
        return self.torch.einsum(sub, *ops)

    def permute(self, x, perm):
        if self.torch is None:
            return np.transpose(x, perm)
        return x.permute(*perm)

    def sumsq(self, x) -> float:
        return float((x * x).sum())

    def item(self, x) -> float:
        return float(x)

    def to_numpy(self, x):
        if self.torch is None:
            return np.asarray(x)
        return x.detach().cpu().numpy()

    def sync(self):
        if self.torch is not None and str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()

    def bool_mask(self, m: np.ndarray):
        if self.torch is None:
            return m
        return self.torch.tensor(m, device=self.device)


def _distinct_mask(n: int, d: int) -> np.ndarray:
    """Boolean array of shape (n,)*d, True where all d indices are distinct."""
    if d <= 1:
        return np.ones((n,) * d, dtype=bool)
    idx = np.indices((n,) * d)
    m = np.ones((n,) * d, dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            m &= idx[i] != idx[j]
    return m


def state_bytes(n: int, k: int, dtype_bytes: int = 8) -> int:
    return sum(n ** d for d in range(k + 1)) * dtype_bytes


def max_live_nets(circ: Circuit) -> int:
    """Peak number of simultaneously held states when nets are released after their last fan-out."""
    refs = [0] * len(circ.nodes)
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    live = 0; peak = 0
    for i, nd in enumerate(circ.nodes):
        live += 1; peak = max(peak, live)
        for j in nd.fanins:
            refs[j] -= 1
            if refs[j] == 0:
                live -= 1
    return peak


def propagate_dense_k(circ: Circuit, p, k: int = 3, backend: str = "numpy", project: Optional[str] = "norm",
                      keep_states: bool = False, dtype: str = "float64", memory_budget: Optional[int] = None):
    """Order-k dense propagation. Returns (P over nodes, runtime seconds, kept states or None, diag dict).

    diag: per node ρ_k (variance explained by the held non-constant coefficients) and τ_k estimate P(1−P)−Σ‖T_d‖²/d!.
    Raises MemoryError before starting if the estimated peak state memory exceeds memory_budget (bytes).
    """
    B = _Backend(backend, dtype)
    p = np.asarray(p, dtype=np.float64); n = circ.n_inputs
    sigma = np.sqrt(p * (1 - p)); cvec = B.array((1 - 2 * p) / sigma)
    if memory_budget is not None:
        need = state_bytes(n, k, 8 if dtype == "float64" else 4) * max_live_nets(circ)
        if need > memory_budget:
            raise MemoryError(f"estimated {need/1e9:.1f} GB of state exceeds budget {memory_budget/1e9:.1f} GB")
    TERMS = terms_for(k)
    masks = {d: B.bool_mask(_distinct_mask(n, d)) for d in range(2, k + 1)}
    fact = [math.factorial(d) for d in range(k + 1)]

    def zero_state():
        return [B.zeros(()) if d == 0 else B.zeros((n,) * d) for d in range(k + 1)]

    def sym(x, d):
        if d <= 1:
            return x
        acc = None
        for perm in itertools.permutations(range(d)):
            y = B.permute(x, perm)
            acc = y if acc is None else acc + y
        return acc / fact[d]

    def AND(F, G):
        H = zero_state()
        for d in range(k + 1):
            acc = None
            for (a, b, w, e, coef) in TERMS[d]:
                Fs = F[a + w + e]; Gs = G[b + w + e]
                if a + w + e == 0 and b + w + e == 0:
                    term = Fs * Gs
                else:
                    xs = _LETTERS[:a]; ys = _LETTERS[a:a + b]; ws = _LETTERS[a + b:a + b + w]; ls = _LETTERS[a + b + w:a + b + w + e]
                    ops = [xs + ws + ls, ys + ws + ls] + [ch for ch in ws]
                    sub = ",".join(ops) + "->" + xs + ys + ws
                    operands = [Fs, Gs] + [cvec] * w
                    # 0-dim operands need an empty subscript, which einsum accepts
                    term = B.einsum(sub, *operands)
                    if d >= 2:
                        term = sym(term, d)
                term = term * coef
                acc = term if acc is None else acc + term
            if d >= 2:
                acc = acc * masks[d]
            H[d] = acc
        return H

    def NOT(F):
        return [1.0 - F[0]] + [-F[d] for d in range(1, k + 1)]

    def OR(F, G):
        return NOT(AND(NOT(F), NOT(G)))

    def XOR(F, G):
        H = AND(F, G)
        return [F[d] + G[d] - 2.0 * H[d] for d in range(k + 1)]

    def proj(S):
        P = min(max(B.item(S[0]), 0.0), 1.0)
        var = P * (1.0 - P)
        rest = sum(B.sumsq(S[d]) / fact[d] for d in range(1, k + 1))
        s_ = (var / rest) ** 0.5 if (rest > var and rest > 0) else 1.0
        return [B.array(P)] + [S[d] * s_ for d in range(1, k + 1)]

    def diag_of(S):
        P = B.item(S[0]); var = P * (1.0 - P)
        rest = sum(B.sumsq(S[d]) / fact[d] for d in range(1, k + 1))
        return (rest / var if var > 1e-15 else 1.0, max(var - rest, 0.0))

    idx = circ.input_index(); N = len(circ.nodes)
    refs = [0] * N
    for nd in circ.nodes:
        for j in nd.fanins:
            refs[j] += 1
    left = list(refs); states: List[Optional[list]] = [None] * N
    kept = [None] * N if keep_states else None
    Pout = np.zeros(N); rho = np.zeros(N); tau = np.zeros(N)
    ops = {"AND": AND, "OR": OR, "XOR": XOR}
    B.sync(); t0 = time.time()
    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            j = idx[i]; st = zero_state(); st[0] = B.array(float(p[j]))
            if k >= 1:
                v = B.zeros((n,)); v[j] = float(sigma[j]); st[1] = v
        elif nd.op == "CONST0":
            st = zero_state()
        elif nd.op == "CONST1":
            st = zero_state(); st[0] = B.array(1.0)
        elif nd.op == "BUF":
            st = states[nd.fanins[0]]
        elif nd.op == "NOT":
            st = NOT(states[nd.fanins[0]])
        else:
            fn = ops[nd.op]; st = states[nd.fanins[0]]
            for j in nd.fanins[1:]:
                st = fn(st, states[j])
        if nd.inv:
            st = NOT(st)
        if project and nd.op not in ("INPUT", "CONST0", "CONST1", "BUF"):
            st = proj(st)
        states[i] = st; Pout[i] = B.item(st[0]); rho[i], tau[i] = diag_of(st)
        if keep_states:
            kept[i] = [B.to_numpy(st[d]) for d in range(k + 1)]
        for j in nd.fanins:
            left[j] -= 1
            if left[j] == 0:
                states[j] = None
    B.sync()
    return Pout, time.time() - t0, kept, {"rho": rho, "tau": tau}
