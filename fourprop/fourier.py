# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Order-k p-biased Fourier propagation (Theorem A product rule, degree truncation).

A State maps a sorted tuple of input indices T to the coefficient ĥ(T) in the p-biased
orthonormal basis φ_T = Π_{i∈T} (x_i − p_i)/σ_i. The signal probability is ĥ(()).

Product rule (Theorem A): for h = f·g,
    ĥ(T) = Σ_{S,U : SΔU ⊆ T ⊆ S∪U} f̂(S) ĝ(U) c_{T∖(SΔU)},   c_i = (1−2p_i)/σ_i.
Order-k truncation keeps only |T| ≤ k and uses only held |S|,|U| ≤ k.
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict, Tuple, List, Optional
import math
import numpy as np

from .circuit import Circuit

Key = Tuple[int, ...]
State = Dict[Key, float]


class Bias:
    """Per-input constants: p, σ, c = (1−2p)/σ, and ‖φ_i‖_∞."""

    def __init__(self, p):
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("input probabilities must lie strictly inside (0,1)")
        self.p = p
        self.sigma = np.sqrt(p * (1 - p))
        self.c = (1 - 2 * p) / self.sigma
        self.phi_inf = np.maximum(np.sqrt(p / (1 - p)), np.sqrt((1 - p) / p))

    def input_state(self, i: int) -> State:
        return {(): float(self.p[i]), (i,): float(self.sigma[i])}

    def phi_inf_T(self, T: Key) -> float:
        out = 1.0
        for i in T:
            out *= self.phi_inf[i]
        return out


def _subsets(items: List[int], max_size: int):
    for r in range(0, min(len(items), max_size) + 1):
        yield from combinations(items, r)


def product(F: State, G: State, k: int, bias: Bias, full: bool = False):
    """Order-k product of two states.

    Returns (H, dropped) where H holds coefficients with |T| ≤ k. If `full` is True,
    `dropped` holds the coefficients of degree > k of the complete product f̃·g̃
    (needed for the certified bound); otherwise `dropped` is None.
    """
    H: State = {}
    dropped: Optional[State] = {} if full else None
    c = bias.c
    for S, fs in F.items():
        if len(S) > k:
            continue
        Sset = set(S)
        for U, gu in G.items():
            if len(U) > k:
                continue
            Uset = set(U)
            D = Sset ^ Uset
            nD = len(D)
            if nD > k and not full:
                continue
            C = sorted(Sset & Uset)
            base = fs * gu
            max_w = (len(C) if full else k - nD)
            if max_w < 0:
                continue
            for W in _subsets(C, max_w):
                coef = base
                for i in W:
                    coef *= c[i]
                T = tuple(sorted(D.union(W)))
                if len(T) <= k:
                    H[T] = H.get(T, 0.0) + coef
                else:
                    dropped[T] = dropped.get(T, 0.0) + coef
    return H, dropped


def negate(F: State) -> State:
    H = {T: -v for T, v in F.items()}
    H[()] = 1.0 + H.get((), 0.0)
    return H


def add(F: State, G: State, a: float = 1.0, b: float = 1.0) -> State:
    H: State = {}
    for T, v in F.items():
        H[T] = H.get(T, 0.0) + a * v
    for T, v in G.items():
        H[T] = H.get(T, 0.0) + b * v
    return H


def value(F: State) -> float:
    return F.get((), 0.0)


def tail_mass(F: State) -> float:
    """τ(f) = P − Σ_T f̂(T)² for the held coefficients (≥ 0 for an exact Boolean state)."""
    P = value(F)
    return P - sum(v * v for v in F.values())


def variance_explained(F: State) -> float:
    P = value(F)
    var = P * (1 - P)
    if var <= 0:
        return 1.0
    return (sum(v * v for T, v in F.items() if T) ) / var


def l2_norm_sq(F: State) -> float:
    return sum(v * v for v in F.values())


def sup_bound(F: State, bias: Bias, k: int) -> float:
    """Computable upper bound on ‖f̃‖_∞.

    For k = 1 the held polynomial is affine in x and the exact sup is returned.
    Otherwise Σ_T |f̂(T)| ‖φ_T‖_∞.
    """
    if k <= 1:
        P = value(F)
        hi = lo = P
        for T, v in F.items():
            if not T:
                continue
            i = T[0]
            # coefficient v multiplies (x_i − p_i)/σ_i, which takes the values (1−p_i)/σ_i and −p_i/σ_i
            u1 = v * (1 - bias.p[i]) / bias.sigma[i]; u0 = -v * bias.p[i] / bias.sigma[i]
            hi += max(u1, u0); lo += min(u1, u0)
        return max(abs(hi), abs(lo))
    return sum(abs(v) * bias.phi_inf_T(T) for T, v in F.items())


def feasibility_project(F: State) -> State:
    """Project onto the necessary conditions for a Boolean function: 0 ≤ P ≤ 1 and Σ_{T≠∅} f̂(T)² ≤ P(1−P)."""
    P = min(max(value(F), 0.0), 1.0)
    var = P * (1 - P)
    rest = sum(v * v for T, v in F.items() if T)
    H = dict(F)
    H[()] = P
    if rest > var and rest > 0:
        s = math.sqrt(var / rest)
        for T in H:
            if T:
                H[T] *= s
    return H


class NetResult:
    __slots__ = ("state", "delta", "Delta", "e")

    def __init__(self, state: State, delta: float = 0.0, Delta: float = 0.0, e: float = 0.0):
        self.state = state
        self.delta = delta      # certified bound on ‖h − h̃‖_2 (Theorem C)
        self.Delta = Delta      # certified bound on ‖h − h̃‖_∞
        self.e = e              # certified bound on |P_H − P̂_H| (value only; e ≤ delta)

    @property
    def P(self) -> float:
        return value(self.state)


def _and_certified(Fr: NetResult, Gr: NetResult, k: int, bias: Bias) -> NetResult:
    H, dropped = product(Fr.state, Gr.state, k, bias, full=True)
    rho2 = math.sqrt(sum(v * v for v in dropped.values())) if dropped else 0.0
    rho_inf = sum(abs(v) * bias.phi_inf_T(T) for T, v in dropped.items()) if dropped else 0.0
    # two orderings; take the better
    best = None
    for A, B in ((Fr, Gr), (Gr, Fr)):
        BA = min(sup_bound(A.state, bias, k), 1.0 + A.Delta)
        delta = A.delta + BA * B.delta + rho2
        Delta = A.Delta + BA * B.Delta + rho_inf
        if best is None or delta < best[0]:
            best = (delta, Delta)
    # value error: |E[h − h̃]| ≤ |E[(f−f̃)g]| + |E[f̃(g−g̃)]| + |E[r]|, with E[r] = 0,
    # |E[(f−f̃)g]| ≤ δ_f ‖g‖_2 = δ_f √P_G and |E[f̃(g−g̃)]| ≤ ‖f̃‖_2 δ_g.
    e_best = best[0]
    for A, B in ((Fr, Gr), (Gr, Fr)):
        PB = math.sqrt(min(1.0, max(0.0, value(B.state)) + B.e))
        normA = math.sqrt(l2_norm_sq(A.state))
        e_best = min(e_best, A.delta * PB + normA * B.delta)
    return NetResult(H, best[0], best[1], e_best)


def _and_plain(Fr: NetResult, Gr: NetResult, k: int, bias: Bias) -> NetResult:
    H, _ = product(Fr.state, Gr.state, k, bias)
    return NetResult(H)


def propagate(circ: Circuit, p, k: int = 1, certify: bool = False, project: bool = False,
              clamp: bool = True, chain: str = "left") -> List[NetResult]:
    """Run order-k propagation over the circuit. Returns one NetResult per node (topological order).

    chain: how n-ary gates are reduced to 2-input gates: "left" (left-to-right), "balanced".
    """
    bias = Bias(p)
    idx = circ.input_index()
    res: List[Optional[NetResult]] = [None] * len(circ.nodes)
    and_op = _and_certified if certify else _and_plain

    def AND(a: NetResult, b: NetResult) -> NetResult:
        r = and_op(a, b, k, bias)
        if project:
            r.state = feasibility_project(r.state)
        elif clamp:
            r.state[()] = min(max(r.state[()], 0.0), 1.0)
        return r

    def NOT(a: NetResult) -> NetResult:
        return NetResult(negate(a.state), a.delta, a.Delta, a.e)

    def OR(a: NetResult, b: NetResult) -> NetResult:
        # a ∨ b = ¬(¬a ∧ ¬b): same product, exact complement bookkeeping
        return NOT(AND(NOT(a), NOT(b)))

    def XOR(a: NetResult, b: NetResult) -> NetResult:
        ab = AND(a, b)
        st = add(add(a.state, b.state), ab.state, 1.0, -2.0)
        r = NetResult(st, a.delta + b.delta + 2 * ab.delta, a.Delta + b.Delta + 2 * ab.Delta,
                      a.e + b.e + 2 * ab.e)
        if project:
            r.state = feasibility_project(r.state)
        elif clamp:
            r.state[()] = min(max(r.state[()], 0.0), 1.0)
        return r

    ops = {"AND": AND, "OR": OR, "XOR": XOR}

    def reduce_chain(fn, items: List[NetResult]) -> NetResult:
        if chain == "balanced":
            while len(items) > 1:
                nxt = [fn(items[i], items[i + 1]) if i + 1 < len(items) else items[i]
                       for i in range(0, len(items), 2)]
                items = nxt
            return items[0]
        acc = items[0]
        for it in items[1:]:
            acc = fn(acc, it)
        return acc

    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            j = idx[i]
            if k >= 1:
                r = NetResult(bias.input_state(j))
            else:  # order 0 holds only the constant; the dropped degree-1 term has L2 norm σ and sup max(p, 1−p)
                r = NetResult({(): float(bias.p[j])}, float(bias.sigma[j]), float(max(bias.p[j], 1 - bias.p[j])), 0.0)
        elif nd.op == "CONST0":
            r = NetResult({(): 0.0})
        elif nd.op == "CONST1":
            r = NetResult({(): 1.0})
        elif nd.op == "BUF":
            r = res[nd.fanins[0]]
        elif nd.op == "NOT":
            r = NOT(res[nd.fanins[0]])
        else:
            r = reduce_chain(ops[nd.op], [res[j] for j in nd.fanins])
        if nd.inv:
            r = NOT(r)
        res[i] = r
    return res  # type: ignore


def probabilities(results: List[NetResult]) -> np.ndarray:
    return np.array([r.P for r in results])


def switching_activity(P: np.ndarray) -> np.ndarray:
    return 2 * P * (1 - P)
