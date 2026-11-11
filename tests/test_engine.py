# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import math
import numpy as np
import pytest
from fourprop.circuit import parse_bench
from fourprop.fourier import propagate, Bias, product, probabilities
from fourprop.exact import exact_probabilities, exact_coefficients
from fourprop.synth import random_circuit

C17 = """
INPUT(1)
INPUT(2)
INPUT(3)
INPUT(6)
INPUT(7)
OUTPUT(22)
OUTPUT(23)
10 = NAND(1, 3)
11 = NAND(3, 6)
16 = NAND(2, 11)
19 = NAND(11, 7)
22 = NAND(10, 16)
23 = NAND(16, 19)
"""

EX3 = """
INPUT(x)
INPUT(y)
INPUT(z)
OUTPUT(a1)
OUTPUT(a2)
OUTPUT(a3)
OUTPUT(a4)
OUTPUT(a5)
nx = NOT(x)
a1 = AND(x, x)
a2 = AND(x, nx)
oxy = OR(x, y)
oxz = OR(x, z)
a3 = AND(oxy, oxz)
exy = XOR(x, y)
exz = XOR(x, z)
a4 = AND(exy, exz)
axy = AND(x, y)
o1 = OR(axy, z)
a5 = AND(axy, o1)
"""


def test_notes_table_e3_order1():
    circ = parse_bench(EX3)
    p = [0.7, 0.4, 0.55]
    res = propagate(circ, p, k=1)
    P = {circ.nodes[i].name: res[i].P for i in circ.outputs}
    assert abs(P["a1"] - 0.7000) < 1e-9
    assert abs(P["a2"] - 0.0000) < 1e-9
    assert abs(P["a3"] - 0.7660) < 1e-4
    assert abs(P["a4"] - 0.2550) < 1e-4
    assert abs(P["a5"] - 0.2573) < 1e-4
    ex = exact_probabilities(circ, p)
    assert abs(ex[circ.name_to_id["a5"]] - 0.28) < 1e-12
    # order 0 = independence column of the table
    res0 = propagate(circ, p, k=0)
    assert abs(res0[circ.name_to_id["a1"]].P - 0.49) < 1e-9
    assert abs(res0[circ.name_to_id["a5"]].P - 0.1893) < 1e-4
    # order 2 is exact on a5 (shared support of size 2)
    res2 = propagate(circ, p, k=2)
    assert abs(res2[circ.name_to_id["a5"]].P - 0.28) < 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_full_order_is_exact(seed):
    n = 6
    circ = random_circuit(n, 20, seed=seed, max_fanin=3)
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    res = propagate(circ, p, k=n, clamp=False)
    ex = exact_probabilities(circ, p)
    assert np.max(np.abs(probabilities(res) - ex)) < 1e-9
    # all coefficients exact too
    coef = exact_coefficients(circ, p, k=n)
    for i in range(len(circ.nodes)):
        for T, v in coef[i].items():
            assert abs(res[i].state.get(T, 0.0) - v) < 1e-9, (i, T)


def test_theorem_a_product_vs_exact():
    """Product of two exact states at full order equals the exact state of the AND."""
    rng = np.random.default_rng(1)
    n = 5
    for seed in range(5):
        circ = random_circuit(n, 12, seed=100 + seed)
        p = rng.uniform(0.05, 0.95, size=n)
        coef = exact_coefficients(circ, p, k=n)
        bias = Bias(p)
        a, b = len(circ.nodes) - 1, len(circ.nodes) - 3
        H, _ = product(coef[a], coef[b], n, bias)
        # exact AND via enumeration
        from fourprop.circuit import Node, Circuit
        nodes = list(circ.nodes) + [Node("h", "AND", [a, b])]
        c2 = Circuit(nodes, {nd.name: i for i, nd in enumerate(nodes)}, circ.inputs, [len(nodes) - 1])
        ch = exact_coefficients(c2, p, k=n, nodes=[len(nodes) - 1])[0]
        for T, v in ch.items():
            assert abs(H.get(T, 0.0) - v) < 1e-9


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("seed", range(10))
def test_certified_bound_is_sound(k, seed):
    n = 7
    circ = random_circuit(n, 25, seed=seed, max_fanin=3)
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.15, 0.85, size=n)
    res = propagate(circ, p, k=k, certify=True, clamp=False)
    ex = exact_probabilities(circ, p)
    for i in range(len(circ.nodes)):
        err = abs(res[i].P - ex[i])
        assert err <= res[i].delta + 1e-9, (i, err, res[i].delta)
        assert err <= res[i].e + 1e-9, (i, err, res[i].e)
        assert res[i].e <= res[i].delta + 1e-12


def test_c17_orders_vs_exact():
    circ = parse_bench(C17)
    p = np.full(5, 0.5)
    ex = exact_probabilities(circ, p)
    errs = {}
    for k in (0, 1, 2, 3, 5):
        res = propagate(circ, p, k=k)
        errs[k] = np.max(np.abs(probabilities(res) - ex))
    assert errs[5] < 1e-12
    assert errs[1] <= errs[0] + 1e-12
