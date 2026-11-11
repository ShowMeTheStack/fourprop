# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import numpy as np
import pytest
from fourprop.synth import random_circuit
from fourprop.fourier import propagate, probabilities
from fourprop.dense import propagate_dense
from fourprop.circuit import parse_bench
from tests.test_engine import C17, EX3


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("seed", range(8))
def test_dense_matches_sparse(k, seed):
    n = 7
    circ = random_circuit(n, 40, seed=seed, max_fanin=3)
    p = np.random.default_rng(seed).uniform(0.1, 0.9, size=n)
    Ps = probabilities(propagate(circ, p, k=k, clamp=False))
    Pd, rt, kept = propagate_dense(circ, p, k=k, backend="numpy", keep_states=True)
    assert np.max(np.abs(Ps - Pd)) < 1e-9
    # compare full states on the last node
    st = propagate(circ, p, k=k, clamp=False)[-1].state
    P, a, A = kept[-1]
    for T, v in st.items():
        if len(T) == 1:
            assert abs(a[T[0]] - v) < 1e-9
        elif len(T) == 2:
            assert abs(A[T[0], T[1]] - v) < 1e-9 and abs(A[T[1], T[0]] - v) < 1e-9


def test_dense_notes_table():
    circ = parse_bench(EX3); p = [0.7, 0.4, 0.55]
    P1, _, _ = propagate_dense(circ, p, k=1)
    P2, _, _ = propagate_dense(circ, p, k=2)
    nid = circ.name_to_id
    assert abs(P1[nid["a5"]] - 0.2573) < 1e-4 and abs(P2[nid["a5"]] - 0.28) < 1e-9
    assert abs(P1[nid["a3"]] - 0.766) < 1e-4


def test_dense_c17():
    circ = parse_bench(C17); p = np.full(5, 0.5)
    from fourprop.exact import exact_probabilities
    ex = exact_probabilities(circ, p)
    P2, _, _ = propagate_dense(circ, p, k=2)
    assert np.max(np.abs(P2 - ex)) < 1e-9


@pytest.mark.parametrize("project", ["norm", "box"])
def test_projection_keeps_feasible_and_exact_when_exact(project):
    # on a circuit where order 2 is exact, projection must not change anything
    circ = parse_bench(EX3); p = [0.7, 0.4, 0.55]
    P2, _, _ = propagate_dense(circ, p, k=2, project=project)
    assert abs(P2[circ.name_to_id["a5"]] - 0.28) < 1e-9
    # deep random circuit: values stay in [0,1] and states feasible
    n = 8
    circ = random_circuit(n, 400, seed=3, ops=("AND", "NOT"), locality=12)
    p = np.full(n, 0.5)
    P, _, kept = propagate_dense(circ, p, k=2, project=project, keep_states=True)
    assert np.all(np.isfinite(P)) and np.all(P >= -1e-12) and np.all(P <= 1 + 1e-12)
    for Pn, a, A in kept:
        assert (a @ a) + 0.5 * (A * A).sum() <= Pn * (1 - Pn) + 1e-9


def test_dense_order0_is_independence():
    circ = parse_bench(EX3); p = [0.7, 0.4, 0.55]
    P0, _, _ = propagate_dense(circ, p, k=0)
    P0p, _, _ = propagate_dense(circ, p, k=0, project="norm")
    assert np.allclose(P0, P0p)
    assert abs(P0[circ.name_to_id["a1"]] - 0.49) < 1e-12 and abs(P0[circ.name_to_id["a5"]] - 0.1893) < 1e-4


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("seed", range(10))
def test_dense_certified_matches_sparse_and_is_sound(k, seed):
    from fourprop.exact import exact_probabilities
    n = 7
    circ = random_circuit(n, 30, seed=seed, max_fanin=3)
    p = np.random.default_rng(seed).uniform(0.15, 0.85, size=n)
    ex = exact_probabilities(circ, p)
    P, _, _, E, D = propagate_dense(circ, p, k=k, certify=True)
    assert np.all(np.abs(P - ex) <= E + 1e-9) and np.all(E <= D + 1e-9)
    # the sparse engine additionally caps the sup norm with its L∞ radius, so its δ can only be smaller
    res = propagate(circ, p, k=k, certify=True, clamp=False)
    for i in range(len(circ.nodes)):
        assert res[i].delta <= D[i] + 1e-9


def test_verilog_parser_c17():
    from fourprop.verilog import parse_verilog_primitives
    from fourprop.exact import exact_probabilities
    v = """// c17
module c17 (N1,N2,N3,N6,N7,N22,N23);
input N1,N2,N3,N6,N7;
output N22,N23;
wire N10,N11,N16,N19;
nand NAND2_1 (N10, N1, N3);
nand NAND2_2 (N11, N3, N6);
nand NAND2_3 (N16, N2, N11);
nand NAND2_4 (N19, N11, N7);
nand NAND2_5 (N22, N10, N16);
nand NAND2_6 (N23, N16, N19);
endmodule
"""
    c = parse_verilog_primitives(v); b = parse_bench(C17)
    assert c.n_inputs == 5 and len(c.outputs) == 2 and c.n_gates == 6
    p = np.full(5, 0.5)
    assert np.allclose(exact_probabilities(c, p)[c.outputs], exact_probabilities(b, p)[b.outputs])
