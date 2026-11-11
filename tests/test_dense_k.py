# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import itertools
import numpy as np
import pytest
from fourprop.synth import random_circuit
from fourprop.fourier import propagate, probabilities
from fourprop.dense import propagate_dense
from fourprop.dense_k import propagate_dense_k, terms_for, max_live_nets, state_bytes
from fourprop.circuit import parse_bench, evaluate
from tests.test_engine import C17, EX3


def _coef(kept_state, T):
    """ĥ(T) from the dense tensors."""
    d = len(T)
    return float(kept_state[d][tuple(T)]) if d else float(kept_state[0])


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("seed", range(6))
def test_matches_dense2(k, seed):
    n = 7
    circ = random_circuit(n, 40, seed=seed, max_fanin=3)
    p = np.random.default_rng(seed).uniform(0.1, 0.9, size=n)
    Pd, _, kd = propagate_dense(circ, p, k=k, backend="numpy", keep_states=True, project="norm")
    Pk, _, kk, _ = propagate_dense_k(circ, p, k=k, backend="numpy", keep_states=True, project="norm")
    assert np.max(np.abs(Pd - Pk)) < 1e-9
    if k >= 1:
        assert np.max(np.abs(kd[-1][1] - kk[-1][1])) < 1e-9
    if k == 2:
        assert np.max(np.abs(kd[-1][2] - kk[-1][2])) < 1e-9


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(5))
def test_matches_sparse(k, seed):
    n = 6
    circ = random_circuit(n, 25, seed=seed, max_fanin=3)
    p = np.random.default_rng(100 + seed).uniform(0.1, 0.9, size=n)
    res = propagate(circ, p, k=k, clamp=False)
    Ps = probabilities(res)
    Pk, _, kept, diag = propagate_dense_k(circ, p, k=k, backend="numpy", keep_states=True, project=None)
    assert np.max(np.abs(Ps - Pk)) < 1e-9
    for i in (len(circ.nodes) - 1, len(circ.nodes) // 2):
        st = res[i].state
        for T, v in st.items():
            assert abs(_coef(kept[i], T) - v) < 1e-9, (i, T)
        # dense has no coefficients outside the sparse state
        for d in range(1, k + 1):
            for T in itertools.combinations(range(n), d):
                if T not in st:
                    assert abs(_coef(kept[i], T)) < 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_exact_at_full_order(seed):
    n = 5
    circ = random_circuit(n, 20, seed=seed, max_fanin=2)
    p = np.random.default_rng(7 + seed).uniform(0.2, 0.8, size=n)
    Pk, _, _, _ = propagate_dense_k(circ, p, k=n, backend="numpy", project=None)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.uint8)
    w = np.prod(np.where(X == 1, p, 1 - p), axis=1)
    V = evaluate(circ, X)
    exact = (w[:, None] * V).sum(0)
    assert np.max(np.abs(Pk - exact)) < 1e-9


def test_symmetry_and_zero_repeats():
    circ = random_circuit(6, 30, seed=3, max_fanin=3)
    p = np.full(6, 0.37)
    _, _, kept, _ = propagate_dense_k(circ, p, k=3, backend="numpy", keep_states=True, project="norm")
    T3 = kept[-1][3]
    for perm in itertools.permutations(range(3)):
        assert np.allclose(T3, np.transpose(T3, perm))
    for i in range(6):
        assert np.all(T3[i, i, :] == 0) and np.all(T3[i, :, i] == 0) and np.all(T3[:, i, i] == 0)
    T2 = kept[-1][2]
    assert np.allclose(T2, T2.T) and np.all(np.diag(T2) == 0)


def test_notes_table_orders():
    circ = parse_bench(EX3); p = [0.7, 0.4, 0.55]; nid = circ.name_to_id
    for k, expect in ((0, 0.1893), (1, 0.2573), (2, 0.28), (3, 0.28)):
        P, _, _, _ = propagate_dense_k(circ, p, k=k, backend="numpy", project=None)
        assert abs(P[nid["a5"]] - expect) < 2e-4


def test_c17_exact_from_k2():
    circ = parse_bench(C17); p = np.full(5, 0.5)
    from fourprop.exact import exact_probabilities
    ex = exact_probabilities(circ, p)
    for k in (2, 3, 5):
        P, _, _, _ = propagate_dense_k(circ, p, k=k, backend="numpy", project=None)
        assert np.max(np.abs(P - ex)) < 1e-9


def test_terms_count_and_feasibility():
    T = terms_for(2)
    assert sorted(T[0]) == sorted([(0, 0, 0, 0, 1.0), (0, 0, 0, 1, 1.0), (0, 0, 0, 2, 0.5)])
    assert len(T[2]) == 7 and len(T[1]) == 6
    circ = random_circuit(8, 50, seed=1)
    assert 1 <= max_live_nets(circ) <= len(circ.nodes)
    assert state_bytes(10, 2) == (1 + 10 + 100) * 8
    with pytest.raises(MemoryError):
        propagate_dense_k(circ, np.full(8, 0.5), k=3, backend="numpy", memory_budget=10)


def test_projection_idempotent_on_feasible():
    circ = random_circuit(6, 20, seed=9, max_fanin=2)
    p = np.full(6, 0.5)
    P1, _, k1, _ = propagate_dense_k(circ, p, k=3, backend="numpy", keep_states=True, project="norm")
    P2, _, k2, _ = propagate_dense_k(circ, p, k=3, backend="numpy", keep_states=True, project=None)
    # at full-ish order on a small circuit the states are feasible, so projection must be the identity
    assert np.max(np.abs(P1 - P2)) < 1e-9
