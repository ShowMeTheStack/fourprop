# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import numpy as np
import pytest
from fourprop.synth import random_circuit
from fourprop.exact import exact_probabilities
from fourprop.cutting import cutting_bounds, reconvergent_stems
from fourprop.bdd import bdd_probabilities
from fourprop.circuit import parse_bench
from tests.test_engine import C17, EX3


@pytest.mark.parametrize("policy", ["all", "reconvergent"])
@pytest.mark.parametrize("seed", range(40))
def test_cutting_is_sound(policy, seed):
    n = 7
    circ = random_circuit(n, 30, seed=seed, max_fanin=3)
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    ex = exact_probabilities(circ, p)
    iv = cutting_bounds(circ, p, policy=policy)
    assert np.all(ex >= iv[:, 0] - 1e-12) and np.all(ex <= iv[:, 1] + 1e-12)
    # tree-only circuits get exact intervals
    assert np.all(iv[:, 1] >= iv[:, 0])


def test_cutting_exact_on_tree():
    circ = parse_bench("INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(o)\nx = AND(a,b)\no = OR(x,c)\n")
    p = [0.3, 0.6, 0.2]
    iv = cutting_bounds(circ, p)
    ex = exact_probabilities(circ, p)
    assert np.allclose(iv[:, 0], ex) and np.allclose(iv[:, 1], ex)
    assert reconvergent_stems(circ) == []


@pytest.mark.parametrize("seed", range(10))
def test_bdd_matches_enumeration(seed):
    n = 8
    circ = random_circuit(n, 30, seed=seed, max_fanin=3)
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    ex = exact_probabilities(circ, p)
    bp = bdd_probabilities(circ, p)
    assert np.max(np.abs(ex - bp)) < 1e-12


def test_bdd_c17_and_e3():
    for txt, p in ((C17, [0.5] * 5), (EX3, [0.7, 0.4, 0.55])):
        circ = parse_bench(txt)
        assert np.max(np.abs(exact_probabilities(circ, p) - bdd_probabilities(circ, p))) < 1e-12
