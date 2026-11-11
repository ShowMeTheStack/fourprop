# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import numpy as np
from fourprop.synth import random_circuit
from fourprop.exact import exact_probabilities, exact_coefficients
from fourprop.exactgpu import exact_coeffs_deg2


def test_exactgpu_numpy_matches_reference():
    n = 7
    circ = random_circuit(n, 25, seed=4, max_fanin=3)
    p = np.random.default_rng(4).uniform(0.1, 0.9, size=n)
    P, A1, A2 = exact_coeffs_deg2(circ, p, device=None, batch_bits=5)
    assert np.max(np.abs(P - exact_probabilities(circ, p))) < 1e-12
    ref = exact_coefficients(circ, p, k=2)
    for node in range(len(circ.nodes)):
        for T, v in ref[node].items():
            if len(T) == 1:
                assert abs(A1[node, T[0]] - v) < 1e-10
            elif len(T) == 2:
                assert abs(A2[node, T[0], T[1]] - v) < 1e-10 and abs(A2[node, T[1], T[0]] - v) < 1e-10
