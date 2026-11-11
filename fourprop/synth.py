# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Random reconvergent circuit generator for tests and scaling experiments."""
from __future__ import annotations
import numpy as np
from .circuit import Circuit, Node


def random_circuit(n_inputs: int, n_gates: int, seed: int = 0, ops=("AND", "OR", "XOR", "NOT"),
                   max_fanin: int = 2, locality: int | None = None, name: str = "") -> Circuit:
    """Random circuit: each gate picks fan-ins among earlier nodes (within `locality` most recent if set)."""
    rng = np.random.default_rng(seed)
    nodes = [Node(f"x{i}", "INPUT") for i in range(n_inputs)]
    for g in range(n_gates):
        op = ops[rng.integers(len(ops))]
        lo = 0 if locality is None else max(0, len(nodes) - locality)
        pool = np.arange(lo, len(nodes))
        if op == "NOT":
            fan = [int(rng.choice(pool))]
        else:
            fi = int(rng.integers(2, max_fanin + 1)) if max_fanin > 2 else 2
            fan = [int(v) for v in rng.choice(pool, size=min(fi, len(pool)), replace=False)]
        nodes.append(Node(f"g{g}", op, fan, inv=bool(rng.random() < 0.2) if op != "NOT" else False))
    name_to_id = {nd.name: i for i, nd in enumerate(nodes)}
    outputs = [len(nodes) - 1]
    return Circuit(nodes, name_to_id, list(range(n_inputs)), outputs, name or f"rand_n{n_inputs}_g{n_gates}_s{seed}")
