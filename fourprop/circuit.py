# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Circuit representation and ISCAS/ABC .bench parser.

Nodes are stored in topological order. Each node has an op in
{INPUT, AND, OR, XOR, NOT, BUF, CONST0, CONST1}, a list of fan-in node ids and an
`inv` flag meaning the gate output is complemented (NAND = AND + inv, etc.).
Multi-input AND/OR/XOR gates are kept n-ary; the propagation engine chains them.
DFF(x) in ISCAS-89 files is cut: the DFF output becomes a pseudo-input and x a pseudo-output.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_GATE_RE = re.compile(r"^\s*(\S+)\s*=\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$")
_IO_RE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(\s*(\S+)\s*\)\s*$")

_OPMAP = {
    "AND": ("AND", False), "NAND": ("AND", True),
    "OR": ("OR", False), "NOR": ("OR", True),
    "XOR": ("XOR", False), "XNOR": ("XOR", True),
    "NOT": ("NOT", False), "BUF": ("BUF", False), "BUFF": ("BUF", False),
}


@dataclass
class Node:
    name: str
    op: str
    fanins: List[int] = field(default_factory=list)
    inv: bool = False


@dataclass
class Circuit:
    nodes: List[Node]
    name_to_id: Dict[str, int]
    inputs: List[int]            # node ids of primary (and pseudo) inputs, in order
    outputs: List[int]           # node ids of primary (and pseudo) outputs
    name: str = ""

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_gates(self) -> int:
        return sum(1 for nd in self.nodes if nd.op not in ("INPUT", "CONST0", "CONST1"))

    def input_index(self) -> Dict[int, int]:
        """node id -> position among inputs (the Fourier variable index)."""
        return {nid: i for i, nid in enumerate(self.inputs)}

    def depth(self) -> int:
        d = [0] * len(self.nodes)
        for i, nd in enumerate(self.nodes):
            if nd.fanins:
                d[i] = 1 + max(d[j] for j in nd.fanins)
        return max(d) if d else 0

    def cone_sizes(self) -> List[int]:
        """Number of primary inputs in the fan-in cone of each node."""
        cones: List[set] = [set() for _ in self.nodes]
        idx = self.input_index()
        for i, nd in enumerate(self.nodes):
            if nd.op == "INPUT":
                cones[i] = {idx[i]}
            else:
                s: set = set()
                for j in nd.fanins:
                    s |= cones[j]
                cones[i] = s
        return [len(s) for s in cones]


def parse_bench(text: str, name: str = "") -> Circuit:
    """Parse ISCAS-85/89 or ABC `write_bench` format. Returns a topologically ordered Circuit."""
    raw_inputs: List[str] = []
    raw_outputs: List[str] = []
    gates: Dict[str, tuple] = {}   # name -> (op, inv, [fanin names])
    order: List[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _IO_RE.match(line)
        if m:
            (raw_inputs if m.group(1) == "INPUT" else raw_outputs).append(m.group(2))
            continue
        mc = re.match(r"^\s*(\S+)\s*=\s*(vdd|gnd|1|0)\s*$", line, re.IGNORECASE)
        if mc:
            gates[mc.group(1)] = ("CONST1" if mc.group(2).lower() in ("vdd", "1") else "CONST0", False, [])
            order.append(mc.group(1))
            continue
        m = _GATE_RE.match(line)
        if not m:
            raise ValueError(f"cannot parse bench line: {line!r}")
        out, g, args = m.group(1), m.group(2).upper(), [a.strip() for a in m.group(3).split(",") if a.strip()]
        if g == "DFF":
            # cut the flip-flop: output is a pseudo-input, its D input a pseudo-output
            raw_inputs.append(out)
            raw_outputs.extend(args)
            continue
        if g in ("VDD", "CONST1"):
            gates[out] = ("CONST1", False, [])
        elif g in ("GND", "CONST0"):
            gates[out] = ("CONST0", False, [])
        elif g in _OPMAP:
            op, inv = _OPMAP[g]
            if op in ("NOT", "BUF") and len(args) != 1:
                raise ValueError(f"{g} with {len(args)} inputs: {line!r}")
            gates[out] = (op, inv, args)
        else:
            raise ValueError(f"unknown gate {g!r} in {line!r}")
        order.append(out)

    nodes: List[Node] = []
    name_to_id: Dict[str, int] = {}
    inputs: List[int] = []
    seen_in = set()
    for nm in raw_inputs:
        if nm in seen_in:
            continue
        seen_in.add(nm)
        name_to_id[nm] = len(nodes)
        inputs.append(len(nodes))
        nodes.append(Node(nm, "INPUT"))

    # topological order by DFS over gate definitions
    state: Dict[str, int] = {}

    def visit(nm: str) -> None:
        if nm in name_to_id:
            return
        if state.get(nm) == 1:
            raise ValueError(f"combinational cycle through {nm}")
        if nm not in gates:
            raise ValueError(f"undefined signal {nm!r}")
        state[nm] = 1
        op, inv, args = gates[nm]
        for a in args:
            visit(a)
        name_to_id[nm] = len(nodes)
        nodes.append(Node(nm, op, [name_to_id[a] for a in args], inv))
        state[nm] = 2

    import sys
    sys.setrecursionlimit(max(10000, 10 * len(gates) + 1000))
    for nm in order:
        visit(nm)
    outputs = [name_to_id[o] for o in raw_outputs]
    return Circuit(nodes, name_to_id, inputs, outputs, name)


def load_bench(path: str) -> Circuit:
    import os
    with open(path) as f:
        return parse_bench(f.read(), name=os.path.splitext(os.path.basename(path))[0])


def evaluate(circ: Circuit, x):
    """Evaluate all nodes on a batch of input patterns.

    x: numpy bool/uint8 array of shape (N, n_inputs). Returns array (N, n_nodes) of bools.
    """
    import numpy as np
    x = np.asarray(x, dtype=bool)
    N = x.shape[0]
    vals = np.empty((N, len(circ.nodes)), dtype=bool)
    idx = circ.input_index()
    for i, nd in enumerate(circ.nodes):
        if nd.op == "INPUT":
            vals[:, i] = x[:, idx[i]]
        elif nd.op == "CONST0":
            vals[:, i] = False
        elif nd.op == "CONST1":
            vals[:, i] = True
        elif nd.op in ("NOT", "BUF"):
            v = vals[:, nd.fanins[0]]
            vals[:, i] = ~v if nd.op == "NOT" else v
        else:
            v = vals[:, nd.fanins[0]].copy()
            for j in nd.fanins[1:]:
                w = vals[:, j]
                if nd.op == "AND":
                    v &= w
                elif nd.op == "OR":
                    v |= w
                else:
                    v ^= w
            vals[:, i] = v
        if nd.inv:
            vals[:, i] = ~vals[:, i]
    return vals
