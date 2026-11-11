# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Parser for gate-level Verilog netlists built from primitives (ISCAS-85 .v files)."""
from __future__ import annotations
import re
from typing import Dict, List
from .circuit import Circuit, Node

_PRIM = {"and": ("AND", False), "nand": ("AND", True), "or": ("OR", False), "nor": ("OR", True),
         "xor": ("XOR", False), "xnor": ("XOR", True), "not": ("NOT", False), "buf": ("BUF", False)}


def parse_verilog_primitives(text: str, name: str = "") -> Circuit:
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    inputs: List[str] = []; outputs: List[str] = []
    gates: Dict[str, tuple] = {}; order: List[str] = []
    for stmt in text.split(";"):
        s = " ".join(stmt.split())
        if not s:
            continue
        m = re.match(r"^(input|output|wire)\s+(.*)$", s)
        if m:
            names = [x.strip() for x in m.group(2).split(",") if x.strip()]
            if m.group(1) == "input":
                inputs += names
            elif m.group(1) == "output":
                outputs += names
            continue
        m = re.match(r"^(\w+)\s+\w*\s*\((.*)\)$", s)
        if m and m.group(1).lower() in _PRIM:
            op, inv = _PRIM[m.group(1).lower()]
            ports = [x.strip() for x in m.group(2).split(",")]
            out, ins = ports[0], ports[1:]
            gates[out] = (op, inv, ins); order.append(out)
            continue
        if s.startswith(("module", "endmodule")):
            continue
        raise ValueError(f"unrecognised statement: {s[:60]!r}")
    nodes: List[Node] = []; name_to_id: Dict[str, int] = {}; in_ids: List[int] = []
    for nm in inputs:
        if nm not in name_to_id:
            name_to_id[nm] = len(nodes); in_ids.append(len(nodes)); nodes.append(Node(nm, "INPUT"))
    state: Dict[str, int] = {}

    def visit(nm: str) -> None:
        if nm in name_to_id:
            return
        if state.get(nm) == 1:
            raise ValueError(f"cycle at {nm}")
        if nm not in gates:
            raise ValueError(f"undefined signal {nm}")
        state[nm] = 1
        op, inv, ins = gates[nm]
        for a in ins:
            visit(a)
        name_to_id[nm] = len(nodes); nodes.append(Node(nm, op, [name_to_id[a] for a in ins], inv)); state[nm] = 2

    import sys
    sys.setrecursionlimit(max(10000, 10 * len(gates) + 1000))
    for nm in order:
        visit(nm)
    return Circuit(nodes, name_to_id, in_ids, [name_to_id[o] for o in outputs], name)


def load_verilog(path: str) -> Circuit:
    import os
    with open(path) as f:
        return parse_verilog_primitives(f.read(), name=os.path.splitext(os.path.basename(path))[0])
