# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Minimal Liberty (.lib) reader: cell names, input-pin capacitances, output-pin functions; plus a parser
for Liberty Boolean function strings and a loader for Yosys-mapped structural Verilog."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .circuit import Circuit, Node


@dataclass
class Cell:
    name: str
    inputs: Dict[str, float] = field(default_factory=dict)      # pin -> capacitance
    outputs: Dict[str, str] = field(default_factory=dict)       # pin -> function string
    area: float = 0.0


def read_liberty(path: str) -> Dict[str, Cell]:
    """Tokenise on braces and semicolons (comments stripped), track cell/pin nesting, collect attributes."""
    text = open(path, errors="ignore").read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S); text = re.sub(r"//.*", "", text)
    toks = re.split(r"([{};])", text)
    cells: Dict[str, Cell] = {}
    stack: List[str] = []            # block headers
    cell: Optional[Cell] = None; pin: Optional[str] = None; pin_dir = pin_cap = pin_fn = None
    cell_re = re.compile(r"^\s*cell\s*\(\s*\"?([^\")]+)\"?\s*\)\s*$"); pin_re = re.compile(r"^\s*pin\s*\(\s*\"?([^\")]+)\"?\s*\)\s*$")
    header = ""
    for t in toks:
        if t == "{":
            stack.append(header.strip())
            m = cell_re.match(header)
            if m and cell is None:
                cell = Cell(m.group(1))
            m = pin_re.match(header)
            if m and cell is not None and pin is None:
                pin = m.group(1); pin_dir = pin_cap = pin_fn = None
            header = ""
        elif t == "}":
            h = stack.pop() if stack else ""
            if pin is not None and pin_re.match(h):
                if pin_dir == "input":
                    cell.inputs[pin] = pin_cap if pin_cap is not None else 0.0
                elif pin_dir == "output" and pin_fn:
                    cell.outputs[pin] = pin_fn
                pin = None
            elif cell is not None and cell_re.match(h):
                cells[cell.name] = cell; cell = None
            header = ""
        elif t == ";":
            st = header.strip(); header = ""
            if cell is None or not st:
                continue
            mm = re.match(r"^(\w+)\s*:\s*(.*)$", st, re.S)
            if not mm:
                continue
            key, val = mm.group(1), mm.group(2).strip().strip('"')
            if pin is not None:
                if key == "direction": pin_dir = val
                elif key == "capacitance" and pin_cap is None:
                    try: pin_cap = float(val)
                    except ValueError: pass
                elif key == "function": pin_fn = val
            elif key == "area":
                try: cell.area = float(val)
                except ValueError: pass
        else:
            header += t
    return cells


# ---- Liberty Boolean expression parser -> gate nodes ----
_TOK = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_\[\]]*|[()!'&*|+^]|0|1)")


def parse_function(expr: str):
    """Return an AST: ('var', name) | ('const', 0/1) | ('not', a) | ('and'|'or'|'xor', [args])."""
    toks = _TOK.findall(expr); pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = toks[pos[0]]; pos[0] += 1; return t

    def primary():
        t = take()
        if t == "(":
            e = orexpr(); assert take() == ")"
        elif t == "!":
            e = ("not", primary())
        elif t in ("0", "1"):
            e = ("const", int(t))
        else:
            e = ("var", t)
        while peek() == "'":
            take(); e = ("not", e)
        return e

    def andexpr():
        args = [primary()]
        while True:
            t = peek()
            if t in ("&", "*"):
                take(); args.append(primary())
            elif t is not None and t not in (")", "|", "+", "^"):   # juxtaposition = AND
                args.append(primary())
            else:
                break
        return args[0] if len(args) == 1 else ("and", args)

    def xorexpr():
        args = [andexpr()]
        while peek() == "^":
            take(); args.append(andexpr())
        return args[0] if len(args) == 1 else ("xor", args)

    def orexpr():
        args = [xorexpr()]
        while peek() in ("|", "+"):
            take(); args.append(xorexpr())
        return args[0] if len(args) == 1 else ("or", args)

    e = orexpr()
    assert pos[0] == len(toks), f"trailing tokens in {expr!r}"
    return e


@dataclass
class MappedNet:
    name: str
    node: int                 # Circuit node id carrying this net
    load_cap: float = 0.0     # sum of input-pin capacitances of fanout cells
    is_input: bool = False
    driver: Optional[Tuple[str, str]] = None   # (instance, output pin) for cell-driven nets; None for ports/assigns


def load_mapped_verilog(path: str, cells: Dict[str, Cell], name: str = "") -> Tuple[Circuit, List[MappedNet]]:
    """Parse Yosys `write_verilog -noattr` output with named port connections into a Circuit whose nodes
    are the expanded cell functions. Returns the circuit and the list of real nets (cell outputs + inputs)."""
    text = re.sub(r"//.*", "", open(path).read()); text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    inputs: List[str] = []; outputs: List[str] = []; insts = []; assigns = []
    for stmt in text.split(";"):
        s = " ".join(stmt.split())
        if not s:
            continue
        m = re.match(r"^(input|output|wire)\s+(.*)$", s)
        if m:
            names = []
            for x in m.group(2).split(","):
                x = x.strip()
                mv = re.match(r"^(\[\d+:\d+\])?\s*(\S+)$", x)
                if mv:
                    names.append(mv.group(2))
            if m.group(1) == "input": inputs += names
            elif m.group(1) == "output": outputs += names
            continue
        m = re.match(r"^assign\s+(\S+)\s*=\s*(.+)$", s)
        if m:
            assigns.append((m.group(1), m.group(2).strip())); continue
        m = re.match(r"^(\S+)\s+(\S+)\s*\((.*)\)$", s)
        if m and m.group(1) in cells:
            conns = dict(re.findall(r"\.(\w+)\s*\(\s*([^()]*?)\s*\)", m.group(3)))
            insts.append((m.group(1), m.group(2), conns)); continue
        if m and m.group(1) not in cells and not re.search(r"\.\w+\s*\(\s*[^()\s]", m.group(3) or ""):
            continue   # filler / tap / decap instance without logical connections
        if m and m.group(1).upper().startswith(("FILL", "TAP", "DECAP", "ENDCAP")):
            continue
        if s.startswith(("module", "endmodule", "timescale", "`")):
            continue
        raise ValueError(f"unparsed: {s[:80]!r}")
    nodes: List[Node] = []; name_to_id: Dict[str, int] = {}
    nets: Dict[str, MappedNet] = {}
    in_ids = []
    for nm in inputs:
        name_to_id[nm] = len(nodes); in_ids.append(len(nodes)); nodes.append(Node(nm, "INPUT")); nets[nm] = MappedNet(nm, name_to_id[nm], 0.0, True)
    const_ids: Dict[int, int] = {}

    def const(v: int) -> int:
        if v not in const_ids:
            const_ids[v] = len(nodes); nodes.append(Node(f"const{v}", "CONST1" if v else "CONST0"))
        return const_ids[v]

    # drivers: net -> (celltype, pin fn, conns) ; resolve lazily in topological order by recursion
    driver: Dict[str, tuple] = {}
    for ctype, iname, conns in insts:
        for opin, fn in cells[ctype].outputs.items():
            if opin in conns:
                driver[conns[opin]] = (ctype, iname, fn, conns, opin)
    for lhs, rhs in assigns:
        driver[lhs] = ("assign", None, rhs, None, None)
    state: Dict[str, int] = {}

    def net_node(nm: str) -> int:
        if nm in ("1'b0", "1'h0"): return const(0)
        if nm in ("1'b1", "1'h1"): return const(1)
        if nm in name_to_id:
            return name_to_id[nm]
        if state.get(nm) == 1:
            raise ValueError(f"cycle at {nm}")
        if nm not in driver:
            raise ValueError(f"undriven net {nm}")
        state[nm] = 1
        ctype, iname, fn, conns, opin = driver[nm]
        if ctype == "assign":
            nid = net_node(fn)
            name_to_id[nm] = nid
        else:
            nid = build(parse_function(fn), conns, f"{iname}.")
            name_to_id[nm] = nid; nodes[nid].name = nm
        nets[nm] = MappedNet(nm, nid, driver=None if ctype == "assign" else (iname, opin)); state[nm] = 2
        return nid

    def build(ast, conns, prefix) -> int:
        kind = ast[0]
        if kind == "var":
            return net_node(conns[ast[1]])
        if kind == "const":
            return const(ast[1])
        if kind == "not":
            a = build(ast[1], conns, prefix)
            nodes.append(Node(f"{prefix}n{len(nodes)}", "NOT", [a])); return len(nodes) - 1
        args = [build(x, conns, prefix) for x in ast[1]]
        nodes.append(Node(f"{prefix}n{len(nodes)}", kind.upper(), args)); return len(nodes) - 1

    import sys
    sys.setrecursionlimit(max(10000, 20 * len(insts) + 1000))
    for ctype, iname, conns in insts:
        for opin in cells[ctype].outputs:
            if opin in conns:
                net_node(conns[opin])
    out_ids = [net_node(o) for o in outputs]
    # load capacitances
    for ctype, iname, conns in insts:
        for ipin, cap in cells[ctype].inputs.items():
            if ipin in conns and conns[ipin] in nets:
                nets[conns[ipin]].load_cap += cap
    circ = Circuit(nodes, name_to_id, in_ids, out_ids, name)
    return circ, list(nets.values())
