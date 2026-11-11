# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import numpy as np, tempfile, os
from fourprop.liberty import read_liberty, parse_function, load_mapped_verilog
from fourprop.exact import exact_probabilities

LIB = """
library (tiny) {
  cell (nand2) { area : 1.0;
    pin (A) { direction : input; capacitance : 0.002; }
    pin (B) { direction : input; capacitance : 0.003; }
    pin (Y) { direction : output; function : "!(A&B)"; }
  }
  cell (a21oi) { area : 2.0;
    pin (A1) { direction : input; capacitance : 0.0015; }
    pin (A2) { direction : input; capacitance : 0.0015; }
    pin (B1) { direction : input; capacitance : 0.0020; }
    pin (Y) { direction : output; function : "!((A1&A2)|B1)"; }
  }
  cell (xor2) {
    pin (A) { direction : input; capacitance : 0.004; }
    pin (B) { direction : input; capacitance : 0.004; }
    pin (X) { direction : output; function : "(A^B)"; }
  }
}
"""
V = """
module top(a, b, c, y);
input a; input b; input c; output y;
wire n1; wire n2;
nand2 u1 ( .A(a), .B(b), .Y(n1) );
a21oi u2 ( .A1(n1), .A2(c), .B1(b), .Y(n2) );
xor2 u3 ( .A(n2), .B(a), .X(y) );
endmodule
"""


def test_liberty_and_mapped():
    d = tempfile.mkdtemp(); lib = os.path.join(d, "t.lib"); v = os.path.join(d, "t.v")
    open(lib, "w").write(LIB); open(v, "w").write(V)
    cells = read_liberty(lib)
    assert set(cells) == {"nand2", "a21oi", "xor2"} and cells["nand2"].inputs["B"] == 0.003 and cells["a21oi"].outputs["Y"] == "!((A1&A2)|B1)"
    assert parse_function("A B + C'") == ("or", [("and", [("var", "A"), ("var", "B")]), ("not", ("var", "C"))])
    circ, nets = load_mapped_verilog(v, cells)
    p = [0.3, 0.6, 0.5]
    P = exact_probabilities(circ, p)
    # brute-force check of y = (!((!(a&b)&c)|b)) ^ a
    exp = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                n1 = 1 - (a & b); n2 = 1 - ((n1 & c) | b); y = n2 ^ a
                exp += y * (p[0] if a else 1 - p[0]) * (p[1] if b else 1 - p[1]) * (p[2] if c else 1 - p[2])
    assert abs(P[circ.outputs[0]] - exp) < 1e-12
    caps = {n.name: n.load_cap for n in nets}
    assert abs(caps["a"] - (0.002 + 0.004)) < 1e-12 and abs(caps["b"] - (0.003 + 0.002)) < 1e-12 and abs(caps["n1"] - 0.0015) < 1e-12
