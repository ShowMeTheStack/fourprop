# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
import numpy as np
from fourprop.circuit import parse_bench
from fourprop.justify import ascend
from fourprop.exact import exact_probabilities


def test_ascent_raises_true_probability():
    # 4-input AND: P(out=1) = 1/16 at p=½; ascent should push every p up
    circ = parse_bench("INPUT(a)\nINPUT(b)\nINPUT(c)\nINPUT(d)\nOUTPUT(o)\nx = AND(a,b)\ny = AND(c,d)\no = AND(x,y)\n")
    t = circ.name_to_id["o"]
    p, tr = ascend(circ, t, 1, np.full(4, 0.5), steps=15)
    assert exact_probabilities(circ, p)[t] > 0.5 and tr[-1] > tr[0]
    p0, tr0 = ascend(circ, t, 0, np.full(4, 0.5), steps=5)
    assert exact_probabilities(circ, p0)[t] < 1 / 16
