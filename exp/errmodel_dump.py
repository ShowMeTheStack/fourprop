#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Error-model figure helper: per-net signed errors (measured vs model M1) for a few enumerable control circuits.

The errmodel runner stores only summary statistics; this re-runs the M1 recursion for the named circuits and writes the
per-net arrays (subsampled to at most --max nets per circuit/order) to a JSON for make_figures.fig_errmodel.
Usage (on mdev): errmodel_dump.py --bench /extra/fourprop/bench --run2 /extra/fourprop/results_run2 \
                   --ablation /extra/fourprop/ext/ablation --out m1_nets.json --circuits ctrl int2float cavlc
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from errmodel_runner import _abl, _propagate_signed, KS_M1
from ext_common import load_truth, regime_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--run2", required=True); ap.add_argument("--ablation", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--circuits", nargs="+", default=["ctrl", "int2float", "cavlc"])
    ap.add_argument("--regime", default="half"); ap.add_argument("--max", type=int, default=400); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from fourprop.circuit import load_bench
    from fourprop.exactgpu import exact_coeffs_deg2
    import math
    out = {}
    for c in a.circuits:
        path = Path(a.bench) / "epfl_bench" / f"{c}.bench"
        circ = load_bench(str(path)); n = circ.n_inputs; N = len(circ.nodes)
        p = regime_p(n, a.regime, c)
        truth, tkind, _ = load_truth(Path(a.run2), "epfl", c, a.regime)
        bb = max(10, min(20, int(math.log2(1.5e9 / (8 * max(1, N))))))
        Pex, A1, A2 = exact_coeffs_deg2(circ, p, device=a.device, batch_bits=bb)
        Pex = np.asarray(Pex); A1 = np.asarray(A1); A2 = np.asarray(A2)
        rng = np.random.default_rng(1)
        for k in KS_M1:
            def eps_src(i, f, g, op, k=k):
                v = Pex[f] * Pex[g] + (A1[f] @ A1[g] if k >= 1 else 0.0) + (0.5 * np.sum(A2[f] * A2[g]) if k >= 2 else 0.0)
                exact = 1.0 - Pex[i] if circ.nodes[i].inv else Pex[i]
                return float(v - exact)
            d = _abl(Path(a.ablation), "epfl", c, a.regime, k)
            if d is None:
                continue
            meas = np.array(d["P"]) - truth
            pred = _propagate_signed(circ, Pex, None, eps_src)
            good = np.where(np.isfinite(meas) & np.isfinite(pred) & (np.arange(N) >= n))[0]
            sel = np.sort(rng.choice(good, size=min(a.max, len(good)), replace=False))
            out[f"{c}__k{k}"] = {"circuit": c, "k": k, "regime": a.regime, "truth": tkind, "n_nets": int(len(good)),
                                 "meas": meas[sel].round(7).tolist(), "pred": pred[sel].round(7).tolist(),
                                 "corr": float(np.corrcoef(meas[good], pred[good])[0, 1])}
            print(c, k, "corr %.3f" % out[f"{c}__k{k}"]["corr"], "nets", len(good))
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
