#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Monte Carlo as a competitor.

Part A ("widths" tasks, CPU): per-net interval widths on the six small ISCAS AIGs at p = 1/2 and rand1: cutting
algorithm, order-1 certificate (dense, certified), order-2 certificate (sparse engine, 1 h cap). Stored per net for the
CDF figure and for the comparison with Monte Carlo confidence intervals (analytic from the truth) in the verdict.

Part B ("cv0"/"cv1" tasks, GPU): plain vs control-variate Monte Carlo (order 1 for all nets, order 2 for the primary
outputs when the order-2 states fit) at N in {1e4, 1e5, 1e6}, seeds 1..3; variance-reduction factors per net
(empirical var_f / var_residual), the prediction 1/(1 - rho_1) from the state, and the certified lower bound
P(1-P)/delta^2.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from ext_common import CIRCUITS, REGIMES, regime_p, load_truth, errors, circuit_class  # noqa: E402

NS = [10 ** 4, 10 ** 5, 10 ** 6]
SEEDS = [1, 2, 3]
SMALL6 = ["c17", "c432", "c499", "c880", "c1355", "c1908"]
TIMEOUTS = {"cv0": 2 * 3600, "cv1": 2 * 3600, "widths": 3 * 3600 + 600}
A2_BUDGET = 2 * 1024 ** 3


def task_list(bench: Path):
    tasks = []; i = 0
    for fam, c, rel in CIRCUITS:
        if not (bench / rel).exists():
            continue
        for reg in REGIMES:
            if fam == "aig" and c in SMALL6:
                tasks.append({"id": f"widths__{c}__{reg}", "kind": "widths", "family": fam, "circuit": c, "regime": reg, "path": str(bench / rel)})
            for N in NS:
                for s in SEEDS:
                    kind = "cv0" if i % 2 == 0 else "cv1"; i += 1
                    tasks.append({"id": f"cv__{fam}__{c}__{reg}__N{N}__s{s}", "kind": kind, "family": fam, "circuit": c, "regime": reg, "N": N, "seed": s, "path": str(bench / rel)})
    return tasks


def run_task(task: dict, extra: dict) -> dict:
    from fourprop.circuit import load_bench
    from fourprop.dense import propagate_dense
    circ = load_bench(task["path"]); n = circ.n_inputs; N = len(circ.nodes)
    p = regime_p(n, task["regime"], task["circuit"])
    truth, tkind, _ = load_truth(Path(extra["run2"]), task["family"], task["circuit"], task["regime"])
    res = {"n_inputs": n, "n_nodes": N, "truth": tkind, "load_start": os.getloadavg()[0]}
    if task["kind"] == "widths":
        from fourprop.cutting import cutting_bounds
        from fourprop.fourier import propagate
        t0 = time.time(); iv = cutting_bounds(circ, p, policy="reconvergent"); res["t_cut"] = time.time() - t0
        res["w_cut"] = [float(x) for x in (iv[:, 1] - iv[:, 0])]
        t0 = time.time(); P1, _, _, E1, D1 = propagate_dense(circ, p, k=1, backend="numpy", certify=True); res["t_cert1"] = time.time() - t0
        res["w_cert1"] = [float(min(2 * e, 1.0)) for e in E1]; res["delta1"] = [float(x) for x in D1]
        t0 = time.time(); out = propagate(circ, p, k=2, certify=True, clamp=False); res["t_cert2"] = time.time() - t0
        res["w_cert2"] = [float(min(2 * r.e, 1.0)) for r in out]
        res["truth_P"] = [float(x) for x in truth]
        return res
    import torch
    from fourprop.cv import cv_monte_carlo
    dev = "cuda:0" if task["kind"] == "cv0" else "cuda:1"
    # order-1 state for every net (+ certified radius) and order-2 state for the outputs when it fits
    P1, _, k1, E1, D1 = propagate_dense(circ, p, k=1, backend=dev, keep_states=True, certify=True)
    A1 = np.stack([st[1] for st in k1]); P1v = np.array([st[0] for st in k1])
    A2 = None; nets2 = None
    if N * n * n * 8 <= A2_BUDGET:
        _, _, k2 = propagate_dense(circ, p, k=2, backend=dev, keep_states=True, project="norm")
        nets2 = np.array(sorted(set(circ.outputs))); A2 = np.stack([k2[i][2] for i in nets2])
        P1v2 = np.array([k2[i][0] for i in nets2]); a2 = np.stack([k2[i][1] for i in nets2])
        # use the order-2 state's own P and a for the order-2 variate
        A1_for2 = A1.copy(); A1_for2[nets2] = a2; P1_for2 = P1v.copy(); P1_for2[nets2] = P1v2
    else:
        A1_for2 = A1; P1_for2 = P1v
    out = cv_monte_carlo(circ, p, P1_for2, A1_for2, task["N"], seed=task["seed"], device=dev, A2=A2, nets2=nets2)
    # order-1 CV must use the order-1 state for all nets: recompute cheaply for the outputs if order-2 replaced them
    plain, cv1 = out["plain"], out["cv1"]
    res.update({"N": task["N"], "seed": task["seed"], "device": dev, "t_sim": out["t_sim"], "t_cv": out["t_cv"], "t_total": out["t_total"], "batch": out["batch"]})
    res["plain"] = errors(plain, truth); res["cv1"] = errors(cv1, truth)
    var_f = out["var_f"]; var_r = out["var_r1"]
    good = (var_f > 1e-9) & (var_r > 1e-12)
    vrf = np.where(good, var_f / np.maximum(var_r, 1e-12), np.nan)
    res["vrf1"] = {"median": float(np.nanmedian(vrf)), "q1": float(np.nanquantile(vrf, 0.25)), "q3": float(np.nanquantile(vrf, 0.75)), "mean": float(np.nanmean(vrf)), "n_nets": int(good.sum())}
    rho1 = np.sum(A1 * A1, axis=1) / np.maximum(P1v * (1 - P1v), 1e-12)
    pred = 1.0 / np.maximum(1.0 - rho1, 1e-6)
    res["pred1"] = {"median": float(np.nanmedian(pred[good])), "q1": float(np.nanquantile(pred[good], 0.25)), "q3": float(np.nanquantile(pred[good], 0.75))}
    with np.errstate(divide="ignore", invalid="ignore"):
        cert = np.where(D1 > 0, P1v * (1 - P1v) / (D1 * D1), np.nan)
    res["certbound1"] = {"median": float(np.nanmedian(cert[good])), "frac_informative": float(np.mean(cert[good] > 1.0))}
    if A2 is not None:
        res["cv2"] = errors(out["cv2"], truth[nets2]); res["plain_outputs"] = errors(plain[nets2], truth[nets2]); res["cv1_outputs"] = errors(cv1[nets2], truth[nets2])
        vf2 = var_f[nets2]; vr2 = out["var_r2"]; g2 = (vf2 > 1e-9) & (vr2 > 1e-12)
        v2 = np.where(g2, vf2 / np.maximum(vr2, 1e-12), np.nan)
        res["vrf2"] = {"median": float(np.nanmedian(v2)), "q1": float(np.nanquantile(v2, 0.25)), "q3": float(np.nanquantile(v2, 0.75)), "n_nets": int(g2.sum())}
    if task["N"] == NS[-1] and task["seed"] == SEEDS[0]:
        res["vrf1_per_net"] = [None if not np.isfinite(x) else float(x) for x in vrf]
        res["rho1_per_net"] = [float(x) for x in rho1]
    return res


def write_verdict(root: Path) -> str:
    rows = []; wrows = []
    for f in sorted((root / "results").glob("*.json")):
        d = json.load(open(f))
        if not d.get("ok"):
            continue
        t = d["task"]
        if t["kind"] == "widths":
            wc = np.array(d["w_cut"]); w1 = np.array(d["w_cert1"]); w2 = np.array(d["w_cert2"]); P = np.array(d["truth_P"])
            r = {"circuit": t["circuit"], "regime": t["regime"], "n_nets": len(P), "cut_median": float(np.median(wc)), "cert1_median": float(np.median(w1)), "cert2_median": float(np.median(w2))}
            for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
                wmc = 2 * 1.96 * np.sqrt(np.maximum(P * (1 - P), 0) / N)
                r[f"cert1_narrower_N{N}"] = float(np.mean(w1 < wmc)); r[f"cert2_narrower_N{N}"] = float(np.mean(w2 < wmc)); r[f"cut_narrower_N{N}"] = float(np.mean(wc < wmc))
            wrows.append(r)
        else:
            r = {"family": t["family"], "circuit": t["circuit"], "regime": t["regime"], "class": circuit_class(t["circuit"]), "N": t["N"], "seed": t["seed"],
                 "plain_err": d["plain"]["mean_err"], "cv1_err": d["cv1"]["mean_err"], "vrf1_median": d["vrf1"]["median"], "vrf1_q1": d["vrf1"]["q1"], "vrf1_q3": d["vrf1"]["q3"],
                 "pred1_median": d["pred1"]["median"], "certbound1_median": d["certbound1"]["median"], "cert_informative": d["certbound1"]["frac_informative"],
                 "t_sim": d["t_sim"], "t_cv": d["t_cv"], "t_total": d["t_total"]}
            if "cv2" in d:
                r.update({"cv2_err_out": d["cv2"]["mean_err"], "plain_err_out": d["plain_outputs"]["mean_err"], "cv1_err_out": d["cv1_outputs"]["mean_err"], "vrf2_median": d["vrf2"]["median"], "vrf2_q1": d["vrf2"]["q1"], "vrf2_q3": d["vrf2"]["q3"]})
            rows.append(r)
    rows.sort(key=lambda r: (r["family"], r["circuit"], r["regime"], r["N"], r["seed"]))
    L = ["# Monte Carlo as competitor\n", "## Part B: control variates (N = 1e6, mean over seeds; VRF = var(f)/var(f - f~) per net, median over nets)\n",
         "| family | circuit | p | class | plain err | cv1 err | VRF1 med [q1,q3] | pred 1/(1-rho1) | cert bound | cv2 err (outputs) | plain err (outputs) | VRF2 med | cv overhead t_cv/t_sim |", "|---|" * 13]
    keys = sorted({(r["family"], r["circuit"], r["regime"]) for r in rows})
    for key in keys:
        rs = [r for r in rows if (r["family"], r["circuit"], r["regime"]) == key and r["N"] == NS[-1]]
        if not rs:
            continue
        m = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None
        f = lambda v, d=3: "" if v is None else f"{v:.{d}g}"
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {rs[0]['class']} | {f(m('plain_err'))} | {f(m('cv1_err'))} | {f(m('vrf1_median'))} [{f(m('vrf1_q1'))},{f(m('vrf1_q3'))}] | {f(m('pred1_median'))} | {f(m('certbound1_median'))} | {f(m('cv2_err_out'))} | {f(m('plain_err_out'))} | {f(m('vrf2_median'))} | {f(m('t_cv')/max(m('t_sim'),1e-9) if m('t_sim') else None, 2)} |")
    L += ["\n## Part A: fraction of nets whose interval is narrower than the Monte Carlo 95% CI at N patterns (six small ISCAS AIGs)\n",
          "| circuit | p | nets | cut med | cert1 med | cert2 med | cert1<MC 1e3 | cert1<MC 1e4 | cert2<MC 1e3 | cert2<MC 1e4 | cert2<MC 1e5 | cut<MC 1e4 |", "|---|" * 12]
    for r in sorted(wrows, key=lambda r: (SMALL6.index(r["circuit"]), r["regime"])):
        L.append(f"| {r['circuit']} | {r['regime']} | {r['n_nets']} | {r['cut_median']:.3g} | {r['cert1_median']:.3g} | {r['cert2_median']:.3g} | {r['cert1_narrower_N1000']:.2f} | {r['cert1_narrower_N10000']:.2f} | {r['cert2_narrower_N1000']:.2f} | {r['cert2_narrower_N10000']:.2f} | {r['cert2_narrower_N100000']:.2f} | {r['cut_narrower_N10000']:.2f} |")
    txt = "\n".join(L) + "\n"; (root / "E12_RESULTS.md").write_text(txt)
    json.dump(rows, open(root / "e12b_rows.json", "w"), indent=1); json.dump(wrows, open(root / "e12a_rows.json", "w"), indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--run2", default="/extra/fourprop/run2"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(); root = Path(a.root)
    return run_tasks(root, task_list(Path(a.bench)), "cv_runner", "run_task", {"run2": a.run2}, TIMEOUTS, workers=a.workers,
                     serial_kinds=("cv0", "cv1", "widths"), kind_order={"widths": 0, "cv0": 1, "cv1": 1}, on_done=lambda r: print(write_verdict(root)))


if __name__ == "__main__":
    sys.exit(main())
