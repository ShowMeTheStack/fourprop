#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Compute the gate criteria from the gate results and write GATE_VERDICT.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908"]
REGIMES = ["half", "rand1"]


def load(root: Path, tid: str):
    f = root / "results" / f"{tid}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d if d.get("ok") else d


def width_from_delta(P, delta):
    P = np.asarray(P); d = np.asarray(delta)
    return np.minimum(P + d, 1.0) - np.maximum(P - d, 0.0)


def analyse(root: Path):
    rows = []
    for c in CIRCUITS:
        for r in REGIMES:
            truth = load(root, f"bdd__{c}__{r}")
            src = "bdd"
            if not truth or not truth.get("ok"):
                truth = load(root, f"mc__{c}__{r}"); src = "mc"
            if not truth or not truth.get("ok"):
                rows.append({"circuit": c, "regime": r, "status": "no ground truth"}); continue
            ex = np.array(truth["P"])
            row = {"circuit": c, "regime": r, "truth": src, "status": "ok", "n_nodes": len(ex)}
            cut = load(root, f"cut__{c}__{r}__reconvergent")
            if cut and cut.get("ok"):
                lo, hi = np.array(cut["lo"]), np.array(cut["hi"])
                row["cut_width_median"] = float(np.median(hi - lo)); row["cut_width_mean"] = float(np.mean(hi - lo))
                row["cut_sound"] = bool(np.all(ex >= lo - 1e-9) and np.all(ex <= hi + 1e-9))
                row["cut_lt_half"] = float(np.mean((hi - lo) < 0.5))
                row["cut_stems"] = cut.get("n_reconvergent_stems")
            for k in (0, 1, 2, 3):
                fp = load(root, f"fp__{c}__{r}__k{k}")
                if not fp:
                    continue
                if not fp.get("ok"):
                    row[f"k{k}_error"] = fp.get("error", "?")[:60]; continue
                P = np.array(fp["P"]); d = np.array(fp.get("e", fp["delta"]))
                err = np.abs(np.clip(P, 0, 1) - ex)
                row[f"k{k}_mean"] = float(err.mean()); row[f"k{k}_p95"] = float(np.quantile(err, 0.95)); row[f"k{k}_max"] = float(err.max())
                row[f"k{k}_rt"] = float(fp["runtime_propagate"])
                w = width_from_delta(P, d)
                row[f"k{k}_cert_median"] = float(np.median(w)); row[f"k{k}_cert_mean"] = float(np.mean(w))
                row[f"k{k}_cert_sound"] = bool(np.all(np.abs(P - ex) <= d + 1e-9))
                row[f"k{k}_cert_lt_half"] = float(np.mean(w < 0.5))
                if "cut_width_median" in row:
                    row[f"k{k}_tighter_frac"] = float(np.mean(w < (hi - lo) - 1e-12))
            rows.append(row)
    return rows


def decide(rows):
    ok = [r for r in rows if r["status"] == "ok" and "k1_mean" in r and "k2_mean" in r]
    crit_a = crit_b = 0; n = 0
    sound_fail = []
    for r in ok:
        n += 1
        if r.get("k2_cert_median", 9) < r.get("cut_width_median", 0):
            crit_a += 1
        if r["k2_mean"] <= 0.5 * r["k1_mean"] + 1e-12 and r["k2_rt"] <= 10 * max(r["k1_rt"], 1e-3):
            crit_b += 1
        for k in (0, 1, 2, 3):
            if r.get(f"k{k}_cert_sound") is False:
                sound_fail.append((r["circuit"], r["regime"], k))
        if r.get("cut_sound") is False:
            sound_fail.append((r["circuit"], r["regime"], "cut"))
    go = n > 0 and (crit_a * 2 >= n or crit_b * 2 >= n)
    return go, crit_a, crit_b, n, sound_fail


def write_verdict(root: Path) -> str:
    rows = analyse(root)
    go, a, b, n, sound_fail = decide(rows)
    L = []
    L.append(f"# Gate criteria: **{'met' if go else 'not met'}**\n")
    L.append(f"Criterion A (order-2 certified median width < cutting-algorithm median width): {a}/{n} circuit-regimes.")
    L.append(f"Criterion B (order-2 mean error <= 0.5 x order-1 mean error at <= 10x runtime): {b}/{n} circuit-regimes.")
    L.append(f"Rule: met if A or B holds on at least half. Soundness violations: {sound_fail or 'none'}\n")
    cols = ["circuit", "regime", "truth", "n_nodes", "cut_stems", "k0_mean", "k1_mean", "k2_mean", "k3_mean", "k1_p95", "k2_p95",
            "k1_max", "k2_max", "k1_rt", "k2_rt", "cut_width_median", "cut_lt_half", "k1_cert_median", "k2_cert_median", "k1_cert_lt_half", "k2_cert_lt_half", "k2_tighter_frac",
            "k1_cert_sound", "k2_cert_sound", "cut_sound", "status"]
    L.append("| " + " | ".join(cols) + " |"); L.append("|" + "---|" * len(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        L.append("| " + " | ".join(vals) + " |")
    errs = [(r["circuit"], r["regime"], k, r[k]) for r in rows for k in r if k.endswith("_error")]
    if errs:
        L.append("\nTask errors:"); L += [f"- {e}" for e in errs]
    txt = "\n".join(L) + "\n"
    (root / "GATE_VERDICT.md").write_text(txt)
    json.dump(rows, open(root / "gate_rows.json", "w"), indent=1)
    return txt


if __name__ == "__main__":
    print(write_verdict(Path(sys.argv[1])))
