#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Recompute every number quoted in paper/main.tex from the result summaries.

Usage: python exp/audit_numbers.py [results_run2_dir]
Prints one block per manuscript passage; compare by eye against the sentence quoted in the header line.
"""
import json, statistics as st, sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results_run2")

def load(p):
    d = json.load(open(R / p))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d

def fmt(x, d=3):
    return "--" if x is None else (f"{x:.{d}g}" if isinstance(x, float) else str(x))

def block(title):
    print(f"\n== {title}")

proj = load("proj/proj_rows.json"); gate = load("gate/gate_rows.json"); fast = load("fast/fast_rows.json")
p2d = json.load(open(R / "phase2/phase2_rows.json")); p2 = p2d["rows"]; struct = p2d["struct"]
p2b = json.load(open(R / "phase2b/phase2_rows.json"))["rows"]
e3 = load("oracle/e3_rows.json"); e6 = load("power/e6_rows.json"); e8 = load("justify/e8_rows.json")
e8h = load("justify_hard/e8_rows.json"); e8r = load("justify_hardr/e8_rows.json"); e10 = load("timing/e10_rows.json")

# ---------------------------------------------------------------- accuracy vs order (Table 1, proj rows)
block("Sec 7.2 / abstract: 'Order 2 reduces the mean error of BAM by a factor between A and B (median M) on every circuit except c6288'")
ratios = {}
for r in proj:
    if r["status"] != "ok" or r["circuit"] in ("c17", "c6288"):
        continue
    ratios[(r["circuit"], r["regime"])] = r["k1_clamp_mean"] / r["k2_norm_mean"]
vals = sorted(ratios.values())
print("  ratios k1_clamp/k2_norm:", {k: round(v, 2) for k, v in sorted(ratios.items())})
print(f"  min {vals[0]:.2f} max {vals[-1]:.2f} median {st.median(vals):.2f}; <2.0 on", [k for k, v in ratios.items() if v < 2.0])
c6 = {r["regime"]: r for r in proj if r["circuit"] == "c6288"}
print("  c6288: k1 clamp/norm/k2 norm mean", {k: (fmt(v["k1_clamp_mean"]), fmt(v["k1_norm_mean"]), fmt(v["k2_norm_mean"])) for k, v in c6.items()},
      "nonfinite clamp k1/k2", {k: (v["k1_clamp_nonfinite"], v["k2_clamp_nonfinite"]) for k, v in c6.items()})
mx = [(r["circuit"], r["regime"], r["k1_clamp_max"], r["k2_norm_max"]) for r in proj if r["status"] == "ok" and r["circuit"] not in ("c17",)]
print("  'maximum error over nets remains between X and Y': k2_norm_max range",
      f"{min(m[3] for m in mx):.3f}..{max(m[3] for m in mx):.3f}; k1_clamp_max range {min(m[2] for m in mx):.3f}..{max(m[2] for m in mx):.3f}")
same = [r["circuit"] for r in proj if r["status"] == "ok" and abs(r["k1_clamp_mean"] - r["k1_norm_mean"]) > 1e-4 * max(1e-9, r["k1_clamp_mean"])]
print("  circuits where projection changes order-1 mean by >1e-4 relative:", sorted(set(same)))

# ---------------------------------------------------------------- phase2 (Table 2)
block("Sec 7.2 para 2: bar/adder/arbiter/priority/div k1->k2; arithmetic 'between A and B at every order'; max up to; sqrt worse at k2 than k0")
def row(fam, c, reg="half"):
    return next(r for r in p2 if r["family"] == fam and r["circuit"] == c and r["regime"] == reg)
for c in ("bar", "adder", "arbiter", "priority", "div", "i2c", "max", "dec"):
    r = row("epfl", c); print(f"  {c:9} k0 {fmt(r['k0_mean'])} k1 {fmt(r['k1_mean'])} k2 {fmt(r.get('k2_mean'))}  (k1/k2 = {fmt(r['k1_mean']/r['k2_mean'] if r.get('k2_mean') else None, 3)}) nodes {r['n_nodes']} depth {r['depth']}")
arith = ("log2", "sin", "sqrt", "multiplier", "square")
vals = []; mxs = []
for c in arith:
    for reg in ("half", "rand1"):
        r = row("epfl", c, reg); vals += [r["k0_mean"], r["k1_mean"], r["k2_mean"]]; mxs += [r["k1_max"], r["k2_max"]]
        print(f"  {c:11}{reg:6} k0 {fmt(r['k0_mean'])} k1 {fmt(r['k1_mean'])} k2 {fmt(r['k2_mean'])} max k1 {fmt(r['k1_max'])} k2 {fmt(r['k2_max'])}")
print(f"  arithmetic mean-error range over orders 0-2: {min(vals):.3f}..{max(vals):.3f}; max error up to {max(mxs):.2f}")

block("Sec 7.5: 'EPFL control blocks order-1 certificate informative (cert<0.5) on A% to B%; deep arithmetic < C%'")
ctrl = ("ctrl", "cavlc", "int2float", "dec", "router", "i2c", "priority", "arbiter", "bar", "adder", "max", "voter")
for c in ctrl + arith + ("div",):
    for reg in ("half", "rand1"):
        try:
            r = row("epfl", c, reg)
        except StopIteration:
            continue
        print(f"  {c:11}{reg:6} cert<0.5 {fmt(r.get('cert_lt_0.5'), 2)} cert_median_w {fmt(r.get('cert_median_w'), 2)} sound {r.get('cert_sound')}")

block("Sec 7.3: c499 gate-level vs AIG; Table 3 spread k1->k2 per circuit; Discussion 'differ by up to 0.4 ... c2670'")
for fam in ("gate", "aig"):
    r = row(fam, "c499"); print(f"  c499 {fam:5} depth {r['depth']} k0 {fmt(r['k0_mean'])} k1 {fmt(r['k1_mean'])} k2 {fmt(r['k2_mean'])}")
sp = {}
for s in struct:
    if s["regime"] == "half":
        sp.setdefault(s["circuit"], {})[s["k"]] = (s["spread_mean"], s["spread_max"], s["variants"])
for c, d in sorted(sp.items()):
    k1, k2 = d.get(1), d.get(2)
    if k1 and k2:
        print(f"  {c:6} spread mean k1 {k1[0]:.3f} -> k2 {k2[0]:.3f} (ratio {k1[0]/k2[0] if k2[0] else float('inf'):.2f}); spread max k1 {k1[1]:.3f} k2 {k2[1]:.3f}; variants {k1[2]}")

block("phase2b (rand2/rand3 + half/rand1): family means over nine ISCAS circuits; k1/k2 ratio per circuit mean ± se")
skip = {"c17", "c6288"}
for fam in ("aig", "gate", "resyn2", "dc2"):
    sel = [x for x in p2b if x["family"] == fam and x["circuit"] not in skip and x.get("status") == "ok"]
    print(f"  {fam:7} n={len(sel)} " + " ".join(f"k{k} {st.mean(x[f'k{k}_mean'] for x in sel):.4f}" for k in (0, 1, 2)))
for c in sorted({x["circuit"] for x in p2b} - {"c17"}):
    rs = [x["k1_mean"] / x["k2_mean"] for x in p2b if x["circuit"] == c and x.get("status") == "ok" and x.get("k2_mean")]
    print(f"  {c:6} k1/k2 {st.mean(rs):.2f} ± {st.stdev(rs)/len(rs)**0.5:.2f} (n={len(rs)})")
print("  certified sound on all exact-truth rows:", all(x.get("cert_sound", True) for x in p2b if x.get("truth") == "bdd"))

# ---------------------------------------------------------------- certified vs cutting (gate rows)
block("Sec 7.5 Table 4: stems range; k2 cert<0.5 range; cutting <0.5 range; k2 tighter range (exclude c17)")
g = [r for r in gate if r["status"] == "ok" and r["circuit"] != "c17"]
print("  stems:", sorted({r["cut_stems"] for r in g}))
print(f"  k2_cert_lt_half {min(r['k2_cert_lt_half'] for r in g):.2f}..{max(r['k2_cert_lt_half'] for r in g):.2f}; k1 {min(r['k1_cert_lt_half'] for r in g):.2f}..{max(r['k1_cert_lt_half'] for r in g):.2f}")
print(f"  cut_lt_half {min(r['cut_lt_half'] for r in g):.3f}..{max(r['cut_lt_half'] for r in g):.3f}; k2_tighter_frac {min(r['k2_tighter_frac'] for r in g):.2f}..{max(r['k2_tighter_frac'] for r in g):.2f}")
print("  sound (k1,k2,cut):", all(r["k1_cert_sound"] and r["k2_cert_sound"] and r["cut_sound"] for r in gate if r["status"] == "ok"))
print("  gate verdict criteria: A", sum(r["k2_cert_median"] < r["cut_width_median"] for r in gate if r["status"] == "ok"), "/", len([r for r in gate if r["status"] == "ok"]))
fb = [r for r in fast if r["status"] == "ok" and r.get("critB") is not None]
print("  fast criterion B true on", sum(1 for r in fb if r["critB"]), "/", len(fb))

# ---------------------------------------------------------------- oracle
block("Sec 7.6: propagated/oracle ratio 'A to B times'; tail mass above degree 2 'X to Y'; oracle k2 error 'U to V'; cavlc tail by depth")
for r in e3:
    if r["circuit"] in ("cavlc", "ctrl", "int2float", "sin", "c17", "dec") and r["regime"] == "half":
        print(f"  {r['circuit']:10} oracle k1 {fmt(r['oracle_k1_mean'])} k2 {fmt(r['oracle_k2_mean'])} | prop k1 {fmt(r['prop_norm_k1_mean'])} k2 {fmt(r['prop_norm_k2_mean'])} "
              f"| ratio k1 {fmt(r['prop_norm_k1_mean']/r['oracle_k1_mean'] if r['oracle_k1_mean'] else None,3)} k2 {fmt(r['prop_norm_k2_mean']/r['oracle_k2_mean'] if r['oracle_k2_mean'] else None,3)} | tau2 {fmt(r['tau2_mean'])}")
cav = next(r for r in e3 if r["circuit"] == "cavlc" and r["regime"] == "half")
print("  cavlc by_depth:", json.dumps(cav["by_depth"])[:600])

# ---------------------------------------------------------------- power
block("Sec 7.7: total power relerr within ±A% (k0), ±B% (k1), ±C% (k2) except c6288 nangate; c880 sky130 alpha mean k0/k1/k2; nangate smaller errors")
for k in (0, 1, 2):
    ex = [(abs(r[f"k{k}_power_relerr"]), r["circuit"], r["lib"], r["regime"]) for r in e6 if r["status"] == "ok" and not (r["circuit"] == "c6288" and r["lib"] == "nangate45")]
    worst = max(ex); print(f"  k{k} max |relerr| excluding c6288/nangate: {worst[0]*100:.2f}% ({worst[1]} {worst[2]} {worst[3]})")
    print(f"      c6288 nangate relerr k{k}:", [f"{r[f'k{k}_power_relerr']*100:+.1f}%" for r in e6 if r["circuit"] == "c6288" and r["lib"] == "nangate45"])
r = next(r for r in e6 if r["circuit"] == "c880" and r["lib"] == "sky130" and r["regime"] == "half")
print("  c880 sky130 half alpha mean k0/k1/k2:", fmt(r["k0_alpha_mean"]), fmt(r["k1_alpha_mean"]), fmt(r["k2_alpha_mean"]))
for k in (0, 1, 2):
    for lib in ("sky130", "nangate45"):
        xs = [r[f"k{k}_alpha_mean"] for r in e6 if r["lib"] == lib and r["status"] == "ok" and r["circuit"] not in ("c17",)]
        print(f"  mean alpha error k{k} {lib:9}: {st.mean(xs):.4f}")

# ---------------------------------------------------------------- justification
block("Sec 6 heuristic: true success 0.5 -> [A,B] on N of 17; median gain; frac improved range; no-gain circuits; certificate fire fractions")
for r in sorted(e8, key=lambda r: r["mean_true_opt"]):
    print(f"  {r['circuit']:10} opt {r['mean_true_opt']:.3f} medgain {r['median_gain']:.2f} improved {r['frac_improved']:.2f} worse {r['frac_worse']:.2f} certified {r['frac_certified']:.2f}")
block("Sec 6 rare targets: n_hard total; single-start gains; restart gains; >0.1 counts")
tot = sum(r["n_hard"] for r in e8h); a = {r["circuit"]: r for r in e8h}; b = {r["circuit"]: r for r in e8r}
print("  total hard", tot, "| >0.1 single", sum(round(a[c]["hard_frac_above_0.1"] * a[c]["n_hard"]) for c in a if a[c]["n_hard"]),
      "restart", sum(round(b[c]["hard_frac_above_0.1"] * b[c]["n_hard"]) for c in b if b[c]["n_hard"]))
for c in a:
    if a[c]["n_hard"]:
        print(f"  {c:10} n_hard {a[c]['n_hard']:3d} gain single {a[c]['hard_mean_opt']/a[c]['hard_mean_half']:6.2f} restart {b[c]['hard_mean_opt']/b[c]['hard_mean_half']:6.2f} >0.1 restart {b[c]['hard_frac_above_0.1']:.2f} half {a[c]['hard_mean_half']:.1e} opt_r {b[c]['hard_mean_opt']:.1e}")

# ---------------------------------------------------------------- cost
block("Sec 7.8: k2/k1 GPU ratio range; named times; numpy vs gpu crossover; gate counts")
rs = [(r["circuit"], r["n"], r["and"], r.get("k1_cuda0"), r.get("k2_cuda0"), r.get("ratio_cuda"), r.get("k1_numpy"), r.get("k2_numpy")) for r in e10]
rat = [x[5] for x in rs if x[5] and x[0] != "max"]
print(f"  k2/k1 GPU ratio excluding max: {min(rat):.2f}..{max(rat):.2f}; max: {next(x[5] for x in rs if x[0]=='max'):.1f}")
for x in sorted(rs, key=lambda x: x[1]):
    print(f"  {x[0]:11} n {x[1]:4} AND {x[2]:6} k1 {fmt(x[3])} k2 {fmt(x[4])} | np k1 {fmt(x[6])} k2 {fmt(x[7])} | np/gpu k2 {fmt(x[7]/x[4] if x[7] and x[4] else None, 2)}")

block("Sec 5.4 / 7.6: per-gate bound tightness (exp/bound_tightness.py -> bound_tightness.json)")
try:
    bt = json.load(open(R / "bound_tightness.json"))
    print("  pairs", bt["pairs"], "max ratio", {k: round(v, 3) for k, v in bt["max_ratio"].items()}, "mean", {k: round(v, 2) for k, v in bt["mean_ratio"].items()})
except FileNotFoundError:
    print("  bound_tightness.json missing: run python exp/bound_tightness.py results_run2/bound_tightness.json")

# ================================================================ extension experiments (results_ext/), skipped if absent
EXT = Path(sys.argv[2]) if len(sys.argv) > 2 else R.parent / "results_ext"


def _ext(p):
    f = EXT / p
    if not f.exists():
        print(f"  (missing {f})"); return None
    d = json.load(open(f))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d

block("ablation: mean error per circuit and order (p=1/2, projected); feasibility; k3 gain over k2; raw k3 divergence")
rows = _ext("ablation/e15_rows.json")
if rows:
    for c in sorted({r["circuit"] for r in rows}):
        rs = {r["k"]: r for r in rows if r["circuit"] == c and r["regime"] == "half" and not r["raw"] and r.get("status") == "ok"}
        cells = " ".join(f"k{k}:{(fmt(rs[k]['mean_err'], 3) if rs[k].get('feasible') else '--'):>8}" for k in sorted(rs))
        raw = next((r for r in rows if r["circuit"] == c and r["regime"] == "half" and r["raw"] and r.get("feasible")), None)
        g = (rs[2]["mean_err"] / rs[3]["mean_err"]) if 2 in rs and 3 in rs and rs[3].get("feasible") and rs[3]["mean_err"] else None
        print(f"  {c:11} {cells}  k2/k3 {fmt(g, 2) if g else '--':>5}  k3raw finite {raw.get('finite') if raw else '--'}")
    feas = {k: sum(1 for r in rows if r["k"] == k and r["regime"] == "half" and not r["raw"] and r.get("feasible")) for k in range(6)}
    print("  feasible circuits per order (p=1/2):", feas)

block("pareto: break-even N*(k) and time ratios; BDD times")
rows = _ext("pareto/e11_rows.json")
if rows:
    for r in rows:
        if r["regime"] != "half": continue
        print(f"  {r['circuit']:11} {r['class']:10} bdd {fmt(r.get('bdd_runtime'), 3):>7}  k2 err {fmt(r.get('k2_err'), 3):>8} t {fmt(r.get('k2_time'), 3):>7}  N*(2) {fmt(r.get('k2_Nstar'), 3):>8} t_MC {fmt(r.get('k2_tMCstar'), 3):>7} ratio {fmt(r.get('k2_ratio'), 3):>6} | MC err 1e4 {fmt(r.get('mc_err_N10000'), 3)} 1e6 {fmt(r.get('mc_err_N1000000'), 3)}")
    ctrl = [r["k2_Nstar"] for r in rows if r["regime"] == "half" and r["class"] == "control" and r.get("k2_Nstar")]
    ar = [r["k2_Nstar"] for r in rows if r["regime"] == "half" and r["class"] == "arithmetic" and r.get("k2_Nstar")]
    if ctrl: print(f"  N*(2) control: min {min(ctrl):.3g} median {st.median(ctrl):.3g} max {max(ctrl):.3g}")
    if ar: print(f"  N*(2) arithmetic: min {min(ar):.3g} median {st.median(ar):.3g} max {max(ar):.3g}")

block("cv: control variates (N=1e6, p=1/2): VRF1 median per circuit, predicted, VRF2 outputs; median over control circuits")
rows = _ext("cv/e12b_rows.json")
if rows:
    meds = []
    for key in sorted({(r["family"], r["circuit"]) for r in rows}):
        rs = [r for r in rows if (r["family"], r["circuit"]) == key and r["regime"] == "half" and r["N"] == 10 ** 6]
        if not rs: continue
        v1 = st.mean([r["vrf1_median"] for r in rs]); p1 = st.mean([r["pred1_median"] for r in rs]); v2 = st.mean([r["vrf2_median"] for r in rs if r.get("vrf2_median")]) if any(r.get("vrf2_median") for r in rs) else None
        ov = st.mean([r["t_cv"] / max(r["t_sim"], 1e-9) for r in rs])
        print(f"  {key[1]:11} {rs[0]['class']:10} VRF1 {v1:6.2f} pred {p1:6.2f} VRF2(out) {fmt(v2, 3):>6} overhead {ov:.2f} plain {fmt(st.mean([r['plain_err'] for r in rs]), 3)} cv1 {fmt(st.mean([r['cv1_err'] for r in rs]), 3)}")
        if rs[0]["class"] == "control": meds.append(v2 if v2 else v1)
    if meds: print(f"  median VRF (order 2 outputs where available) over control circuits = {st.median(meds):.2f} (threshold 3)")
wrows = _ext("cv/e12a_rows.json")
if wrows:
    for r in wrows:
        if r["regime"] == "half": print(f"  widths {r['circuit']:6} cert2<MC: N=1e3 {r['cert2_narrower_N1000']:.2f} 1e4 {r['cert2_narrower_N10000']:.2f} 1e5 {r['cert2_narrower_N100000']:.2f}; cert1<MC 1e4 {r['cert1_narrower_N10000']:.2f}; cut<MC 1e4 {r['cut_narrower_N10000']:.2f}")

block("openroad: total/switching power relative errors per design/platform; tool default vs order 0")
rows = _ext("openroad/e13_rows.json")
if rows:
    for r in rows:
        f = lambda k: fmt(100 * r[k], 3) + "%" if r.get(k) is not None else "--"
        print(f"  {r['platform']:9} {r['design']:10} insts {r.get('n_instances')} truth {r.get('truth')} total: k0 {f('k0_total_relerr'):>8} k1 {f('k1_total_relerr'):>8} k2 {f('k2_total_relerr'):>8} k3 {f('k3_total_relerr'):>8} default {f('default_total_relerr'):>8} | pin-cap k0 {f('k0_pincap_relerr'):>8} k2 {f('k2_pincap_relerr'):>8}")
    for m in ("k0", "k1", "k2", "k3", "default"):
        xs = [abs(r[f"{m}_total_relerr"]) for r in rows if r.get(f"{m}_total_relerr") is not None]
        if xs: print(f"  max |total relerr| {m}: {100*max(xs):.2f}% (n={len(xs)})")

block("errmodel: M1 per-net correlation/slope; M2 fitted c_k and per-circuit ratios")
d = EXT / "errmodel" / "e14_rows.json"
if d.exists():
    dd = json.load(open(d)); print("  fits c_k:", dd["fits"])
    for r in dd["rows"]:
        if r["model"] == "m1" and r.get("status") == "ok": print(f"  M1 {r['circuit']:10} {r['regime']:5} k{r['k']} corr {r['corr']:.3f} slope {r['slope']:.3f} R2 {r['r2']:.3f}")
    for k in (1, 2, 3):
        rs = [r for r in dd["rows"] if r["model"] == "m2" and r["k"] == k and r.get("status") == "ok" and r.get("ratio")]
        if rs:
            c = [r["ratio"] for r in rs if r["class"] == "control"]; a = [r["ratio"] for r in rs if r["class"] == "arithmetic"]
            print(f"  M2 k{k}: pred/meas ratio control median {st.median(c):.2f} range {min(c):.2f}-{max(c):.2f}; arithmetic median {st.median(a) if a else float('nan'):.2f}")
else:
    print("  (missing errmodel rows)")

# ---------------------------------------------------------------- every remaining quoted statistic
block("Sec 7.2: k2 max range excl c6288; projection effect on c432 at order 2; c6288 clamp divergence counts")
ok = [r for r in proj if r["circuit"] != "c6288" and r["status"] == "ok"]
print(f"  k2 norm max excl c6288/c17: {min(r['k2_norm_max'] for r in ok if r['circuit']!='c17'):.3f}..{max(r['k2_norm_max'] for r in ok):.3f}; k1 clamp max: {min(r['k1_clamp_max'] for r in ok if r['circuit']!='c17'):.3f}..{max(r['k1_clamp_max'] for r in ok):.3f}")
for r in proj:
    if r["status"] != "ok": continue
    d1 = abs(r["k1_norm_mean"] - r["k1_clamp_mean"]) / r["k1_clamp_mean"] if r["k1_clamp_mean"] else 0
    d2 = (r["k2_norm_mean"] - r["k2_clamp_mean"]) / r["k2_clamp_mean"] if r["k2_clamp_mean"] and r["k2_clamp_nonfinite"] == 0 else float("nan")
    if d1 > 1e-3 or abs(d2) > 0.02 or r["circuit"] == "c6288":
        print(f"  {r['circuit']:6} {r['regime']:5} k1 |norm-clamp|/clamp {100*d1:.2f}%  k2 (norm-clamp)/clamp {100*d2:+.1f}%  nonfinite k1 clamp {r['k1_clamp_nonfinite']} k2 clamp {r['k2_clamp_nonfinite']}")

block("Sec 7.3 / fig families: c499 gate vs AIG forms at k1,k2; optimised (resyn2/dc2) worse than raw AIG")
def p2get(c, fam, reg="half"): return next((r for r in p2 if r["circuit"] == c and r["family"] == fam and r["regime"] == reg and r["status"] == "ok"), None)
g = p2get("c499", "gate")
for f in ("aig", "resyn2", "dc2"):
    r = p2get("c499", f); print(f"  c499 {f:6} k1 {r['k1_mean']:.5f} ({r['k1_mean']/g['k1_mean']:.0f}x gate) k2 {r['k2_mean']:.5f} ({r['k2_mean']/g['k2_mean']:.0f}x gate)")
viol = [(c, reg, f, k, round(p2get(c, f, reg)[f"{k}_mean"], 5), round(p2get(c, "aig", reg)[f"{k}_mean"], 5)) for c in sorted({r["circuit"] for r in p2 if r["family"] == "aig"}) for reg in ("half", "rand1") for f in ("resyn2", "dc2") for k in ("k0", "k1", "k2") if p2get(c, f, reg) and p2get(c, "aig", reg) and p2get(c, f, reg)[f"{k}_mean"] > p2get(c, "aig", reg)[f"{k}_mean"] * 1.0001]
print("  optimised worse than raw AIG (circuit, regime, form, order, opt, raw):", viol)
print("  phase2 truth kinds by family:", sorted({(r["family"], r.get("truth")) for r in p2 if r["status"] == "ok"}))

block("Sec 7.4: smallest non-zero order-2 certified width per circuit (cv/widths); c17 fractions")
for f in sorted((EXT / "cv" / "widths").glob("widths__*__half.json")):
    d = json.load(open(f)); w2 = d.get("w_cert2") or d.get("cert2") or []
    nz = sorted(x for x in w2 if x > 1e-12)
    if w2: print(f"  {d['task']['circuit'] if 'task' in d else f.name:8} n {len(w2)} zero-width {1-len(nz)/len(w2):.2f} smallest non-zero {nz[0] if nz else float('nan'):.3f}")

block("Sec 7.5: oracle k2 error range and tau2 by depth on cavlc")
cv_ = next(r for r in e3 if r["circuit"] == "cavlc" and r["regime"] == "half")
print("  oracle k2 mean on cavlc/ctrl/int2float:", sorted(round(r["oracle_k2_mean"], 5) for r in e3 if r["circuit"] in ("cavlc", "ctrl", "int2float") and r["regime"] == "half"))
print("  cavlc tau2 by depth bin:", [(b["depth_min"], b["depth_max"], round(b["tau2"], 4)) for b in cv_["by_depth"]])

block("errmodel (text): M1 corr/slope range over cavlc/ctrl/int2float/sin; M2 with measured >= 1e-4 excluded: within 3x / 10x, medians, c6288, corr_log")
d = EXT / "errmodel" / "e14_rows.json"
if d.exists():
    dd = json.load(open(d)); m1 = [r for r in dd["rows"] if r["model"] == "m1" and r.get("status") == "ok" and r["circuit"] in ("cavlc", "ctrl", "int2float", "sin")]
    print(f"  M1 corr {min(r['corr'] for r in m1):.3f}..{max(r['corr'] for r in m1):.3f} slope {min(r['slope'] for r in m1):.3f}..{max(r['slope'] for r in m1):.3f}")
    import math as _m
    for k in (1, 2, 3):
        good = [r for r in dd["rows"] if r["model"] == "m2" and r["k"] == k and r.get("status") == "ok" and r.get("mean_meas") is not None and r["mean_meas"] >= 1e-4]
        for cls in ("control", "arithmetic"):
            rat = [r["ratio"] for r in good if r["class"] == cls]
            if rat: print(f"  M2 k{k} {cls:10}: n {len(rat)} median {st.median(rat):.2f} within3x {sum(1/3<=x<=3 for x in rat)}/{len(rat)} ({100*sum(1/3<=x<=3 for x in rat)/len(rat):.0f}%) within10x {sum(0.1<=x<=10 for x in rat)}/{len(rat)}")
        cl = [r["corr_log"] for r in good if r.get("corr_log") is not None and not _m.isnan(r["corr_log"])]
        print(f"  M2 k{k} corr_log median {st.median(cl):.2f}; c6288 ratio {[round(r['ratio'], 3) for r in good if r['circuit']=='c6288']}")

block("ablation (text): generic-engine runtimes; k3/k2; k4 times; generic k2 vs dense k2 (timing job)")
rows = _ext("ablation/e15_rows.json"); e10 = {r["circuit"]: r for r in json.load(open(R / "timing/e10_rows.json"))}
def _t(c, k): r = next((r for r in rows if r["circuit"] == c and r["k"] == k and r["regime"] == "half" and not r["raw"] and r.get("feasible")), None); return r["runtime"] if r else None
for c in sorted({r["circuit"] for r in rows}):
    t2, t3, t4 = _t(c, 2), _t(c, 3), _t(c, 4)
    if t3: print(f"  {c:11} t2 {t2:7.2f} t3 {t3:7.2f} (x{t3/t2:4.1f}) t4 {fmt(t4, 3):>7}  generic/dense k2 {t2/e10[c]['k2_cuda0']:.1f}" if c in e10 and e10[c].get("k2_cuda0") else f"  {c:11} t2 {t2:7.2f} t3 {t3:7.2f} (x{t3/t2:4.1f}) t4 {fmt(t4, 3):>7}")

block("pareto (text): N*(2) by bucket; ratio medians; k3 ratio range; MC 1e8 times/errors on bar/arbiter/dec; MC error at 1e4/1e6 excl dec")
rows = _ext("pareto/e11_rows.json"); h = [r for r in rows if r["regime"] == "half"]
ns = sorted((r["k2_Nstar"], r["circuit"]) for r in h if r.get("k2_Nstar"))
print("  N*=1e3 (floor):", [c for n, c in ns if n <= 1000], "| (1e3,1.1e4]:", [(c, round(n)) for n, c in ns if 1000 < n <= 1.1e4], "| above:", [(c, f"{n:.3g}") for n, c in ns if n > 1.1e4])
rat = [r["k2_ratio"] for r in h if r.get("k2_ratio")]; print(f"  k2 ratio median {st.median(rat):.3f}; control {min(r['k2_ratio'] for r in h if r.get('k2_ratio') and r['class']=='control'):.3f}..{max(r['k2_ratio'] for r in h if r.get('k2_ratio') and r['class']=='control'):.3f}; arithmetic {min(r['k2_ratio'] for r in h if r.get('k2_ratio') and r['class']=='arithmetic'):.3f}..{max(r['k2_ratio'] for r in h if r.get('k2_ratio') and r['class']=='arithmetic'):.3f}")
k3 = [r["k3_ratio"] for r in h if r.get("k3_ratio")]; print(f"  k3 ratio {min(k3):.4f}..{max(k3):.4f} (n={len(k3)})")
for r in h:
    if r["circuit"] in ("bar", "arbiter", "dec", "c17"): print(f"  {r['circuit']:8} k2 err {fmt(r['k2_err'], 3)} t2 {fmt(r['k2_time'], 3)} | MC 1e8 err {fmt(r['mc_err_N100000000'], 3)} t {fmt(r['mc_time_N100000000'], 3)}")
print(f"  MC err 1e4 excl dec {min(r['mc_err_N10000'] for r in h if r['circuit']!='dec'):.4f}..{max(r['mc_err_N10000'] for r in h):.4f}; 1e6 {min(r['mc_err_N1000000'] for r in h if r['circuit']!='dec'):.5f}..{max(r['mc_err_N1000000'] for r in h):.5f}; dec {next(r['mc_err_N10000'] for r in h if r['circuit']=='dec'):.5f}")
print(f"  BDD times ISCAS excl c6288: {min(r['bdd_runtime'] for r in h if r.get('bdd_runtime')):.3f}..{max(r['bdd_runtime'] for r in h if r.get('bdd_runtime')):.3f}")

block("cv (text): VRF1 range/median over control-type circuits, c17; prediction error max; VRF2<1 circuits; overhead where VRF1>=1.5 and on multiplicative blocks")
rows = _ext("cv/e12b_rows.json"); per = {}
for r in rows:
    if r["regime"] == "half" and r["N"] == 10 ** 6: per.setdefault((r["circuit"], r["class"]), []).append(r)
agg = {k: (st.mean([r["vrf1_median"] for r in v]), st.mean([r["pred1_median"] for r in v]), st.mean([r["vrf2_median"] for r in v if r.get("vrf2_median")]) if any(r.get("vrf2_median") for r in v) else None, st.mean([r["t_cv"] / max(r["t_sim"], 1e-9) for r in v])) for k, v in per.items()}
ctrl = {k: a for k, a in agg.items() if k[1] == "control"}
print(f"  VRF1 control-type: {min(a[0] for k, a in ctrl.items() if k[0]!='c17'):.2f}..{max(a[0] for k, a in ctrl.items() if k[0]!='c17'):.2f} (c17 {ctrl[('c17','control')][0]:.2f}) median {st.median(a[0] for a in ctrl.values()):.2f}")
print(f"  max |pred/meas - 1|: {max(abs(a[1]/a[0]-1) for a in agg.values()):.3f}")
print("  VRF2(out) < 1:", [(k[0], round(a[2], 2)) for k, a in agg.items() if a[2] is not None and a[2] < 1])
print("  overhead where VRF1 >= 1.5:", sorted((round(a[3], 2), k[0]) for k, a in agg.items() if a[0] >= 1.5))
print("  overhead on multiplicative blocks:", sorted((round(a[3], 3), k[0]) for k, a in agg.items() if k[1] == "arithmetic"))

block("openroad (text): maxima/medians excl c6288 per order; c6288 k3; pin-cap vs flow differences; OpenSTA default vs k0; order-2 closest counts")
rows = _ext("openroad/e13_rows.json"); nm = [r for r in rows if r["design"] != "c6288"]
for k in ("k0", "k1", "k2", "k3"):
    xs = [(abs(r[f"{k}_total_relerr"]) * 100, r["platform"], r["design"]) for r in nm if r.get(f"{k}_total_relerr") is not None]
    med = {pl: st.median([abs(r[f"{k}_total_relerr"]) * 100 for r in nm if r["platform"] == pl and r.get(f"{k}_total_relerr") is not None]) for pl in ("nangate45", "sky130hd")}
    print(f"  {k}: max excl c6288 {max(xs)[0]:.2f}% ({max(xs)[1]} {max(xs)[2]}); medians nangate45 {med['nangate45']:.2f}% sky130hd {med['sky130hd']:.2f}%")
print("  c6288:", [(r["platform"], [round(100 * r[f"{k}_total_relerr"], 1) for k in ("k0", "k1", "k2", "k3") if r.get(f"{k}_total_relerr") is not None]) for r in rows if r["design"] == "c6288"])
pc = sorted(((abs(r[f"{k}_pincap_relerr"] - r[f"{k}_total_relerr"]) * 100, r["platform"], r["design"], k) for r in rows for k in ("k0", "k2")), reverse=True)
print("  pin-cap vs flow, largest differences (points):", [(round(d, 2), p, c, k) for d, p, c, k in pc[:6]])
dk = sorted(((abs(r["default_total_relerr"] - r["k0_total_relerr"]) * 100, r["platform"], r["design"]) for r in rows), reverse=True)
print("  OpenSTA default vs k0, nangate45 largest:", [(round(d, 2), c) for d, p, c in dk if p == "nangate45"][:3], "sky130 max:", round(max(d for d, p, c in dk if p == "sky130hd"), 2))
best = lambda r: min({k: abs(r[f"{k}_total_relerr"]) for k in ("k0", "k1", "k2", "default")}.items(), key=lambda kv: kv[1])[0]
print(f"  k2 <= k0 on {sum(abs(r['k2_total_relerr']) <= abs(r['k0_total_relerr']) + 1e-12 for r in rows)}/38; k2 <= k1 on {sum(abs(r['k2_total_relerr']) <= abs(r['k1_total_relerr']) + 1e-12 for r in rows)}/38; default closer than k2 on {sum(abs(r['default_total_relerr']) < abs(r['k2_total_relerr']) for r in rows)} (sky130: {sum(abs(r['default_total_relerr']) < abs(r['k2_total_relerr']) and r['platform']=='sky130hd' for r in rows)}), max |k2| among those {100*max(abs(r['k2_total_relerr']) for r in rows if abs(r['default_total_relerr']) < abs(r['k2_total_relerr'])):.2f}%")
print("  sky130 c499/c1355 k0 vs default:", [(r["design"], round(100 * r["k0_total_relerr"], 2), round(100 * r["default_total_relerr"], 4)) for r in rows if r["platform"] == "sky130hd" and r["design"] in ("c499", "c1355")])

block("Appendix B (text): circuits whose k1/k2 ratio interval (mean ± se) excludes 2 from above; range of the rest")
rat = {}
for r in p2b:
    if r.get("status") == "ok" and r.get("k2_mean") and r["regime"] in ("rand2", "rand3"): rat.setdefault(r["circuit"], []).append(r["k1_mean"] / r["k2_mean"])
import math as _m2
summ = {c: (st.mean(v), st.stdev(v) / _m2.sqrt(len(v))) for c, v in rat.items()}
print("  above 2:", sorted(c for c, (m, se) in summ.items() if m - se > 2), "| others:", sorted((c, round(m, 2)) for c, (m, se) in summ.items() if m - se <= 2), "| max se", round(max(se for m, se in summ.values()), 3))
