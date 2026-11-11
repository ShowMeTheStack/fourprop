#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Validation of the power claims in the OpenROAD flow (ORFS in a podman container) + OpenSTA power reports.

Kinds (all resumable, one JSON per task):
  orfs  (design, platform): run the ORFS flow in the container; artefacts under ORFS_ROOT/work/<platform>/<design>/results/...
  prop  (design, platform): parse 6_final.v with the platform Liberty, propagate orders 0..3 (3 if feasible), compute the truth
        (BDD within 1 h, else Monte Carlo 10^8), per-net P and alpha for each method -> annotations JSON
  sta   (design, platform, method): annotate OpenSTA per net (set_power_activity -pins driver), report_power (totals, per instance)
        methods: exact, k0, k1, k2, k3, default (input ports only; OpenSTA propagates)
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from runner_lib import run_tasks  # noqa: E402
from orfs_prep import ISCAS, EPFL_CTRL, PLATFORMS, TOP  # noqa: E402

IMAGE = "docker.io/openroad/orfs:latest"
LIB = {"nangate45": "/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib",
       "sky130hd": "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"}
LOCAL_LIB = {"nangate45": "NangateOpenCellLibrary_typical.lib", "sky130hd": "sky130_fd_sc_hd__tt_025C_1v80.lib"}
METHODS = ["exact", "k0", "k1", "k2", "k3", "default"]
TIMEOUTS = {"orfs": 3 * 3600, "prop": 2 * 3600, "sta": 3 * 3600}
BDD_SKIP = {"c6288"}          # BDDs of the multiplier never finish: straight to Monte Carlo
BDD_LIMIT = 900
KS = (0, 1, 2, 3)


def designs():
    return [("iscas", c) for c in ISCAS] + [("epfl", c) for c in EPFL_CTRL]


def task_list(bench: Path):
    tasks = []
    for fam, d in designs():
        for plat in PLATFORMS:
            tasks.append({"id": f"orfs__{plat}__{d}", "kind": "orfs", "family": fam, "design": d, "platform": plat})
            tasks.append({"id": f"prop__{plat}__{d}", "kind": "prop", "family": fam, "design": d, "platform": plat})
            for m in METHODS:
                tasks.append({"id": f"sta__{plat}__{d}__{m}", "kind": "sta", "family": fam, "design": d, "platform": plat, "method": m})
    return tasks


def podman(orfs_root: Path, cmd: str, timeout: int, log: Path):
    full = ["podman", "run", "--rm", "-v", f"{orfs_root}:/work", IMAGE, "bash", "-c", "source /OpenROAD-flow-scripts/env.sh; " + cmd]
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(full)}\n"); f.flush()
        r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def results_dir(orfs_root: Path, plat: str, d: str) -> Path:
    return orfs_root / "work" / plat / d / "results" / plat / TOP.get(d, d) / "base"   # ORFS names the results dir after DESIGN_NAME


def _wait_for(path: Path, timeout: float, what: str):
    """Tasks of a later kind can be dispatched while an earlier kind for the same design is still running: wait for its artefact."""
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{what} not available after {timeout:.0f}s: {path}")
        time.sleep(20)


def run_task(task: dict, extra: dict) -> dict:
    orfs = Path(extra["orfs"]); plat = task["platform"]; d = task["design"]
    rd = results_dir(orfs, plat, d); res = {"load_start": os.getloadavg()[0]}
    if task["kind"] == "orfs":
        if (rd / "6_final.v").exists() and (rd / "6_final.spef").exists():
            res["reused"] = True
        else:
            t0 = time.time()
            rc = podman(orfs, f"cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=/work/designs/{plat}/{d}/config.mk WORK_HOME=/work/work/{plat}/{d}", TIMEOUTS["orfs"] - 60, orfs / "logs" / f"orfs_{plat}_{d}.log")
            res["make_rc"] = rc; res["flow_time"] = time.time() - t0
            if rc != 0 or not (rd / "6_final.v").exists():
                raise RuntimeError(f"ORFS flow failed (rc {rc}); see logs/orfs_{plat}_{d}.log")
        # cell count and area from the final netlist
        txt = open(rd / "6_final.v").read()
        res["n_instances"] = len(re.findall(r"^\s*[A-Za-z0-9_]+\s+_\d+_\s*\(", txt, flags=re.M))
        return res
    if task["kind"] == "prop":
        from fourprop.liberty import read_liberty, load_mapped_verilog
        from fourprop.dense import propagate_dense
        from fourprop.dense_k import propagate_dense_k, state_bytes, max_live_nets
        _wait_for(rd / "6_final.spef", TIMEOUTS["orfs"], "ORFS result")
        cells = read_liberty(str(Path(extra["bench"]) / "liberty" / LOCAL_LIB[plat]))
        circ, nets = load_mapped_verilog(str(rd / "6_final.v"), cells, name=d)
        n = circ.n_inputs; N = len(circ.nodes); p = np.full(n, 0.5)
        res.update({"n_inputs": n, "n_nodes": N, "n_nets": len(nets), "depth": circ.depth()})
        # truth
        truth = None; tkind = None; t0 = time.time()
        if d not in BDD_SKIP:
            try:
                from fourprop.bdd import bdd_probabilities
                truth = np.asarray(bdd_probabilities(circ, p, time_limit=BDD_LIMIT)); tkind = "bdd"
            except Exception as e:   # timeout / memory
                res["bdd_error"] = str(e)[:200]
        if truth is None:
            from fourprop.mc_torch import monte_carlo_torch
            truth, _ = monte_carlo_torch(circ, p, n_samples=10 ** 8, seed=0, device="cuda:1"); tkind = "mc"
        res["truth"] = tkind; res["truth_time"] = time.time() - t0
        est = {}
        for k in KS:
            if k <= 2:
                P, rt, _ = propagate_dense(circ, p, k=k, backend="cuda:1", project="norm")
            else:
                if state_bytes(n, k) * max_live_nets(circ) > 8 * 1024 ** 3:
                    res["k3_infeasible"] = True; continue
                P, rt, _, _ = propagate_dense_k(circ, p, k=k, backend="cuda:1", project="norm")
            est[f"k{k}"] = (np.clip(P, 0, 1), rt)
        est["exact"] = (truth, res["truth_time"])
        # per-net annotation records: net name, driver (instance, pin) or input port, P and alpha per method, load cap
        ann = []
        for mn in nets:
            rec = {"net": mn.name, "node": mn.node, "driver": mn.driver, "is_input": mn.is_input, "load_cap": mn.load_cap}
            for m, (P, _) in est.items():
                Pv = float(P[mn.node]); rec[m] = {"P": Pv, "alpha": 2 * Pv * (1 - Pv)}
            ann.append(rec)
        json.dump({"design": d, "platform": plat, "nets": ann, "times": {m: t for m, (_, t) in est.items()}}, open(rd / "fourprop_annotations.json", "w"))
        # error statistics vs truth on the real nets
        idx = np.array([mn.node for mn in nets]); tr = truth[idx]
        for m, (P, t) in est.items():
            e = np.abs(P[idx] - tr); a_tr = 2 * tr * (1 - tr); a_e = np.abs(2 * P[idx] * (1 - P[idx]) - a_tr)
            res[m] = {"mean_err": float(e.mean()), "max_err": float(e.max()), "alpha_mean_err": float(a_e.mean()), "time": t}
            caps = np.array([mn.load_cap for mn in nets]); res[m]["pincap_power_relerr"] = float((np.sum(caps * 2 * P[idx] * (1 - P[idx])) - np.sum(caps * a_tr)) / max(np.sum(caps * a_tr), 1e-18))
        return res
    # sta
    m = task["method"]
    prop_json = Path(extra["root"]) / "results" / f"prop__{plat}__{d}.json"
    t0w = time.time()
    while not (rd / "fourprop_annotations.json").exists():
        if prop_json.exists() and not json.load(open(prop_json)).get("ok"):
            raise RuntimeError("propagation task failed; no annotations")
        if time.time() - t0w > TIMEOUTS["sta"] - 600:
            raise RuntimeError("annotations not available in time")
        time.sleep(20)
    ann = json.load(open(rd / "fourprop_annotations.json"))
    top = TOP.get(d, d)
    tcl = [f"read_liberty {LIB[plat]}", f"read_db /work/work/{plat}/{d}/results/{plat}/{top}/base/6_final.odb",
           f"read_sdc /work/work/{plat}/{d}/results/{plat}/{top}/base/6_final.sdc", f"read_spef /work/work/{plat}/{d}/results/{plat}/{top}/base/6_final.spef"]
    src = "exact" if m == "default" else m
    if not any(src in rec for rec in ann["nets"]):
        res["skipped"] = f"no {src} annotations (order infeasible on this design)"; return res   # never report an unannotated run as a method
    for rec in ann["nets"]:
        if src not in rec:
            continue
        a = rec[src]["alpha"]; duty = rec[src]["P"]
        if rec["is_input"]:
            tcl.append(f"set_power_activity -input_ports [get_ports {{{rec['net']}}}] -activity {a:.6f} -duty {duty:.6f}")
        elif m != "default" and rec["driver"]:
            inst, pin = rec["driver"]
            tcl.append(f"set_power_activity -pins [get_pins {{{inst}/{pin}}}] -activity {a:.6f} -duty {duty:.6f}")
    tcl.append("report_power -digits 8 -format json > /work/work/%s/%s/power_%s.json" % (plat, d, m))
    tcl.append("report_power -instances [get_cells *] -digits 8 -format json > /work/work/%s/%s/power_%s_inst.json" % (plat, d, m))
    tcl.append("report_power -digits 8 > /work/work/%s/%s/power_%s.txt" % (plat, d, m))
    tcl.append("exit")
    (orfs / "work" / plat / d / f"sta_{m}.tcl").write_text("\n".join(tcl) + "\n")
    t0 = time.time()
    rc = podman(orfs, f"openroad -exit /work/work/{plat}/{d}/sta_{m}.tcl", TIMEOUTS["sta"] - 60, orfs / "logs" / f"sta_{plat}_{d}.log")
    res["rc"] = rc; res["sta_time"] = time.time() - t0
    txt = (orfs / "work" / plat / d / f"power_{m}.txt").read_text() if (orfs / "work" / plat / d / f"power_{m}.txt").exists() else ""
    res["report"] = txt[-3000:]
    tot = re.search(r"Total\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)", txt)
    if not tot:
        raise RuntimeError("no Total line in report_power output")
    res["internal"], res["switching"], res["leakage"], res["total"] = [float(x) for x in tot.groups()]
    try:
        inst = json.load(open(orfs / "work" / plat / d / f"power_{m}_inst.json"))
        res["n_inst_reported"] = len(inst) if isinstance(inst, list) else len(inst.get("instances", inst))
    except Exception as e:
        res["inst_json_error"] = str(e)[:100]
    return res


def write_verdict(root: Path, orfs: Path) -> str:
    rows = {}; prop = {}
    for f in sorted((root / "results").glob("*.json")):
        d = json.load(open(f)); t = d["task"]; key = (t["platform"], t["design"])
        if not d.get("ok"):
            rows.setdefault(key, {})[t["kind"] + ("_" + t.get("method", "") if t["kind"] == "sta" else "")] = {"error": d.get("error", "")[:80]}; continue
        if t["kind"] == "prop":
            prop[key] = d
        elif t["kind"] == "sta":
            if d.get("skipped"):
                continue
            rows.setdefault(key, {})[t["method"]] = d
        else:
            rows.setdefault(key, {})["orfs"] = d
    out = []
    L = ["# OpenROAD / OpenSTA validation (activity annotated per net; relative error of reported power vs exact annotation)\n",
         "| platform | design | insts | truth | " + " | ".join(f"{m} tot%" for m in ("k0", "k1", "k2", "k3", "default")) + " | " + " | ".join(f"{m} sw%" for m in ("k0", "k1", "k2", "k3", "default")) + " | pin-cap k0% | pin-cap k2% | alpha err k1 | alpha err k2 |", "|---|" * 17]
    for key in sorted(rows):
        plat, d = key; r = rows[key]; ex = r.get("exact"); pr = prop.get(key)
        row = {"platform": plat, "design": d, "n_instances": r.get("orfs", {}).get("n_instances"), "truth": pr["truth"] if pr else None}
        cells = []
        for grp in ("total", "switching"):
            for m in ("k0", "k1", "k2", "k3", "default"):
                v = r.get(m)
                if ex and v and "total" in v and ex.get(grp, 0) > 0:
                    rel = (v[grp] - ex[grp]) / ex[grp]; row[f"{m}_{grp}_relerr"] = rel; cells.append(f"{100*rel:+.2f}")
                else:
                    cells.append("--")
        pc = []
        for m in ("k0", "k2"):
            v = pr.get(m) if pr else None; pc.append(f"{100*v['pincap_power_relerr']:+.2f}" if v else "--")
            if v: row[f"{m}_pincap_relerr"] = v["pincap_power_relerr"]
        ae = []
        for m in ("k1", "k2"):
            v = pr.get(m) if pr else None; ae.append(f"{v['alpha_mean_err']:.4f}" if v else "--")
            if v: row[f"{m}_alpha_err"] = v["alpha_mean_err"]
        if ex:
            row.update({"exact_total": ex["total"], "exact_switching": ex["switching"], "exact_internal": ex["internal"], "exact_leakage": ex["leakage"]})
        out.append(row)
        L.append(f"| {plat} | {d} | {row['n_instances']} | {row['truth']} | " + " | ".join(cells) + " | " + " | ".join(pc) + " | " + " | ".join(ae) + " |")
    txt = "\n".join(L) + "\n"; (root / "E13_RESULTS.md").write_text(txt); json.dump(out, open(root / "e13_rows.json", "w"), indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--orfs", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); root = Path(a.root); (Path(a.orfs) / "logs").mkdir(parents=True, exist_ok=True)
    return run_tasks(root, task_list(Path(a.bench)), "openroad_runner", "run_task", {"bench": a.bench, "orfs": a.orfs, "root": str(root)}, TIMEOUTS, workers=a.workers,
                     serial_kinds=("prop", "sta"), kind_order={"orfs": 0, "prop": 1, "sta": 2}, on_done=lambda r: print(write_verdict(root, Path(a.orfs))))


if __name__ == "__main__":
    sys.exit(main())
