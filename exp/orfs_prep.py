#!/usr/bin/env python3
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Preparation: design directories for the OpenROAD flow scripts (run inside the orfs container with /work = ORFS_ROOT).

Layout under ORFS_ROOT (default /extra/fourprop/ext/orfs):
  designs/src/<design>/<design>.v            Verilog (ISCAS-85: original gate-level source; EPFL: generated from the AIG)
  designs/<platform>/<design>/config.mk      ORFS design config (paths as seen inside the container: /work/...)
  designs/<platform>/<design>/constraint.sdc virtual clock only (combinational designs)
Usage: orfs_prep.py --bench /extra/fourprop/bench --orfs /extra/fourprop/ext/orfs
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

ISCAS = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
EPFL_CTRL = ["ctrl", "cavlc", "int2float", "router", "i2c", "priority", "dec", "arbiter"]
PLATFORMS = ["nangate45", "sky130hd"]
TOP = {"priority": "epfl_priority"}
CLK_PERIOD = {"nangate45": 2.0, "sky130hd": 10.0}
CELL_AREA = {"nangate45": 1.6, "sky130hd": 7.0}      # µm² per gate, rough mix
MIN_CORE = {"nangate45": 120.0, "sky130hd": 200.0}   # µm; the PDN strap pitch needs this much
MARGIN = {"nangate45": 10.0, "sky130hd": 15.0}
GATES = {"c17": 6, "c432": 160, "c499": 202, "c880": 383, "c1355": 546, "c1908": 880, "c2670": 1193, "c3540": 1669, "c5315": 2307, "c6288": 2406, "c7552": 3512,
         "ctrl": 174, "cavlc": 693, "int2float": 260, "router": 257, "i2c": 1342, "priority": 978, "dec": 304, "arbiter": 11839}


def designs():
    return [("iscas", c) for c in ISCAS] + [("epfl", c) for c in EPFL_CTRL]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); ap.add_argument("--orfs", required=True); ap.add_argument("--util", type=int, default=40)
    a = ap.parse_args(); bench = Path(a.bench); orfs = Path(a.orfs)
    from fourprop.circuit import load_bench
    from fourprop.verilog_out import write_verilog
    meta = {}
    for fam, d in designs():
        src = orfs / "designs" / "src" / d; src.mkdir(parents=True, exist_ok=True)
        if fam == "iscas":
            v = bench / "iscas85_src" / "ISCAS85" / d / f"{d}.v"
            shutil.copy(v, src / f"{d}.v"); top = d; names = None
        else:
            circ = load_bench(str(bench / "epfl_bench" / f"{d}.bench")); top = TOP.get(d, d)   # "priority" is a SystemVerilog keyword
            names = write_verilog(circ, str(src / f"{d}.v"), module=top)
        meta[d] = {"family": fam, "top": top, "verilog": f"/work/designs/src/{d}/{d}.v", "names": names}
        n_gates = GATES.get(d, 1000)
        for plat in PLATFORMS:
            dd = orfs / "designs" / plat / d; dd.mkdir(parents=True, exist_ok=True)
            # explicit die/core: tiny designs cannot host the platform's power-strap pitch at 40 % utilisation (PDN-0185)
            cell_area = n_gates * CELL_AREA[plat]
            side = max(MIN_CORE[plat], (cell_area / (a.util / 100.0)) ** 0.5)
            side = float(int(side / 10 + 1) * 10); m = MARGIN[plat]
            (dd / "config.mk").write_text(f"""export DESIGN_NAME = {top}
export PLATFORM    = {plat}
export VERILOG_FILES = /work/designs/src/{d}/{d}.v
export SDC_FILE      = /work/designs/{plat}/{d}/constraint.sdc
export DIE_AREA  = 0 0 {side + 2*m:.1f} {side + 2*m:.1f}
export CORE_AREA = {m:.1f} {m:.1f} {side + m:.1f} {side + m:.1f}
export PLACE_DENSITY = 0.60
export TNS_END_PERCENT = 100
""")
            T = CLK_PERIOD[plat]
            (dd / "constraint.sdc").write_text(f"""# combinational design: virtual clock for I/O constraints only
create_clock -name vclk -period {T}
set_input_delay {0.2 * T:.3f} -clock vclk [all_inputs]
set_output_delay {0.2 * T:.3f} -clock vclk [all_outputs]
""")
    json.dump(meta, open(orfs / "designs" / "meta.json", "w"), indent=1)
    print("prepared", len(meta), "designs ×", len(PLATFORMS), "platforms under", orfs / "designs")


if __name__ == "__main__":
    main()
