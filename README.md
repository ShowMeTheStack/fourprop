# fourprop: order-k Fourier propagation of signal probabilities

Code and result summaries for

> R. P. Das, J. Mishra, S. K. Sahay, A. Pathak, L. K. Vashishtha, S. Mishra,
> *Order-k Fourier propagation: unified and certified signal-probability estimation for logic circuits*,
> submitted to Integration, the VLSI Journal, 2026.

Every net of a combinational circuit is a vector in L²({0,1}ⁿ, μ_p); in the p-biased Fourier basis the signal
probability is the constant coefficient and an AND gate is a bilinear map with an explicit product rule.
Truncating at degree k gives a family of propagation algorithms: order 0 is the independence method (COP),
order 1 is the Boolean approximation method (BAM) exactly, orders 2 to 5 are new. The paper proves a per-gate error
bound at every order, a sound whole-circuit certificate and a first-order model of the propagated error, and evaluates
everything on ISCAS-85 (four netlist forms) and the EPFL suite against exact ground truth, against GPU Monte Carlo
(time against accuracy, Fourier control variates) and inside an OpenROAD place-and-route flow on NanGate45 and sky130.

## Layout

| path | contents |
|---|---|
| `fourprop/` | the engine: `circuit.py` (bench/AIG parser, evaluation), `fourier.py` (sparse order-k propagation, certified bounds), `dense.py` (dense GPU/NumPy orders 0–2 with feasibility projection), `dense_k.py` (generic dense order-k engine, symmetric tensors), `cv.py` (Monte Carlo with Fourier control variates), `exact.py`, `exactgpu.py` (exact coefficients by enumeration), `bdd.py` (exact probabilities via CUDD), `mc.py`, `mc_torch.py` (Monte Carlo), `cutting.py` (Savir–Ditlow–Bardell bounds), `synth.py`, `verilog.py`, `verilog_out.py`, `liberty.py` (Yosys/ABC flows, Liberty cell functions, power, structural Verilog for ORFS), `justify.py` (probability-guided justification) |
| `exp/` | experiment drivers, one JSON per task, resumable: `gate_runner.py`, `fast_runner.py`, `proj_runner.py`, `phase2_runner.py`, `oracle_runner.py` (one-gate oracle error, tail mass by depth), `power_runner.py` (Liberty-based power), `justify_runner.py` (probability-guided justification), `timing_runner.py`, `bound_tightness.py`; extension runs: `ablation_runner.py` (orders 0–5), `pareto_runner.py` (Monte Carlo sweep and break-even), `cv_runner.py` (certified widths vs Monte Carlo intervals, control variates), `openroad_runner.py` + `orfs_prep.py` (OpenROAD flow scripts in a container, OpenSTA power with per-net activities), `errmodel_runner.py` + `errmodel_dump.py` (error model M1/M2), `ext_common.py`; `audit_numbers.py` (recomputes every number quoted in the paper from `results/`) |
| `scripts/` | `run_*.sh` wrappers (flock, cron-safe), `run_all.sh` (the whole pipeline under one `JOBBASE`), `run_ext.sh` (ablation → pareto → cv → errmodel) and `run_openroad.sh` (the extension runs), `make_tables.py` (LaTeX tables from the result rows), `make_figures.py` (the paper's plots from the result rows; needs matplotlib), `collect.sh` |
| `results/` | summaries (`*_rows.json`, one per job) of the clean run of 11 November 2026 on an idle machine (32-core, RTX 3080 + 3060); `bound_tightness.json`; `ext/` the extension runs of 12 November 2026 (ablation, pareto, cv with per-net widths, openroad, errmodel) |
| `tests/` | 234 pytest tests: product rule against enumeration, bound soundness, dense engines (orders 0–2 and the generic order-k engine) against the sparse one and against exact coefficients, Liberty parsing, justification |

## Reproducing

```
pip install -r requirements.txt           # torch (CUDA build for the GPU backends), numpy, dd (CUDD), pytest
python -m pytest -q tests                 # engine tests, < 1 min on CPU
```

Benchmarks are not included: ISCAS-85 Verilog (converted to AIG bench with `yosys` + `abc`, see `fourprop/synth.py`),
the EPFL combinational suite, and the sky130_fd_sc_hd / NanGate45 Liberty files are downloaded by the user into a
`bench/` directory (`bench/iscas85`, `bench/iscas85_src`, `bench/iscas85_var`, `bench/epfl_bench`, `bench/liberty`).
The whole pipeline then runs with

```
JOBBASE=/path/to/run scripts/run_all.sh   # gate → fast → proj → phase2 → phase2b → oracle → power → justify → justify_hard → justify_hardr → timing
```

(the wrappers assume the layout `/extra/fourprop/{venv,repo,bench}`; edit `ROOT` in `scripts/run_*.sh` otherwise).
Each job writes one JSON per task and a `DONE` marker, so a reboot or a kill resumes where it stopped.
The timing job only starts when the one-minute load average is below `MAXLOAD` (default 6).

To regenerate the paper's tables and figures from the shipped summaries (written to `tables/` and `figs/`):

```
mkdir -p results/_flat && ln -sf ../proj/proj_rows.json ../phase2/phase2_rows.json ../gate/gate_rows.json results/_flat/
python scripts/make_tables.py --algebra results/_flat --out tables --ext results/ext   # the timing, justify_hard(r) and phase2b tables are read from the sibling directories
RESULTS_EXT=results/ext python scripts/make_figures.py results figs
python exp/audit_numbers.py results results/ext       # prints every number quoted in the paper
```

The extension runs need the same `bench/` layout; `scripts/run_ext.sh` chains them under `JOBBASE`, and
`scripts/run_openroad.sh` runs the OpenROAD flow scripts in a `docker.io/openroad/orfs` container (podman or docker)
after `exp/orfs_prep.py` has written the design directories.

## License

Copyright (C) 2026 Jyotiprakash Mishra. GNU General Public License, version 3 only (GPL-3.0-only; verbatim text in
`LICENSE`), with no Classpath or other linking exception, plus an additional term under GPL Section 7(b) in `NOTICE`:
every use, extension or redistribution must preserve the copyright notice and attribute the work to Jyotiprakash Mishra,
citing the paper above. Every source file carries the copyright and SPDX notice.
