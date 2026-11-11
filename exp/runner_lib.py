# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""Generic resumable task scheduler: one JSON per task, atomic writes, per-kind timeouts, worker limit."""
from __future__ import annotations
import json, os, sys, time, traceback, multiprocessing as mp
from pathlib import Path


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _worker(fn_module: str, fn_name: str, task: dict, out_path: str, extra: dict) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import importlib
    fn = getattr(importlib.import_module(fn_module), fn_name)
    t0 = time.time()
    res = {"task": task, "ok": False}
    try:
        res.update(fn(task, extra) or {})
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()
    res["task_wall"] = time.time() - t0            # whole worker (imports, loads, warm-up, work)
    res.setdefault("runtime", res["task_wall"])     # a task that measured its own runtime keeps it
    atomic_write(Path(out_path), res)


def run_tasks(root: Path, tasks: list, fn_module: str, fn_name: str, extra: dict, timeouts: dict,
              workers: int, serial_kinds=("mc",), kind_order=None, on_done=None) -> int:
    (root / "results").mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    log_path = root / "logs" / "runner.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    if (root / "DONE").exists():
        log("DONE marker present; nothing to do"); return 0
    if not tasks:
        log("no tasks; refusing to run"); return 1
    pending = [t for t in tasks if not (root / "results" / f"{t['id']}.json").exists()]
    log(f"{len(tasks)} tasks, {len(pending)} pending")
    rank = kind_order or {}
    order = sorted(pending, key=lambda t: rank.get(t["kind"], 99))
    ctx = mp.get_context("spawn")
    running: list = []
    while order or running:
        still = []
        for proc, t, t0 in running:
            if proc.is_alive():
                if time.time() - t0 > timeouts.get(t["kind"], 3600):
                    proc.terminate(); proc.join(5)
                    atomic_write(root / "results" / f"{t['id']}.json", {"task": t, "ok": False, "error": "timeout", "runtime": time.time() - t0})
                    log(f"TIMEOUT {t['id']}")
                else:
                    still.append((proc, t, t0))
            else:
                log(f"finished {t['id']} (exit {proc.exitcode}, {time.time()-t0:.0f}s)")
                if not (root / "results" / f"{t['id']}.json").exists():
                    atomic_write(root / "results" / f"{t['id']}.json", {"task": t, "ok": False, "error": f"worker exit {proc.exitcode}", "runtime": time.time() - t0})
        running = still
        while order and len(running) < workers:
            t = order[0]
            if t["kind"] in serial_kinds and any(r[1]["kind"] == t["kind"] for r in running):
                # one task of each serial kind at a time (each owns a GPU); try the next task instead
                nxt = next((x for x in order if x["kind"] not in serial_kinds or not any(r[1]["kind"] == x["kind"] for r in running)), None)
                if nxt is None:
                    break
                order.remove(nxt); t = nxt
            if t in order:
                order.remove(t)
            out = root / "results" / f"{t['id']}.json"
            proc = ctx.Process(target=_worker, args=(fn_module, fn_name, t, str(out), extra), daemon=False)
            proc.start(); running.append((proc, t, time.time())); log(f"started {t['id']} pid {proc.pid}")
        time.sleep(3)
    log("all tasks finished")
    if on_done:
        on_done(root)
    (root / "DONE").touch(); log("DONE")
    return 0
