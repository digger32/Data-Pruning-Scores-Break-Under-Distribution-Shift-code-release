#!/usr/bin/env python3
"""Shared job orchestration (house pattern): independent units, resume by
existing output, per-unit hard timeout in an isolated subprocess, one manifest
record per unit. Used by bench_runner.py and run_scoring.py; do not fork it."""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def append_manifest(outdir: Path, record: dict) -> None:
    with (outdir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_batch(outdir, units, unit_out_path, worker_cmd, timeout_s: int,
              no_resume: bool, run_meta_extra: dict | None = None) -> dict:
    """units: list of dicts, each with a unique 'uid' plus its axis values.
    unit_out_path(u) -> Path of the unit's result file (resume key).
    worker_cmd(u) -> argv list for the isolated worker subprocess."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    meta = {"run_started": run_started, "no_resume": bool(no_resume),
            "timeout_s": timeout_s, "n_units": len(units)}
    meta.update(run_meta_extra or {})
    atomic_write_json(outdir / "run_meta.json", meta)

    counts = {"ok": 0, "skip": 0, "fail": 0, "timeout": 0}
    print(f"[runner] {len(units)} units | outdir={outdir} | "
          f"no_resume={no_resume} | per-unit timeout={timeout_s}s", flush=True)
    for u in units:
        uid = u["uid"]
        out_path = unit_out_path(u)
        axes = {k: v for k, v in u.items() if k != "uid"}
        if out_path.exists() and not no_resume:
            counts["skip"] += 1
            append_manifest(outdir, {"unit": uid, **axes, "status": "skip",
                                     "started": run_started,
                                     "finished": datetime.now(timezone.utc).isoformat(),
                                     "wall_s": 0.0, "no_resume": no_resume})
            print(f"[skip] {uid} (output exists)", flush=True)
            continue
        if out_path.exists() and no_resume:
            out_path.unlink()  # fresh pass recomputes every unit

        t0 = time.time()
        status = "ok"
        try:
            subprocess.run(worker_cmd(u), timeout=timeout_s, check=True)
            if not out_path.exists():
                status = "fail(no_output)"
                counts["fail"] += 1
                print(f"[FAIL] {uid} exited 0 but wrote no output", flush=True)
            else:
                counts["ok"] += 1
                print(f"[ok] {uid} ({time.time() - t0:.1f}s)", flush=True)
        except subprocess.TimeoutExpired:
            status = "timeout"
            counts["timeout"] += 1
            print(f"[TIMEOUT] {uid} > {timeout_s}s — unit killed, batch continues",
                  flush=True)
        except subprocess.CalledProcessError as e:
            status = f"fail(rc={e.returncode})"
            counts["fail"] += 1
            print(f"[FAIL] {uid} rc={e.returncode} — batch continues", flush=True)

        append_manifest(outdir, {"unit": uid, **axes, "status": status,
                                 "started": run_started,
                                 "finished": datetime.now(timezone.utc).isoformat(),
                                 "wall_s": round(time.time() - t0, 1),
                                 "no_resume": no_resume})
    print(f"[runner] done | ok={counts['ok']} skip={counts['skip']} "
          f"fail={counts['fail']} timeout={counts['timeout']}", flush=True)
    return counts
