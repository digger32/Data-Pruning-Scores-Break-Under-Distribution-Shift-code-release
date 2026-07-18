#!/usr/bin/env python3
"""Unit-test the review-proofing gate BEFORE any real run (the B5/MAU lesson):
build a synthetic CLEAN run (must exit 0) and a synthetic DIRTY run (resume
on, a skipped unit, a mid-run score-hash change, missing stats; must exit 1),
plus targeted single-fault runs so each check is proven to bite individually.
Run: python scripts/test_gate.py   (exits non-zero if the gate misbehaves)"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
GATE = HERE / "review_gate.py"
from review_gate import CANON_CORRUPTIONS, CANON_SEVERITIES  # noqa: E402

AXES = {"datasets": ["cifar10", "cifar100"], "methods": ["random", "el2n"],
        "ratios": [30], "seeds": [0, 1], "include_full": False}


def score_sha(arr):
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


def build_run(root: Path, *, no_resume=True, skip_one=False, fake=False,
              drop_unit=False, break_hash=False, missing_sev=False,
              stats=True, tamper_disk_score=False):
    run = root / "run"
    (run / "results").mkdir(parents=True)
    (run / "stats").mkdir()
    scores_dir = root / "scores"
    scores_dir.mkdir()
    started = "2026-07-09T00:00:00+00:00"
    meta = {"run_started": started, "no_resume": no_resume, "fake": fake,
            "axes": AXES,
            "corruption_suite": {"corruptions": CANON_CORRUPTIONS,
                                 "severities": CANON_SEVERITIES,
                                 "subset": None}}
    (run / "run_meta.json").write_text(json.dumps(meta))

    rng = np.random.default_rng(0)
    shas = {}
    for d in AXES["datasets"]:
        for m in AXES["methods"]:
            arr = rng.random(64, dtype=np.float32)
            np.savez(scores_dir / f"{d}__{m}__s0.npz", score=arr, meta="{}")
            shas[(d, m)] = score_sha(arr)
    if tamper_disk_score:
        d, m = "cifar10", "el2n"
        np.savez(scores_dir / f"{d}__{m}__s0.npz",
                 score=rng.random(64, dtype=np.float32), meta="{}")

    manifest = []
    units = [(d, m, r, s) for d in AXES["datasets"] for m in AXES["methods"]
             for r in AXES["ratios"] for s in AXES["seeds"]]
    for i, (d, m, r, s) in enumerate(units):
        uid = f"{d}__{m}__r{r}__seed{s}"
        if skip_one and i == 0:
            manifest.append({"unit": uid, "status": "skip", "started": started})
            continue
        if drop_unit and i == 0:
            continue
        sha = shas[(d, m)]
        if break_hash and i == 1:
            sha = "deadbeef" + sha[8:]
        corr = {c: {str(v): 0.5 for v in CANON_SEVERITIES}
                for c in CANON_CORRUPTIONS}
        if missing_sev and i == 0:
            del corr[CANON_CORRUPTIONS[0]]["3"]
        (run / "results" / f"{uid}.json").write_text(json.dumps({
            "dataset": d, "method": m, "ratio": r, "seed": s, "sseed": 0,
            "score_file": f"scores/{d}__{m}__s0.npz", "score_sha256": sha,
            "metrics": {"clean_acc": 0.9, "mca": 0.6}, "corr": corr,
            "fake": fake}))
        manifest.append({"unit": uid, "status": "ok", "started": started})
    (run / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n")
    if stats:
        (run / "stats" / "omnibus.json").write_text("{}")
        (run / "stats" / "posthoc.json").write_text("{}")
    return run


def run_gate(run: Path, cfg_path: Path, repo_root: Path) -> int:
    p = subprocess.run([sys.executable, str(GATE), str(run),
                        "--config", str(cfg_path),
                        "--repo-root", str(repo_root)],
                       capture_output=True, text=True)
    return p.returncode


def main():
    cfg = {"require_stats": True, "require_full_suite": True,
           "verify_score_files_on_disk": True,
           "comparative_claims": [
               {"id": "score_pruning_loses_robustness_vs_random",
                "independent_datasets": ["cifar10", "cifar100"],
                "require_all": True},
               {"id": "fm_feature_selection_more_shift_stable",
                "independent_datasets": ["cifar10", "cifar100"],
                "require_all": True}]}
    cases = [
        ("CLEAN", {}, 0),
        ("DIRTY combo", {"no_resume": False, "skip_one": True,
                         "break_hash": True, "stats": False}, 1),
        ("resume enabled", {"no_resume": False}, 1),
        ("skipped unit", {"skip_one": True}, 1),
        ("missing unit (grid)", {"drop_unit": True}, 1),
        ("fake data", {"fake": True}, 1),
        ("score hash changed mid-run", {"break_hash": True}, 1),
        ("score tampered on disk", {"tamper_disk_score": True}, 1),
        ("missing severity", {"missing_sev": True}, 1),
        ("missing stats artifacts", {"stats": False}, 1),
    ]
    failures = []
    for name, kw, want in cases:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = build_run(root, **kw)
            cfg_path = root / "gate_config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            rc = run_gate(run, cfg_path, root)
            got = 0 if rc == 0 else 1
            status = "OK " if got == want else "BAD"
            print(f"[{status}] {name:32s} expected exit {want}, got {rc}")
            if got != want:
                failures.append(name)
    if failures:
        sys.exit(f"[test_gate] GATE MISBEHAVES on: {failures}")
    print("[test_gate] all cases behave: gate proven on synthetic "
          "clean/dirty runs.")


if __name__ == "__main__":
    main()
