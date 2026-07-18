#!/usr/bin/env python3
"""D1 benchmark runner. Unit = (dataset x method x keep-ratio x seed); the
full-data baseline rides as method 'full' at ratio 100. Independent units,
resume, per-unit hard timeout (runner_core). Score files must exist first
(run_scoring.py + compute_scores.py).

Stage 2 (resume ON):
    tmux new -s d1_full -d 'bash pipeline.sh full 2>&1 | tee logs/full.log'
Stage 3 (FINAL, fresh dir, resume DISABLED):
    tmux new -s d1_final -d 'bash pipeline.sh final 2>&1 | tee logs/final.log'
"""
import argparse
import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from runner_core import run_batch  # noqa: E402


def unit_uid(d, m, r, s):
    return f"{d}__{m}__r{r}__seed{s}"


def build_units(cfg, datasets, methods, ratios, seeds, include_full):
    units = []
    for d in datasets:
        for m in methods:
            for r in ratios:
                for s in seeds:
                    ss = (s if cfg["scoring_mode"] == "per_seed"
                          else int(cfg["shared_score_seed"]))
                    units.append({"uid": unit_uid(d, m, r, s), "dataset": d,
                                  "method": m, "ratio": r, "seed": s,
                                  "sseed": ss})
        if include_full:
            for s in seeds:
                units.append({"uid": unit_uid(d, "full", 100, s),
                              "dataset": d, "method": "full", "ratio": 100,
                              "seed": s, "sseed": None})
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None, help="required in orchestrator mode")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true",
                    help="FINAL pass: recompute every unit into a fresh outdir")
    ap.add_argument("--datasets", default=None, help="comma override")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--ratios", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--no-full", dest="no_full", action="store_true")
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--ratio", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sseed", type=int, default=-1)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.worker:
        import d1_unit
        d1_unit.run_train_unit(a.dataset, a.method, a.ratio, a.seed,
                               a.sseed, cfg, a.scores_dir, Path(a.out))
        return

    if not a.outdir:
        ap.error("--outdir is required in orchestrator mode")
    datasets = a.datasets.split(",") if a.datasets else cfg["datasets"]
    methods = a.methods.split(",") if a.methods else cfg["methods"]
    ratios = ([int(r) for r in a.ratios.split(",")] if a.ratios
              else cfg["keep_ratios"])
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds
             else cfg["seeds"])
    include_full = cfg.get("include_full_baseline", True) and not a.no_full

    outdir = Path(a.outdir)
    results_dir = outdir / "results"
    units = build_units(cfg, datasets, methods, ratios, seeds, include_full)

    def out_path(u):
        return results_dir / f"{u['uid']}.json"

    def cmd(u):
        return [sys.executable, str(Path(__file__).resolve()), "--worker",
                "--config", a.config, "--scores-dir", a.scores_dir,
                "--dataset", u["dataset"], "--method", u["method"],
                "--ratio", str(u["ratio"]), "--seed", str(u["seed"]),
                "--sseed", str(u["sseed"] if u["sseed"] is not None else -1),
                "--out", str(out_path(u))]

    import os
    cfg_sha = hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    meta = {"stage": "bench",
            "axes": {"datasets": datasets, "methods": methods,
                     "ratios": ratios, "seeds": seeds,
                     "include_full": include_full},
            "scoring_mode": cfg["scoring_mode"],
            "shared_score_seed": cfg["shared_score_seed"],
            "corruption_suite": {"corruptions": cfg["eval"]["corruptions"],
                                 "severities": cfg["eval"]["severities"],
                                 "subset": cfg["eval"].get("subset")},
            "config_file": a.config, "config_sha256": cfg_sha,
            "fake": os.environ.get("D1_FAKE", "0") == "1"}
    counts = run_batch(outdir, units, out_path, cmd,
                       timeout_s=cfg["timeouts"]["train_unit"],
                       no_resume=a.no_resume, run_meta_extra=meta)
    if a.no_resume and (counts["fail"] or counts["timeout"]):
        sys.exit("[bench] FINAL pass has failed/timed-out units — not freezable.")


if __name__ == "__main__":
    main()
