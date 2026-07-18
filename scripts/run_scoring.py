#!/usr/bin/env python3
"""Scoring runner. Units:
  probe: (dataset x sseed x k)   -> <workdir>/probes/{dataset}__s{sseed}__k{k}.npz
  fm:    (dataset x fm_method)   -> <scores-dir>/{dataset}__{method}__sNA.npz
Scoring is an input-producing stage: resume is always on (score files are
content-keyed); rerun with a fresh --workdir to force recomputation.
After this, compute_scores.py turns probe stats into score files.

Launch (house convention):
    tmux new -s d1_scoring -d 'python scripts/run_scoring.py --config configs/d1.yaml'
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from runner_core import run_batch  # noqa: E402


def needed_sseeds(cfg) -> list[int]:
    if cfg["scoring_mode"] == "per_seed":
        return list(cfg["seeds"])
    return [int(cfg["shared_score_seed"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default="scoring")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--datasets", default=None, help="comma override")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--kind", choices=["probe", "fm"])
    ap.add_argument("--dataset")
    ap.add_argument("--sseed", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--method")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.worker:
        import d1_unit
        if a.kind == "probe":
            d1_unit.run_probe_unit(a.dataset, a.sseed, a.k, cfg, Path(a.out))
        else:
            d1_unit.run_fm_unit(a.dataset, a.method, cfg, Path(a.out))
        return

    datasets = a.datasets.split(",") if a.datasets else cfg["datasets"]
    import re
    loss_re = re.compile(r"^(el2n|grand|forgetting|entropy)(_e\d+)?$")
    loss_methods = [m for m in cfg["methods"] if loss_re.match(m)]
    fm_methods = [m for m in cfg["methods"] if m in cfg.get("fm_models", {})]
    workdir, scores_dir = Path(a.workdir), Path(a.scores_dir)
    probes_dir = workdir / "probes"

    units = []
    if loss_methods:
        for d in datasets:
            for ss in needed_sseeds(cfg):
                for k in range(cfg["probe"]["k"]):
                    units.append({"uid": f"probe__{d}__s{ss}__k{k}",
                                  "kind": "probe", "dataset": d,
                                  "sseed": ss, "k": k})
    for d in datasets:
        for m in fm_methods:
            units.append({"uid": f"fm__{d}__{m}", "kind": "fm",
                          "dataset": d, "method": m, "sseed": None, "k": None})

    def out_path(u):
        if u["kind"] == "probe":
            return probes_dir / f"{u['dataset']}__s{u['sseed']}__k{u['k']}.npz"
        return scores_dir / f"{u['dataset']}__{u['method']}__sNA.npz"

    def cmd(u):
        base = [sys.executable, str(Path(__file__).resolve()), "--worker",
                "--config", a.config, "--kind", u["kind"],
                "--dataset", u["dataset"], "--out", str(out_path(u))]
        if u["kind"] == "probe":
            base += ["--sseed", str(u["sseed"]), "--k", str(u["k"])]
        else:
            base += ["--method", u["method"]]
        return base

    tmo = cfg["timeouts"]["probe_unit"]
    counts = run_batch(workdir, units, out_path, cmd, timeout_s=tmo,
                       no_resume=False,
                       run_meta_extra={"stage": "scoring",
                                       "scoring_mode": cfg["scoring_mode"],
                                       "datasets": datasets})
    if counts["fail"] or counts["timeout"]:
        sys.exit("[scoring] some units failed — fix before computing scores.")


if __name__ == "__main__":
    main()
