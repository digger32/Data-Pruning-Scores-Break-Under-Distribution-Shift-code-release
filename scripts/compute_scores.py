#!/usr/bin/env python3
"""Turn probe stats into score files (mean over the K probe models) and emit
the random 'scores' so every method except full flows through one selection
path. Output: <scores-dir>/{dataset}__{method}__s{sseed}.npz  (score, meta).
CPU-only, seconds."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml

VARIANT_RE = re.compile(r"^(el2n|grand)_e(\d+)$")


def probe_key_for(method: str, cfg: dict) -> str | None:
    """Map a method name to its key inside the probe npz files."""
    if method == "el2n":
        return f"el2n_e{cfg['probe']['el2n_epoch']}"
    if method == "grand":
        return f"grand_e{cfg['probe']['grand_epoch']}"
    if method in ("forgetting", "entropy"):
        return method
    if VARIANT_RE.match(method):
        return method
    return None


def save_score(path: Path, score: np.ndarray, meta: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, score=score.astype(np.float32), meta=json.dumps(meta))
    os.replace(tmp, path)
    return hashlib.sha256(score.astype(np.float32).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default="scoring")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    datasets = a.datasets.split(",") if a.datasets else cfg["datasets"]
    sseeds = (list(cfg["seeds"]) if cfg["scoring_mode"] == "per_seed"
              else [int(cfg["shared_score_seed"])])
    probes_dir = Path(a.workdir) / "probes"
    scores_dir = Path(a.scores_dir)
    loss_needed = [m for m in cfg["methods"] if probe_key_for(m, cfg)]

    summary = []
    for ds_i, d in enumerate(datasets):
        for ss in sseeds:
            files = []
            if loss_needed:
                files = [probes_dir / f"{d}__s{ss}__k{k}.npz"
                         for k in range(cfg["probe"]["k"])]
                missing = [str(f) for f in files if not f.exists()]
                if missing:
                    raise SystemExit(f"[scores] missing probe files: {missing}")
                stats = [np.load(f) for f in files]
                for m in loss_needed:
                    key = probe_key_for(m, cfg)
                    absent = [str(f) for f, s in zip(files, stats)
                              if key not in s]
                    if absent:
                        raise SystemExit(
                            f"[scores] probe files lack key '{key}' (legacy "
                            f"format?) — rerun probes into a fresh workdir: "
                            f"{absent[:2]}")
                    score = np.mean([s[key] for s in stats], axis=0)
                    sha = save_score(
                        scores_dir / f"{d}__{m}__s{ss}.npz", score,
                        {"dataset": d, "method": m, "probe_key": key,
                         "sseed": ss, "k_probes": cfg["probe"]["k"],
                         "probe_cfg": cfg["probe"]})
                    summary.append((d, m, ss, sha[:12]))
            if "random" in cfg["methods"]:
                if files:
                    ref = np.load(files[0])
                    n = len(ref["forgetting"])
                else:  # no probes ran: infer N from an fm score file
                    fm = next(scores_dir.glob(f"{d}__*__sNA.npz"), None)
                    if fm is None:
                        raise SystemExit(
                            "[scores] cannot infer train size for random "
                            "scores: no probe or fm score files present")
                    n = len(np.load(fm)["score"])
                rng = np.random.default_rng([ds_i, ss, 20260709])
                sha = save_score(scores_dir / f"{d}__random__s{ss}.npz",
                                 rng.random(n, dtype=np.float32),
                                 {"dataset": d, "method": "random",
                                  "sseed": ss})
                summary.append((d, "random", ss, sha[:12]))
    for row in summary:
        print("[scores] %s %s s%s sha=%s" % row)


if __name__ == "__main__":
    main()
