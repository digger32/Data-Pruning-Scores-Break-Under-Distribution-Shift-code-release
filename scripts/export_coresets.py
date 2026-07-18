#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialise the coreset index files promised in the paper's availability
statement, from the frozen score files.

A coreset is fully determined by a score file and a keep ratio: the runner
keeps the hardest fraction via a stable argsort (see d1_unit.py:
``idx = np.argsort(-arr, kind="stable")[:keep]``). This script reproduces
that derivation exactly for every (dataset, method, sseed) score file found
in the scores directory and every keep ratio, and writes one .npz of kept
indices per combination, plus a SHA256SUMS manifest of both the score files
and the exported index files.

USAGE (repo root, after the final run):
  python scripts/export_coresets.py --scores-dir scores_v2 \
      --out coresets --ratios 10 30 50 70 90 \
      --manifest runs/final_*/results

With --manifest, every score file is verified against the frozen run's
unit JSONs before export. Two modes (--verify):
  sha        exact byte match of the file against the recorded score_sha256
             (default; fails for files re-derived after the run, since the
             npz container embeds timestamps even when values are equal).
  class-hist semantic match: the coreset is derived from the CURRENT file
             (stable argsort, keep-hardest) and its per-class histogram is
             compared to the class_hist recorded by EVERY unit that consumed
             the file. Requires --data-dir with the CIFAR train sets. A file
             passes only if all of its units match exactly; the ordering the
             released indices encode is then the frozen run's ordering, even
             if the container bytes differ.
Any failure aborts the export; a VERIFICATION.txt with per-file verdicts is
written next to the exported indices.

Output: coresets/{dataset}__{method}__{stag}__r{ratio}.npz with a single
array ``indices`` (int64, sorted by hardness rank), and
coresets/SHA256SUMS covering scores_v2/*.npz and coresets/*.npz.
"""
from __future__ import annotations
import argparse
import glob as _glob
import hashlib
import json
from pathlib import Path

import numpy as np

N_TRAIN = {"cifar10": 50000, "cifar100": 50000}


def load_labels(dataset: str, data_dir: str) -> "np.ndarray":
    """Train labels exactly as the runner loads them (d1_unit.load_data)."""
    import torchvision
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[dataset]
    tr = cls(root=data_dir, train=True, download=True)
    return np.asarray(tr.targets, dtype=np.int64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="scores_v2")
    ap.add_argument("--out", default="coresets")
    ap.add_argument("--ratios", type=int, nargs="+",
                    default=[10, 30, 50, 70, 90])
    ap.add_argument("--manifest", default=None,
                    help="results dir (or glob) of the frozen run; enables "
                         "verification against its unit JSONs")
    ap.add_argument("--verify", choices=["sha", "class-hist"], default="sha")
    ap.add_argument("--data-dir", default="data",
                    help="dataset root for --verify class-hist")
    args = ap.parse_args()

    expected = {}
    if args.manifest:
        unit_files = sorted(_glob.glob(str(Path(args.manifest) / "*.json"))) \
            or sorted(_glob.glob(args.manifest + "/*.json")) \
            or sorted(_glob.glob(args.manifest))
        units_by_file = {}
        for uf in unit_files:
            d = json.load(open(uf))
            if d.get("score_file") and d.get("score_sha256"):
                name = Path(d["score_file"]).name
                prev = expected.setdefault(name, d["score_sha256"])
                assert prev == d["score_sha256"], \
                    f"manifest inconsistent for {name}"
                units_by_file.setdefault(name, []).append(
                    {"n_train": int(d["n_train"]),
                     "class_hist": d["class_hist"],
                     "uid": d.get("uid", Path(uf).stem)})
        if not expected:
            raise SystemExit(f"--manifest {args.manifest}: no unit JSONs "
                             "with score_file/score_sha256 found")
        print(f"manifest loaded: {len(expected)} distinct score files")

    scores_dir, out = Path(args.scores_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    score_files = sorted(scores_dir.glob("*.npz"))
    if not score_files:
        raise SystemExit(f"no score files in {scores_dir}")

    mismatches = []
    verdicts = []
    n_exported = 0
    labels_cache = {}
    for sf in score_files:
        dataset, method, stag = sf.stem.split("__")  # e.g. cifar10, el2n, s0
        verdict = None
        if expected:
            got = sha256(sf)
            want = expected.get(sf.name)
            if want is None:
                print(f"NOTE {sf.name}: not referenced by the manifest; "
                      "skipping")
                continue
            if got == want:
                verdict = "sha-exact"
            elif args.verify == "class-hist":
                if dataset not in labels_cache:
                    labels_cache[dataset] = load_labels(dataset,
                                                        args.data_dir)
                ytr = labels_cache[dataset]
                order_v = np.argsort(
                    -np.load(sf)["score"].astype(np.float64), kind="stable")
                bad = []
                for u in units_by_file[sf.name]:
                    idx = order_v[:u["n_train"]]
                    hist = np.bincount(
                        ytr[idx], minlength=len(u["class_hist"])).tolist()
                    if hist != u["class_hist"]:
                        bad.append(u["uid"])
                if bad:
                    mismatches.append(
                        (sf.name, want,
                         f"{got} AND class-hist differs on "
                         f"{len(bad)}/{len(units_by_file[sf.name])} units "
                         f"(e.g. {bad[0]})"))
                    continue
                verdict = (f"class-hist verified over "
                           f"{len(units_by_file[sf.name])} units "
                           "(container re-derivation; ordering identical "
                           "to the frozen run)")
            else:
                mismatches.append((sf.name, want, got))
                continue
        verdicts.append((sf.name, verdict or "unverified (no manifest)"))
        n_exported += 1
        arr = np.load(sf)["score"].astype(np.float64)
        n = N_TRAIN.get(dataset, len(arr))
        assert len(arr) == n, f"{sf}: {len(arr)} scores, expected {n}"
        order = np.argsort(-arr, kind="stable")  # keep hardest, as in d1_unit
        lines.append(f"{sha256(sf)}  {scores_dir.name}/{sf.name}")
        for r in args.ratios:
            keep = int(round(n * r / 100))
            op = out / f"{dataset}__{method}__{stag}__r{r}.npz"
            tmp = op.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, indices=order[:keep].astype(np.int64))
            tmp.rename(op)
            lines.append(f"{sha256(op)}  {out.name}/{op.name}")

    if mismatches:
        for name, want, got in mismatches:
            print(f"MISMATCH {name}\n  manifest {want}\n  on disk  {got}")
        raise SystemExit(
            f"{len(mismatches)} score file(s) do not match the frozen run's "
            "manifest -- these are not the files the run consumed. Export "
            "aborted; nothing partial was released as final.")

    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    if verdicts:
        (out / "VERIFICATION.txt").write_text(
            "Provenance verification against the frozen run manifest\n"
            + "\n".join(f"{n}: {v}" for n, v in verdicts) + "\n")
        n_sha = sum(1 for _, v in verdicts if v == "sha-exact")
        print(f"verified: {n_sha} sha-exact, {len(verdicts)-n_sha} "
              "class-hist")
    print(f"exported {n_exported} of {len(score_files)} score files x "
          f"{len(args.ratios)} ratios -> {out}/ "
          f"(+ SHA256SUMS, {len(lines)} entries)")


if __name__ == "__main__":
    main()
