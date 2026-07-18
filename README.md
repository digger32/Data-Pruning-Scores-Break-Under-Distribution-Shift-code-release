# Data-Pruning Scores Break Under Distribution Shift — code release

Code for the benchmark audit of six data-pruning scores (EL2N, GraNd,
Forgetting, Entropy, CLIP-feature and DINOv2-feature prototype distance)
against random selection on CIFAR-10/100 and their official corruption
suites (CIFAR-10-C / CIFAR-100-C), across five keep ratios and five seeds.

Anonymised review copy. Licence and citation information will be added at
camera-ready.

## Layout

- `scripts/` — the full pipeline: scoring (`run_scoring.py`,
  `compute_scores.py`), the per-unit benchmark runner (`bench_runner.py`,
  `d1_unit.py`, `runner_core.py`), aggregation and statistics
  (`merge_results.py`, `aggregate_stats.py`), figures (`make_figures.py`,
  `make_jaccard.py`, `make_ablation_table.py`), and the review gate
  (`review_gate.py`, `test_gate.py`).
- `configs/` — `d1.yaml` (main grid), `d1_ablation.yaml` (scoring-choice
  ablation), `d1_smoke.yaml` (smoke test).
- `gate_config.yaml` — the strict review-proofing gate applied to the
  final run (resume disabled, all units present, score files sha256-pinned).
- `pipeline.sh` — stage chain: `smoke | pilot | full | final [outdir]`;
  each stage runs scoring → benchmark units → merge → stats → gate →
  figures, and a failed gate stops the chain before figures.

## Minimal run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py            # CIFAR + official -C suites, checksummed
bash pipeline.sh smoke                    # end-to-end check on a tiny slice
bash pipeline.sh final runs/final         # the from-scratch, resume-disabled run
```

`scripts/export_coresets.py` materialises the released coreset index files
from the score files and writes a `SHA256SUMS` manifest over both. With
`--manifest <run>/results` every score file is first verified against the
frozen run's unit records, either byte-exactly (`--verify sha`) or
semantically (`--verify class-hist`: the derived coreset's per-class
histogram must match the histogram recorded by every unit that consumed
the file); per-file verdicts are written to `coresets/VERIFICATION.txt`.

Every unit is an independent job (dataset × score × keep-ratio × seed) with
resume-by-skip and a per-unit timeout; results are one JSON per unit plus a
`manifest.jsonl`. `aggregate_stats.py` reproduces the Friedman/Nemenyi/
Wilcoxon analysis and the tables; `make_figures.py` rebuilds every figure in
the paper from the run directory.

## Reported environment

The paper's numbers come from a single from-scratch run (360 units, resume
disabled, gate PASS) under the pinned environment in `requirements.lock`
(PyTorch 2.13.0, torchvision 0.28.0, timm 1.0.28, numpy 2.5.1) on one
NVIDIA A100 (40 GB). Coreset index files and their sha256 checksums
accompany the release.
