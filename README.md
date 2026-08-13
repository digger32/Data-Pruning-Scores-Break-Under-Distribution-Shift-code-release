# Data-Pruning Scores Break Under Distribution Shift

Code, score files, coreset indices, and extended material for the paper

> **Data-Pruning Scores Break Under Distribution Shift**
> Vadim Tynchenko, Aleksei Borodulin, Vladislav Kukartsev, Sergei O. Kurashkin
> Bauman Moscow State Technical University
> 2nd Workshop on Curated Data for Efficient Learning (CDEL), ECCV 2026, Malmö — archival track.

Six data-pruning scores (EL2N, GraNd, Forgetting, Entropy, CLIP-feature and
DINOv2-feature prototype distance) are audited against random selection on
CIFAR-10/100 and their official corruption suites (CIFAR-10-C / CIFAR-100-C),
across five keep ratios and five seeds. The headline result is that clean test
accuracy does not certify a coreset under distribution shift, and that below
roughly a 50 % keep ratio random selection outranks every score.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | scoring (`run_scoring.py`, `compute_scores.py`), per-unit runner (`bench_runner.py`, `d1_unit.py`, `runner_core.py`), aggregation and statistics (`merge_results.py`, `aggregate_stats.py`), figures (`make_figures.py`, `make_jaccard.py`, `make_ablation_table.py`), review gate (`review_gate.py`, `test_gate.py`), coreset export (`export_coresets.py`) |
| `configs/` | `d1.yaml` (main grid), `d1_ablation.yaml` (scoring-choice ablation), `d1_smoke.yaml` |
| `gate_config.yaml` | the strict review-proofing gate applied to the final run |
| `pipeline.sh` | stage chain `smoke \| pilot \| full \| final [outdir]` |
| `scores_v2/` | the score files consumed by the reported runs |
| `coresets/`, `coresets_ablation/` | released coreset index files, with `SHA256SUMS` and `VERIFICATION.txt` |
| `supplementary.pdf` | extended figures and tables referenced by the paper (critical-difference diagram at keep 10 %, severity-resolved robustness, clean-versus-corruption trade-off, per-cell Friedman statistics, mechanism table, selection-stability Jaccard) |
| `requirements.lock` | the exact environment of the reported runs |

The supplementary PDF is **not** part of the workshop proceedings; the paper is
self-contained and this material is supporting evidence only.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py            # CIFAR + official -C suites, checksummed
bash pipeline.sh smoke                    # end-to-end check on a tiny slice
bash pipeline.sh final runs/final         # the from-scratch, resume-disabled run
```

Every unit is an independent job (dataset × score × keep ratio × seed) with
resume-by-skip and a per-unit timeout; results are one JSON per unit plus a
`manifest.jsonl`. `aggregate_stats.py` reproduces the Friedman, Nemenyi and
Wilcoxon analysis and the tables; `make_figures.py` rebuilds every figure from
a run directory.

## Provenance

The reported numbers come from one from-scratch run (360 units, resume
disabled, gate PASS, 13.6 GPU-hours) plus a self-contained scoring-choice
ablation (110 units, 4.0 GPU-hours), both on a single NVIDIA A100 (40 GB) under
the pinned environment in `requirements.lock` (PyTorch 2.13.0, torchvision
0.28.0, timm 1.0.28, numpy 2.5.1).

Each released score file is tied back to the run that consumed it:

```bash
python scripts/export_coresets.py --scores-dir scores_v2 --out coresets \
    --manifest runs/final_20260715_125212/results --verify class-hist
```

The check derives the coreset from the released bytes and requires its
per-class histogram to match the histogram recorded by **every** unit that used
that file; mismatches abort the export. The resulting verdicts are in
`coresets/VERIFICATION.txt` (54 files / 350 units) and
`coresets_ablation/VERIFICATION.txt` (22 files / 110 units).

## Licence

Code is released under the MIT licence (`LICENSE`). The score files and coreset
index files are released under CC BY 4.0. CIFAR-10/100, CIFAR-10-C/100-C, CLIP
and DINOv2 remain under their own licences and are not redistributed here.

## Citation

```bibtex
@inproceedings{tynchenko2026datapruning,
  title     = {Data-Pruning Scores Break Under Distribution Shift},
  author    = {Tynchenko, Vadim and Borodulin, Aleksei and Kukartsev, Vladislav
               and Kurashkin, Sergei O.},
  booktitle = {ECCV 2026 Workshop on Curated Data for Efficient Learning (CDEL)},
  year      = {2026}
}
```
