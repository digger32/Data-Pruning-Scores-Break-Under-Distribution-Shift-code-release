#!/usr/bin/env bash
# D1 pipeline: smoke | pilot | full | final [outdir]
# Chain per stage: (gate self-test) -> scoring -> scores -> bench -> merge ->
# stats -> GATE -> figures -> report. set -e: a failed gate stops the chain
# BEFORE figures on blocking stages.
#
# Launch every stage inside tmux (house convention), e.g.:
#   tmux new -s d1_smoke  -d 'bash pipeline.sh smoke  2>&1 | tee logs/smoke.log'
#   tmux new -s d1_pilot  -d 'bash pipeline.sh pilot  2>&1 | tee logs/pilot.log'
#   tmux new -s d1_full   -d 'bash pipeline.sh full   2>&1 | tee logs/full.log'
#   tmux new -s d1_final  -d 'bash pipeline.sh final  2>&1 | tee logs/final.log'
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:?usage: pipeline.sh smoke|pilot|full|final|ablation [outdir]}
STAMP=$(date +%Y%m%d_%H%M%S)
SCORING_DIR=${SCORING_DIR:-scoring}   # strengthening run: export SCORING_DIR=scoring_v2
SCORES_DIR=${SCORES_DIR:-scores}      #                    export SCORES_DIR=scores_v2
mkdir -p logs runs

post() { # $1=outdir $2=gate_config $3=gate_mode(block|advisory)
  python scripts/merge_results.py "$1"
  python scripts/aggregate_stats.py "$1"
  if [ "$3" = advisory ]; then
    python scripts/review_gate.py "$1" --config "$2" \
      || echo "[pipeline] GATE FAILED (advisory at this stage) — numbers are NOT freezable"
  else
    python scripts/review_gate.py "$1" --config "$2"
  fi
  python scripts/make_figures.py "$1"
  echo "[pipeline] $STAGE done -> $1 (tables/ stats/ figures/)"
}

case "$STAGE" in
  smoke)
    OUT=${2:-runs/smoke_$STAMP}
    python scripts/test_gate.py
    python scripts/run_scoring.py   --config configs/d1_smoke.yaml --workdir "$OUT/scoring" --scores-dir "$OUT/scores"
    python scripts/compute_scores.py --config configs/d1_smoke.yaml --workdir "$OUT/scoring" --scores-dir "$OUT/scores"
    python scripts/bench_runner.py  --config configs/d1_smoke.yaml --outdir "$OUT" --scores-dir "$OUT/scores" --no-resume
    post "$OUT" gate_config_smoke.yaml block
    ;;
  pilot)  # 4 real-recipe units to time a unit; decision input for D4-lite
    OUT=${2:-runs/pilot_$STAMP}
    python scripts/run_scoring.py   --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR" --datasets cifar10
    python scripts/compute_scores.py --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR" --datasets cifar10
    python scripts/bench_runner.py  --config configs/d1.yaml --outdir "$OUT" --scores-dir "$SCORES_DIR" \
        --datasets cifar10 --methods el2n,random --ratios 30 --seeds 0,1 --no-full --no-resume
    post "$OUT" gate_config_smoke.yaml block
    ;;
  full)   # Stage 2: whole grid, resume ON (interruptible); gate advisory
    OUT=${2:-runs/full}
    python scripts/run_scoring.py   --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/compute_scores.py --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/bench_runner.py  --config configs/d1.yaml --outdir "$OUT" --scores-dir "$SCORES_DIR"
    post "$OUT" gate_config.yaml advisory
    ;;
  final)  # Stage 3: fresh dir, resume DISABLED, STRICT gate
    OUT=${2:-runs/final_$STAMP}
    python scripts/run_scoring.py   --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/compute_scores.py --config configs/d1.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/bench_runner.py  --config configs/d1.yaml --outdir "$OUT" --scores-dir "$SCORES_DIR" --no-resume
    post "$OUT" gate_config.yaml block
    if grep -q '^scoring_mode: per_seed' configs/d1.yaml; then
      python scripts/make_jaccard.py --config configs/d1.yaml --scores-dir "$SCORES_DIR" --outdir "$OUT" \
        || echo "[pipeline] jaccard skipped"
    fi
    echo "[pipeline] FINAL numbers frozen in $OUT — cite ONLY these in the manuscript."
    ;;
  ablation)  # keep-50 scoring-choice ablation; probes reused from SCORING_DIR
    OUT=${2:-runs/ablation}
    python scripts/run_scoring.py   --config configs/d1_ablation.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/compute_scores.py --config configs/d1_ablation.yaml --workdir "$SCORING_DIR" --scores-dir "$SCORES_DIR"
    python scripts/bench_runner.py  --config configs/d1_ablation.yaml --outdir "$OUT" --scores-dir "$SCORES_DIR" --no-resume
    python scripts/merge_results.py "$OUT"
    python scripts/aggregate_stats.py "$OUT"
    python scripts/review_gate.py "$OUT" --config gate_config.yaml
    python scripts/make_ablation_table.py --ablation-dir "$OUT"
    echo "[pipeline] ablation done -> $OUT"
    ;;
  *) echo "unknown stage: $STAGE"; exit 2;;
esac
