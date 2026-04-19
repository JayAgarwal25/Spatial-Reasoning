#!/usr/bin/env bash
# eval_ablations.sh
# -----------------
# Evaluates all 4 ablation checkpoints on the Spatial457 benchmark.
# Each variant runs on a separate GPU in parallel.
#
# Usage:
#   bash eval_ablations.sh
#   DATASET=data/spatial457 bash eval_ablations.sh
#   MAX_SCENES=50 bash eval_ablations.sh          # quick smoke test

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${ROOT}/checkpoints"
RESULTS_DIR="${ROOT}/results"
DATASET="${ROOT}/data/spatial457"
BASELINE="${BASELINE:-qwen2-vl-7b}"
MAX_SCENES="${MAX_SCENES:-}"

mkdir -p "$RESULTS_DIR"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

EVAL="$ROOT/src/step4_evaluation/spatialqa_eval.py"
COMMON="--dataset $DATASET --baselines $BASELINE"
[ -n "$MAX_SCENES" ] && COMMON="$COMMON --max_scenes $MAX_SCENES"

echo "========================================================"
echo "  Evaluating ablation checkpoints on Spatial457"
echo "  Dataset: $DATASET | Baseline: $BASELINE"
echo "  Results: $RESULTS_DIR"
echo "========================================================"

for VARIANT in full no_geom no_epi no_geom_no_epi; do
    CKPT="$CKPT_DIR/best_${VARIANT}.pt"
    [ ! -f "$CKPT" ] && echo "Skipping $VARIANT (checkpoint not found)" && continue
    echo "Evaluating $VARIANT..."
    python "$EVAL" $COMMON \
        --model "$CKPT" \
        --output "$RESULTS_DIR/eval_${VARIANT}.json" \
        > "$RESULTS_DIR/eval_${VARIANT}.log" 2>&1
    echo "  done → $RESULTS_DIR/eval_${VARIANT}.json"
done

echo "All done. Print table: python scripts/print_results.py --results_dir $RESULTS_DIR"
