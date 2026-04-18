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

REPO="/home/jay_agarwal_2022/Spatial-Reasoning"
CKPT_DIR="${REPO}/checkpoints"
RESULTS_DIR="${REPO}/results"
DATA_ROOT="${REPO}/data/spatial457"
CONDA_ENV="epignn"

DATASET="${DATASET:-$DATA_ROOT}"
MAX_SCENES="${MAX_SCENES:-}"
EPSILON="${EPSILON:-0.3}"
BASELINES="${BASELINES:-mock}"

mkdir -p "$RESULTS_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

COMMON="--dataset $DATASET --baselines $BASELINES --epsilon $EPSILON"
if [ -n "$MAX_SCENES" ]; then
    COMMON="$COMMON --max_scenes $MAX_SCENES"
fi

echo "========================================================"
echo "  Evaluating ablation checkpoints on Spatial457"
echo "  Dataset: $DATASET"
echo "  Baselines: $BASELINES"
echo "  Epsilon: $EPSILON"
echo "  Results: $RESULTS_DIR"
echo "========================================================"

# Variant 1: full
echo "[$(date)] Evaluating full QuantEpiGNN on cuda:1"
nohup python "$REPO/step4_evaluation/spatialqa_eval.py" $COMMON \
    --model   "$CKPT_DIR/best_full.pt" \
    --output  "$RESULTS_DIR/eval_full.json" \
    > "$RESULTS_DIR/eval_full.log" 2>&1 &
PID_FULL=$!

sleep 1

# Variant 2: no_geom
echo "[$(date)] Evaluating no_geom on cuda:2"
nohup env CUDA_VISIBLE_DEVICES=2 python "$REPO/step4_evaluation/spatialqa_eval.py" $COMMON \
    --model   "$CKPT_DIR/best_no_geom.pt" \
    --output  "$RESULTS_DIR/eval_no_geom.json" \
    > "$RESULTS_DIR/eval_no_geom.log" 2>&1 &
PID_NOGEOM=$!

sleep 1

# Variant 3: no_epi
echo "[$(date)] Evaluating no_epi on cuda:3"
nohup env CUDA_VISIBLE_DEVICES=3 python "$REPO/step4_evaluation/spatialqa_eval.py" $COMMON \
    --model   "$CKPT_DIR/best_no_epi.pt" \
    --output  "$RESULTS_DIR/eval_no_epi.json" \
    > "$RESULTS_DIR/eval_no_epi.log" 2>&1 &
PID_NOEPI=$!

sleep 1

# Variant 4: plain GNN
echo "[$(date)] Evaluating plain GNN on cuda:0"
nohup env CUDA_VISIBLE_DEVICES=0 python "$REPO/step4_evaluation/spatialqa_eval.py" $COMMON \
    --model   "$CKPT_DIR/best_no_geom_no_epi.pt" \
    --output  "$RESULTS_DIR/eval_plain_gnn.json" \
    > "$RESULTS_DIR/eval_plain_gnn.log" 2>&1 &
PID_PLAIN=$!

echo ""
echo "All 4 eval jobs launched. PIDs: full=$PID_FULL no_geom=$PID_NOGEOM no_epi=$PID_NOEPI plain=$PID_PLAIN"
echo "Monitor: tail -f $RESULTS_DIR/eval_*.log"
echo ""
echo "When done, run: python print_results.py"
