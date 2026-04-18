#!/usr/bin/env bash
# eval_20k.sh
# -----------
# Evaluates all 4 ablation checkpoints (trained on 20k) against a VLM baseline.
#
# Usage:
#   bash eval_20k.sh                              # InternVL2-8B (default)
#   BASELINE=qwen2-vl-7b bash eval_20k.sh         # Qwen2-VL-7B
#   BASELINE=llava-next-7b bash eval_20k.sh        # LLaVA-NeXT

set -euo pipefail

REPO="/home/jay_agarwal_2022/Spatial-Reasoning"
CKPT_DIR="${CKPT_DIR:-${REPO}/checkpoints_20k}"
RESULTS_DIR="${REPO}/results_20k"
DATASET="${REPO}/data/spatial457"

BASELINE="${BASELINE:-internvl2-8b}"

mkdir -p "$RESULTS_DIR"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "======================================================"
echo "  Evaluating 20k checkpoints — baseline: $BASELINE"
echo "  Checkpoints: $CKPT_DIR"
echo "  Results:     $RESULTS_DIR"
echo "======================================================"

# Full model on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup conda run -n epignn --no-capture-output \
    python "$REPO/step4_evaluation/spatialqa_eval.py" \
    --model    "$CKPT_DIR/best_full.pt" \
    --dataset  "$DATASET" \
    --baselines "$BASELINE" \
    --output   "$RESULTS_DIR/eval_${BASELINE}_full.json" \
    > "$RESULTS_DIR/eval_${BASELINE}_full.log" 2>&1 &
echo "full     → GPU 1  (PID $!)"

# no_geom on GPU 2
CUDA_VISIBLE_DEVICES=2 nohup conda run -n epignn --no-capture-output \
    python "$REPO/step4_evaluation/spatialqa_eval.py" \
    --model    "$CKPT_DIR/best_no_geom.pt" \
    --dataset  "$DATASET" \
    --baselines "$BASELINE" \
    --output   "$RESULTS_DIR/eval_${BASELINE}_no_geom.json" \
    > "$RESULTS_DIR/eval_${BASELINE}_no_geom.log" 2>&1 &
echo "no_geom  → GPU 2  (PID $!)"

# no_epi on GPU 3
CUDA_VISIBLE_DEVICES=3 nohup conda run -n epignn --no-capture-output \
    python "$REPO/step4_evaluation/spatialqa_eval.py" \
    --model    "$CKPT_DIR/best_no_epi.pt" \
    --dataset  "$DATASET" \
    --baselines "$BASELINE" \
    --output   "$RESULTS_DIR/eval_${BASELINE}_no_epi.json" \
    > "$RESULTS_DIR/eval_${BASELINE}_no_epi.log" 2>&1 &
echo "no_epi   → GPU 3  (PID $!)"

# plain on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup conda run -n epignn --no-capture-output \
    python "$REPO/step4_evaluation/spatialqa_eval.py" \
    --model    "$CKPT_DIR/best_no_geom_no_epi.pt" \
    --dataset  "$DATASET" \
    --baselines "$BASELINE" \
    --output   "$RESULTS_DIR/eval_${BASELINE}_plain.json" \
    > "$RESULTS_DIR/eval_${BASELINE}_plain.log" 2>&1 &
echo "plain    → GPU 0  (PID $!)"

echo ""
echo "Monitor: tail -f $RESULTS_DIR/eval_${BASELINE}_*.log"
echo "Results: python print_results.py --results_dir $RESULTS_DIR"
