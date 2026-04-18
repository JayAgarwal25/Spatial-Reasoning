#!/usr/bin/env bash
# train_20k.sh
# -----------
# Retrains all 4 ablation variants on Spatial457-20k with VLM noise augmentation.
# Runs 4 variants in parallel across GPUs 0-3 (GNN is small, ~10 GB each).
#
# Usage:
#   bash train_20k.sh
#   EPOCHS=100 bash train_20k.sh

set -euo pipefail

REPO="/home/jay_agarwal_2022/Spatial-Reasoning"
DATA="${REPO}/data/spatial457_20k/scenes"
CKPT="${REPO}/checkpoints_20k"
LOGS="${REPO}/logs_20k"

EPOCHS="${EPOCHS:-100}"
NOISE="${NOISE:-0.8}"    # log-normal sigma: covers factor-2 to factor-10 errors

mkdir -p "$CKPT" "$LOGS"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "======================================================"
echo "  Training on Spatial457-20k  (${EPOCHS} epochs)"
echo "  VLM noise sigma: ${NOISE}"
echo "  Checkpoints:     ${CKPT}"
echo "======================================================"

COMMON="--dataset spatial457_20k --data_root $DATA \
        --epochs $EPOCHS --vlm_noise_sigma $NOISE \
        --checkpoint_dir $CKPT --batch_size 32 --lr 3e-4"

# Full model (geom + epistemic)
CUDA_VISIBLE_DEVICES=1 nohup conda run -n epignn --no-capture-output \
    python "$REPO/train.py" $COMMON \
    > "$LOGS/train_full.log" 2>&1 &
echo "full        → GPU 1  (PID $!)"

# No-geom constraint
CUDA_VISIBLE_DEVICES=2 nohup conda run -n epignn --no-capture-output \
    python "$REPO/train.py" $COMMON --no_geom_constraint \
    > "$LOGS/train_no_geom.log" 2>&1 &
echo "no_geom     → GPU 2  (PID $!)"

# No-epistemic uncertainty
CUDA_VISIBLE_DEVICES=3 nohup conda run -n epignn --no-capture-output \
    python "$REPO/train.py" $COMMON --no_epistemic \
    > "$LOGS/train_no_epi.log" 2>&1 &
echo "no_epi      → GPU 3  (PID $!)"

# Plain GNN (no geom, no epistemic)
CUDA_VISIBLE_DEVICES=0 nohup conda run -n epignn --no-capture-output \
    python "$REPO/train.py" $COMMON --no_geom_constraint --no_epistemic \
    > "$LOGS/train_plain.log" 2>&1 &
echo "plain_gnn   → GPU 0  (PID $!)"

echo ""
echo "All 4 training jobs launched."
echo "Monitor: tail -f $LOGS/train_*.log"
echo ""
echo "After training, run eval with:"
echo "  CKPT_DIR=$CKPT bash eval_20k.sh"
