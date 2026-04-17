#!/usr/bin/env bash
# train_ablations.sh
# ------------------
# Trains all ablation variants of QuantEpiGNN in parallel across GPUs 1-3
# (GPU 0 is available as overflow).  Each variant gets its own log file.
#
# Variants:
#   full          QuantEpiGNN (geom_constraint=T, epistemic=T)  — the full model
#   no_geom       AblationGNN (geom_constraint=F, epistemic=T)
#   no_epi        AblationGNN (geom_constraint=T, epistemic=F)
#   no_geom_no_epi AblationGNN (geom_constraint=F, epistemic=F)  — plain GNN
#
# Usage:
#   bash train_ablations.sh
#   EPOCHS=100 bash train_ablations.sh   # override epoch count

set -euo pipefail

REPO="/home/jay_agarwal_2022/Spatial-Reasoning"
LOG_DIR="${REPO}/logs"
CKPT_DIR="${REPO}/checkpoints"
DATA_ROOT="${REPO}/data/scene_graphs"
CONDA_ENV="epignn"
EPOCHS="${EPOCHS:-100}"
LR="5e-4"
BATCH="16"
HIDDEN="256"
HUBER_DELTA="0.5"    # 0.5 metres — tighter than default for real metric GT

mkdir -p "$LOG_DIR" "$CKPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON="--dataset step1 --data_root $DATA_ROOT \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH \
        --hidden_dim $HIDDEN --huber_delta $HUBER_DELTA \
        --checkpoint_dir $CKPT_DIR"

echo "========================================================"
echo "  Launching ablation training"
echo "  Epochs: $EPOCHS   LR: $LR   Hidden: $HIDDEN"
echo "  Data:   $DATA_ROOT"
echo "  Logs:   $LOG_DIR"
echo "========================================================"

# Variant 1: Full QuantEpiGNN (geom + epistemic)
echo "[$(date)] Starting full QuantEpiGNN on cuda:1"
nohup python "$REPO/train.py" $COMMON \
    --device cuda:1 \
    > "$LOG_DIR/train_full.log" 2>&1 &
PID_FULL=$!
echo "  PID: $PID_FULL  log: $LOG_DIR/train_full.log"

sleep 2   # stagger launches to avoid race on embed model download

# Variant 2: No geometric constraint
echo "[$(date)] Starting no_geom on cuda:2"
nohup python "$REPO/train.py" $COMMON \
    --device cuda:2 \
    --no_geom_constraint \
    > "$LOG_DIR/train_no_geom.log" 2>&1 &
PID_NOGEOM=$!
echo "  PID: $PID_NOGEOM  log: $LOG_DIR/train_no_geom.log"

sleep 2

# Variant 3: No epistemic uncertainty
echo "[$(date)] Starting no_epi on cuda:3"
nohup python "$REPO/train.py" $COMMON \
    --device cuda:3 \
    --no_epistemic \
    > "$LOG_DIR/train_no_epi.log" 2>&1 &
PID_NOEPI=$!
echo "  PID: $PID_NOEPI  log: $LOG_DIR/train_no_epi.log"

sleep 2

# Variant 4: Plain GNN (no geom, no epistemic)
echo "[$(date)] Starting plain GNN (no_geom + no_epi) on cuda:0"
nohup python "$REPO/train.py" $COMMON \
    --device cuda:0 \
    --no_geom_constraint \
    --no_epistemic \
    > "$LOG_DIR/train_plain_gnn.log" 2>&1 &
PID_PLAIN=$!
echo "  PID: $PID_PLAIN  log: $LOG_DIR/train_plain_gnn.log"

echo ""
echo "All 4 training jobs launched.  Monitor with:"
echo "  tail -f $LOG_DIR/train_full.log"
echo "  tail -f $LOG_DIR/train_no_geom.log"
echo "  tail -f $LOG_DIR/train_no_epi.log"
echo "  tail -f $LOG_DIR/train_plain_gnn.log"
echo ""
echo "Or watch all at once:"
echo "  tail -f $LOG_DIR/train_*.log"
echo ""
echo "PIDs: full=$PID_FULL no_geom=$PID_NOGEOM no_epi=$PID_NOEPI plain=$PID_PLAIN"
echo "$(date): all training jobs launched" >> "$LOG_DIR/launch.log"
