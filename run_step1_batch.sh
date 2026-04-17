#!/usr/bin/env bash
# run_step1_batch.sh
# ------------------
# Runs step1_scene_graph/run_pipeline.py on all images in data/spatial457/images/
# using the full (real) backends: owlvit + dpt + blip2 + use_vlm_relations.
#
# If those are too slow, set FAST_MODE=1 to use contour+pseudo+heuristic instead.
#
# Distributes work across up to 4 GPUs (cuda:0..3) in round-robin fashion.
# All 4 jobs run in parallel; the script waits for all to finish.
#
# Usage:
#   bash run_step1_batch.sh             # real backends, all 4 GPUs
#   FAST_MODE=1 bash run_step1_batch.sh # fast backends (for quick smoke test)
#   MAX_IMAGES=50 bash run_step1_batch.sh  # process only first 50 images

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
IMG_DIR="${REPO_DIR}/data/spatial457/images"
JSON_OUT="${REPO_DIR}/data/scene_graphs"
VIZ_OUT="${REPO_DIR}/data/viz"
CONDA_ENV="epignn"
NUM_GPUS=4

FAST_MODE="${FAST_MODE:-0}"
MAX_IMAGES="${MAX_IMAGES:-0}"   # 0 = all

if [ "$FAST_MODE" = "1" ]; then
    DETECTOR="contour"
    DEPTH="pseudo"
    RELATION="heuristic"
    VLM_FLAG=""
    echo "[batch] FAST_MODE=1: using contour + pseudo + heuristic backends"
else
    DETECTOR="owlvit"
    DEPTH="dpt"
    RELATION="blip2"
    VLM_FLAG="--use_vlm_relations"
    echo "[batch] Real backends: owlvit + dpt + blip2 + use_vlm_relations"
fi

mkdir -p "$JSON_OUT" "$VIZ_OUT"

# Collect images that don't already have a JSON output
mapfile -t ALL_IMAGES < <(find "$IMG_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) | sort)
echo "[batch] Found ${#ALL_IMAGES[@]} total images in $IMG_DIR"

IMAGES=()
for img_path in "${ALL_IMAGES[@]}"; do
    stem="$(basename "$img_path")"
    stem="${stem%.*}"
    json_path="${JSON_OUT}/${stem}.json"
    if [ ! -f "$json_path" ]; then
        IMAGES+=("$img_path")
    fi
done
echo "[batch] ${#IMAGES[@]} images need processing (already done: $((${#ALL_IMAGES[@]} - ${#IMAGES[@]})))"

if [ "${MAX_IMAGES}" -gt 0 ] && [ "${#IMAGES[@]}" -gt "${MAX_IMAGES}" ]; then
    IMAGES=("${IMAGES[@]:0:${MAX_IMAGES}}")
    echo "[batch] Capped to first ${MAX_IMAGES} images"
fi

if [ "${#IMAGES[@]}" -eq 0 ]; then
    echo "[batch] Nothing to do."
    exit 0
fi

# Split images across GPUs
declare -a GPU_LISTS
for (( g=0; g<NUM_GPUS; g++ )); do
    GPU_LISTS[$g]=""
done
idx=0
for img_path in "${IMAGES[@]}"; do
    gpu_id=$(( idx % NUM_GPUS ))
    GPU_LISTS[$gpu_id]+=" $img_path"
    (( idx++ ))
done

# Source conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

worker() {
    local gpu_id="$1"
    shift
    local images=("$@")
    local n="${#images[@]}"
    local done_count=0
    echo "[GPU $gpu_id] Processing $n images"
    for img_path in "${images[@]}"; do
        stem="$(basename "$img_path")"
        stem="${stem%.*}"
        json_path="${JSON_OUT}/${stem}.json"
        viz_path="${VIZ_OUT}/${stem}.jpg"
        python "${REPO_DIR}/step1_scene_graph/run_pipeline.py" \
            --image_path       "$img_path" \
            --json_output_path "$json_path" \
            --viz_output_path  "$viz_path" \
            --detector_backend "$DETECTOR" \
            --depth_backend    "$DEPTH" \
            --relation_backend "$RELATION" \
            $VLM_FLAG \
            --device "cuda:${gpu_id}" 2>/dev/null
        (( done_count++ ))
        echo "[GPU $gpu_id] $done_count/$n done: $stem"
    done
    echo "[GPU $gpu_id] Finished all $n images."
}

export -f worker
export JSON_OUT VIZ_OUT DETECTOR DEPTH RELATION VLM_FLAG REPO_DIR

# Launch one worker per GPU in parallel
PIDS=()
for (( g=0; g<NUM_GPUS; g++ )); do
    read -ra GPU_IMG_LIST <<< "${GPU_LISTS[$g]}"
    if [ "${#GPU_IMG_LIST[@]}" -gt 0 ]; then
        worker "$g" "${GPU_IMG_LIST[@]}" &
        PIDS+=($!)
    fi
done

# Wait for all workers
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

if [ "$FAIL" -eq 0 ]; then
    echo "[batch] All done. JSONs in: $JSON_OUT"
    echo "[batch] Visualizations in: $VIZ_OUT"
    json_count=$(find "$JSON_OUT" -name '*.json' | wc -l)
    echo "[batch] Total JSON files: $json_count"
else
    echo "[batch] Some workers failed — check logs above."
    exit 1
fi
