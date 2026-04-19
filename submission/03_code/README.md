# Quant-EpiGNN — Setup & Run Instructions

## Hardware Used
- GPU: NVIDIA RTX 6000 Ada (48 GB VRAM) — training and evaluation
- A GPU with at least 16 GB VRAM is required for evaluation with Qwen2-VL-7B
- The GNN itself is small (~10 MB); inference without the VLM baseline runs on any GPU

## Environment Setup

```bash
conda create -n epignn python=3.10
conda activate epignn
pip install -r requirements.txt
```

All commands below assume:
1. You are inside the `03_code/` directory
2. PYTHONPATH is set: `export PYTHONPATH=$(pwd)/src`

## Dataset

### Spatial457 (evaluation, 999 scenes)
```bash
huggingface-cli download RyanWW/Spatial457 --local-dir data/spatial457 --repo-type dataset
```

### Spatial457-20k (training, 23,999 scenes)
```bash
huggingface-cli download RyanWW/Spatial457_20k --local-dir data/spatial457_20k --repo-type dataset
```

See `../04_data/dataset_links.txt` for direct links.

## Training

Train all 4 ablation variants in parallel across GPUs 0–3:
```bash
export PYTHONPATH=$(pwd)/src
bash scripts/train_20k.sh
# Checkpoints saved to checkpoints/
# Logs saved to logs/
```

Train a single variant manually:
```bash
export PYTHONPATH=$(pwd)/src
python scripts/train.py \
    --dataset spatial457_20k \
    --data_root data/spatial457_20k/scenes \
    --epochs 100 \
    --vlm_noise_sigma 0.8 \
    --checkpoint_dir checkpoints \
    --batch_size 32 \
    --lr 3e-4
# Add --no_geom_constraint and/or --no_epistemic for ablation variants
```

## Evaluation

Evaluate all 4 variants against Qwen2-VL-7B baseline:
```bash
export PYTHONPATH=$(pwd)/src
BASELINE=qwen2-vl-7b bash scripts/eval_20k.sh
# Results saved to results/
```

Evaluate with Step 3 visual feedback loop:
```bash
export PYTHONPATH=$(pwd)/src
python src/step4_evaluation/spatialqa_eval.py \
    --model checkpoints/best_no_geom_no_epi.pt \
    --dataset data/spatial457 \
    --baselines qwen2-vl-7b \
    --output results/eval_feedback.json \
    --feedback
```

Print results table:
```bash
python scripts/print_results.py --results_dir results/
```

## Demo

A pre-trained checkpoint is included at `checkpoints/best_no_geom_no_epi.pt`.
See `../06_demo/demo_instructions.md` for the full live demo walkthrough.

Quick single-scene inference:
```bash
export PYTHONPATH=$(pwd)/src
python scripts/infer.py \
    --image ../06_demo/demo_inputs/sample_image.png \
    --checkpoint checkpoints/best_no_geom_no_epi.pt \
    --out_dir ../06_demo/demo_out/
```

## Project Structure

```
03_code/
  README.md
  requirements.txt
  checkpoints/            Pre-trained GNN checkpoint (for demo)
  src/
    step1_scene_graph/    Scene graph construction (detection, depth, relations)
    step2_epistemic_gnn/  QuantEpiGNN model
    step3_visual_agent/   Visual feedback loop
    step4_evaluation/     Evaluation metrics and baselines
  scripts/
    train.py              Training entry point
    infer.py              Single-scene inference (Steps 1→2→3)
    train_20k.sh          Parallel training script (4 ablation variants)
    eval_20k.sh           Parallel evaluation script
    eval_ablations.sh     Sequential ablation eval
    print_results.py      Results table printer
  configs/
    train_20k.yaml        Training hyperparameters
    eval.yaml             Evaluation settings
```
