# Quant-EpiGNN — Setup & Run Instructions

## Hardware Used
- GPU: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- CPU: 128 cores
- RAM: 503 GB
- OS: Linux (Ubuntu)

## Environment Setup

```bash
conda create -n epignn python=3.10
conda activate epignn
pip install -r requirements.txt
```

Or restore from the provided environment file:
```bash
conda env create -f environment.yml
conda activate epignn
```

Set PYTHONPATH from the repo root before any command:
```bash
export PYTHONPATH=/path/to/Spatial-Reasoning
```

## Dataset

### Spatial457 (evaluation, 999 scenes)
```bash
# Download from HuggingFace
huggingface-cli download RyanWW/Spatial457 --local-dir data/spatial457
```

### Spatial457-20k (training, 23,999 scenes)
```bash
huggingface-cli download RyanWW/Spatial457_20k --local-dir data/spatial457_20k
```

See `04_data/dataset_links.txt` for direct links.

## Training

Train all 4 ablation variants in parallel (GPUs 0–3):
```bash
bash scripts/train_20k.sh
# Checkpoints saved to checkpoints_20k/
# Logs saved to logs_20k/
```

Train a single variant manually:
```bash
python src/train.py \
    --dataset spatial457_20k \
    --data_root data/spatial457_20k/scenes \
    --epochs 100 \
    --vlm_noise_sigma 0.8 \
    --checkpoint_dir checkpoints_20k \
    --batch_size 32 \
    --lr 3e-4
# Add --no_geom_constraint and/or --no_epistemic for ablation variants
```

## Evaluation

Evaluate all 4 variants vs Qwen2-VL-7B baseline:
```bash
BASELINE=qwen2-vl-7b bash scripts/eval_20k.sh
# Results saved to results_20k/
```

Evaluate with Step 3 visual feedback loop:
```bash
python src/step4_evaluation/spatialqa_eval.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --dataset data/spatial457 \
    --baselines qwen2-vl-7b \
    --output results_20k/eval_qwen_feedback_plain.json \
    --feedback
```

Print results table:
```bash
python scripts/print_results.py --results_dir results_20k/
```

## Demo

See `06_demo/demo_instructions.md` for the live demo walkthrough.

Quick single-scene inference:
```bash
python src/infer.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --scene 06_demo/demo_inputs/sample_scene.json \
    --image 06_demo/demo_inputs/sample_image.png
```

## Project Structure

```
src/
  train.py                        Training entry point
  infer.py                        Single-scene inference
  step1_scene_graph/              Scene graph construction
  step2_epistemic_gnn/            QuantEpiGNN model
  step3_visual_agent/             Visual feedback loop
  step4_evaluation/               Evaluation metrics and baselines
scripts/
  train_20k.sh                    Parallel training script
  eval_20k.sh                     Parallel evaluation script
  print_results.py                Results table printer
configs/
  train_20k.yaml                  Training hyperparameters
  eval.yaml                       Evaluation settings
```
