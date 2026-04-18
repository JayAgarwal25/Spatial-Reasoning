# Quant-EpiGNN — Evaluation Results

All results are on the **Spatial457** benchmark (999-scene test split, superCLEVR format).
VLM baseline: **Qwen2-VL-7B-Instruct** (local, no rate limits).
GNN trained on **Spatial457-20k** with log-normal VLM noise augmentation (σ=0.8).

---

## Main Result: GNN vs VLM Baseline

| Method | MAE (m) ↓ | MRA ↑ | Tri-viol Rate ↓ | Mean Residual ↓ |
|---|---|---|---|---|
| Qwen2-VL-7B (baseline) | 6.1924 | — | 0.0530 | 3.8962 |
| GNN only (plain) | 2.8885 | **0.3726** | 0.0530 | 3.8962 |
| GNN + feedback loop | **2.8868** | **0.3726** | 0.0530 | 3.8962 |

**GNN reduces MAE by ~53% over Qwen2-VL-7B.**
**Feedback loop gives marginal additional improvement** (2.8885 → 2.8868, ~0.06%).

**Why the feedback loop improvement is small:** The GNN already corrects most VLM error.
Remaining residuals reflect GNN limitations, not VLM noise — so re-querying Qwen2-VL
with annotated images provides little new signal. On synthetic superCLEVR scenes
(visually unambiguous objects), annotations don't change what the VLM can infer.
This is reported honestly in the paper's Discussion section.

---

## Ablation Study (all vs Qwen2-VL-7B baseline, 999 scenes)

| Variant | Description | Best Epoch | Val Loss | MAE (m) ↓ | MRA ↑ | Tri-viol Rate ↓ |
|---|---|---|---|---|---|---|
| QuantEpiGNN (full) | geom constraint + epistemic σ | 60 | 0.0384 | 3.4048 | 0.3336 | 0.0530 |
| No geom constraint | epistemic σ only | 77 | 0.0386 | 3.2050 | 0.3197 | 0.0530 |
| No epistemic σ | geom constraint only | 68 | 0.0385 | 2.8885 | 0.3726 | 0.0530 |
| Plain GNN | neither component | 68 | 0.0382 | **2.8885** | **0.3726** | 0.0530 |

**Key finding:** Adding geom constraint and epistemic σ on top of plain GNN hurts eval MAE
(3.40 vs 2.89). Likely because log-normal noise augmentation already accounts for
uncertainty implicitly — explicit epistemic layers add gradient noise on this synthetic dataset.
Plain GNN is the best-performing checkpoint.

Baseline MAE (Qwen2-VL) is identical across all rows (6.1924) — it's the same 999-scene VLM
evaluation, used as a reference per variant.

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | Spatial457-20k (23,999 training scenes, superCLEVR format) |
| Train / Val split | 95% / 5% (22,799 / 1,200 scenes) |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Epochs | 100 (early-stop on val loss) |
| GNN hidden dim | 256 |
| VLM noise sigma (σ) | 0.8 (log-normal; 80% moderate, 20% severe) |
| Loss | CE (relation class) + λ·Huber (distance), λ=1.0 |

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB VRAM) |
| CPU cores | 128 |
| RAM | 503 GB |
| CUDA | default conda env (epignn) |

---

## Eval Commands

```bash
# Reproduce ablation results (Qwen2-VL baseline, no feedback):
BASELINE=qwen2-vl-7b bash eval_20k.sh

# Reproduce feedback loop result (plain GNN + Step 3):
PYTHONPATH=. python step4_evaluation/spatialqa_eval.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --dataset data/spatial457 \
    --baselines qwen2-vl-7b \
    --output results_20k/eval_qwen_feedback_plain.json \
    --feedback

# Print results table:
python print_results.py --results_dir results_20k/
```

---

## Checkpoint Files (`checkpoints_20k/`)

| File | Variant | Epoch | Val Loss |
|---|---|---|---|
| `best_full.pt` | QuantEpiGNN (full) | 60 | 0.0384 |
| `best_no_geom.pt` | No geom constraint | 77 | 0.0386 |
| `best_no_epi.pt` | No epistemic σ | 68 | 0.0385 |
| `best_no_geom_no_epi.pt` | Plain GNN | 68 | 0.0382 |

---

## Metrics Not Yet Computed

- `Resid-Hall Pearson r` — requires hallucination labels (not in Spatial457 test set)
- `Resid-Hall AUROC` — same
- `Trigger F1` — same
- LLaVA-NeXT baseline — deprioritised (InternVL2 incompatible with transformers 5.x; Qwen2-VL used instead)
