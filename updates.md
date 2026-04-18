# Project Updates — Quant-EpiGNN

Chronological log of all major code changes, design decisions, and experiment runs.

---

## Phase 1 — Initial Pipeline (pre-session)

Core pipeline implemented across 4 steps:

- **step1_scene_graph/**: Object detection (OWL-ViT / GroundingDINO / Florence-2), monocular depth (DPT), geometric pair pruning, pairwise relation extraction.
- **step2_epistemic_gnn/**: `QuantEpiGNN` — epistemic node states (mean μ + uncertainty σ), geometric consistency residual computation (Eq. 1 in paper), residual-weighted message passing, dual-stream loss (CE + Huber).
- **step3_visual_agent/**: Trigger (`trigger.py`) + annotation actions (`actions.py`, draw_bbox / draw_line / depth_crop) + feedback loop orchestrator (`feedback_loop.py`). **Fully implemented but not connected to eval.**
- **step4_evaluation/**: `spatialqa_eval.py` — Spatial457 loader, GNN inference, MAE/MRA/triangle-violation metrics, baseline VLM comparison.

---

## Phase 2 — Fixing the Train/Test Distribution Mismatch

**Problem discovered:** GNN `edge_dist` input during training = GT 3D distances (from `3d_coords`).
At eval time, `edge_dist` = VLM predictions (e.g. LLaVA: ~0.2–0.8 m vs GT ~2–10 m — 10× scale mismatch).
The GNN had never seen noisy VLM-scale inputs during training, making eval results meaningless.

**Fix (`train.py`):**
- Added `_apply_vlm_noise(data, sigma)`: multiplicative log-normal noise on `edge_dist`.
  Formula: `edge_dist = GT_dist × exp(N(0, σ))`, with 80/20 mixture of σ and 2σ to simulate
  moderate and severe VLM errors respectively.
- Added `--vlm_noise_sigma` arg (default `0.8`), covering factor-2 to factor-10 VLM errors.
- Added `load_spatial457_20k()`: loads 23,999 per-scene JSON files from Spatial457-20k
  (HuggingFace: `RyanWW/Spatial457_20k`), with label embedding caching (~120 unique labels).
- Added `_scene_json_to_pyg()`: converts per-scene JSON → PyG `Data` using correct attribute
  names (`node_sem`, `node_bbox`, `node_depth`) matching `AblationGNN` / `QuantEpiGNN`.
- Added `"spatial457_20k"` branch to `get_dataset()`.

**Key fix detail:** `Data(x=...)` → `Data(node_sem=..., node_bbox=..., node_depth=...)`.
GNN accesses `data.node_sem` directly; `x` caused `AttributeError: GlobalStorage has no attribute 'node_sem'`.

---

## Phase 3 — Switching from Gemini to Local VLM Baselines

**Problem:** Gemini Pro 1.5 API hit 5 RPM rate limits, then 503 errors, halting eval.

**Decision:** Switch to local open-source VLMs matching Spatial457 paper's comparison table.

**Changes to `step4_evaluation/baseline.py`:**

- **`LLaVANextBaseline`** (rewritten): `llava-hf/llava-v1.6-vicuna-7b-hf`, batched distance
  prediction in one forward pass, `max_new_tokens=512`, correct token slice for generation-only decode.
- **`Qwen2VLBaseline`** (new): `Qwen/Qwen2-VL-7B-Instruct`, transformers 5.x compatible,
  `apply_chat_template` + image content dict format, batch decode.
- **`InternVL2Baseline`** attempted but abandoned: incompatible with transformers 5.5.4
  (`_tied_weights_keys` / `all_tied_weights_keys` API change, `language_model` module access).
  Qwen2-VL used as replacement (equivalent 8B-class quality).
- **`GeminiBaseline`** patched: partial array handling (NaN-pad if response shorter than pairs),
  debug logging, 13s inter-scene sleep. Kept in codebase but not used for final results.
- Registry: `"qwen2-vl-7b": Qwen2VLBaseline`, `"internvl2-8b": InternVL2Baseline` (disabled).

---

## Phase 4 — Training on Spatial457-20k

**Dataset:** `RyanWW/Spatial457_20k` — 23,999 individual per-scene JSONs, same superCLEVR
format as original Spatial457 (objects with `3d_coords`, `mask_box`, `pixel_coords`).

**4 ablation variants trained in parallel across GPUs 0–3 (`train_20k.sh`):**

| Variant | Flag | GPU | Best Epoch | Val Loss |
|---|---|---|---|---|
| Full (geom + epistemic) | — | 1 | 60 | 0.0384 |
| No geom constraint | `--no_geom_constraint` | 2 | 77 | 0.0386 |
| No epistemic σ | `--no_epistemic` | 3 | 68 | 0.0385 |
| Plain GNN | both flags | 0 | 68 | 0.0382 |

**Training issues encountered:**
- `full` job SIGTERM'd at epoch 71 (external kill); best checkpoint was epoch 60 — used as-is.
- `no_epi` OOM on GPU 3 (13.6 GB free, two jobs already running); moved to GPU 1 (48 GB, two jobs fit at ~14 GB each).
- `no_epi` stdout empty initially (disk I/O contention + Python buffering); relaunched with `PYTHONUNBUFFERED=1`.
- All 4 checkpoints in `checkpoints_20k/`.

---

## Phase 5 — Evaluation (Qwen2-VL Baseline)

**`eval_20k.sh`** — runs 4 eval jobs in parallel (GPUs 0–3):

```bash
BASELINE=qwen2-vl-7b bash eval_20k.sh
```

Results in `results_20k/eval_qwen_{variant}.json`. All 4 completed (~33 min at ~2 s/scene).

**`print_results.py`** updated to glob `eval_*_{variant}.json` patterns (previously only matched
`eval_{variant}.json`), so it now handles both naming conventions.

---

## Phase 6 — Step 3 Feedback Loop Integration

**Problem:** Step 3 (`step3_visual_agent/`) was fully implemented but never connected to eval.
The paper claims a "trigger-based visual grounding loop" as a core contribution, but no
quantitative results existed for it.

**Root cause:** `spatialqa_eval.py` used pre-built Spatial457 scene graphs directly, bypassing
Step 1 (detection + depth) and skipping Step 3 entirely.

**Changes to `step4_evaluation/spatialqa_eval.py`:**

1. **`_load_super_clevr_format`**: now also extracts `pixel_center` `[cx, cy]` (from
   `pixel_coords[0][:2]`) and `bbox` `[x, y, w, h]` (from `obj_mask_box[str(i)]['obj'][0]`)
   per object — needed for annotation drawing.

2. **`run_step3_feedback()`** (new function): visual grounding feedback loop for eval.
   - Identifies high-residual edges (residual > ε, default ε=0.3).
   - Builds `ObjectNode` + `RelationEdge` instances from Spatial457 pixel data.
   - Calls `draw_bbox` + `draw_line` from `step3_visual_agent/actions.py` to annotate image.
   - Saves annotated image to temp file, re-queries Qwen2-VL for **flagged pairs only**.
   - Updates `edge_dist` in PyG Data, re-runs GNN. Repeats up to `max_iters=2`.

3. **`evaluate_scene`**: added `run_feedback: bool = False` parameter; calls
   `run_step3_feedback` when enabled; adds `pred_gnn_feedback` to per-scene result dict.

4. **`aggregate_results`**: computes `feedback_mae` alongside `gnn_mae` and `baseline_mae`.

5. **`main()`**: added `--feedback` flag.

**Feedback eval run (in progress):**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python step4_evaluation/spatialqa_eval.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --dataset data/spatial457 --baselines qwen2-vl-7b \
    --output results_20k/eval_qwen_feedback_plain.json --feedback
# Monitor: tail -f results_20k/eval_qwen_feedback_plain.log
```

Smoke test (3 scenes): `gnn_mae=2.7698`, `feedback_mae=2.7679` — confirmed loop runs correctly.

---

## Project Direction Summary

**What this system does:**
1. Takes an image → builds a scene graph (Step 1)
2. Refines pairwise distances using a geometry-aware GNN with epistemic uncertainty (Step 2)
3. Flags geometrically inconsistent edges (residual > ε) and re-queries the VLM with
   annotated images to correct them (Step 3)
4. Evaluates against GT 3D distances (Step 4)

**What we actually measured:**
- GNN (Step 2) vs Qwen2-VL raw predictions on 999 Spatial457 test scenes.
- GNN achieves **~53% MAE reduction** (2.89 m vs 6.19 m).
- Feedback loop (Step 3) connected and running — results pending.

**Honest scope:**
- Hallucination detection metrics (Resid-Hall Pearson r, AUROC, Trigger F1) are NaN —
  Spatial457 test set has no hallucination labels. These are not computed.
- InternVL2 dropped due to transformers 5.x incompatibility; Qwen2-VL used instead.
- Full Step 1 pipeline (detection + depth) not re-run at eval time — Spatial457 GT scene
  graphs used directly for GNN eval. Step 3 re-queries VLM but keeps GT object positions.
- This is a course project (25% weightage). Results are real and honest.

---

## Key Files Added / Modified

| File | Change |
|---|---|
| `train.py` | VLM noise augmentation, Spatial457-20k loader, fixed PyG attribute names |
| `train_20k.sh` | New — parallel training script for 4 ablation variants |
| `eval_20k.sh` | New — parallel eval script with BASELINE env var |
| `print_results.py` | New — results table printer, handles multiple filename patterns |
| `step4_evaluation/baseline.py` | Qwen2VLBaseline added, LLaVA rewritten, Gemini patched |
| `step4_evaluation/spatialqa_eval.py` | pixel/bbox extraction, `run_step3_feedback()`, `--feedback` flag |
| `results.md` | This project's results summary |
| `updates.md` | This file |
