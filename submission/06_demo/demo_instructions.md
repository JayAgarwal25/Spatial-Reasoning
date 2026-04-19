# Demo Instructions (3–5 minute live demo)

## Prerequisites
- Environment activated: `conda activate epignn`
- Run all commands from `submission/03_code/`
- PYTHONPATH set: `export PYTHONPATH=$(pwd)/src`
- Checkpoint present: `checkpoints_20k/best_no_geom_no_epi.pt` (in repo root)

---

## Step 1 — Show the problem (30 seconds)
Open `../06_demo/demo_inputs/sample_image.png`. Point out the objects.
Explain: Qwen2-VL-7B averages ~6.19m error on pairwise 3D distances.

---

## Step 2 — Show GNN correction + feedback loop (2 minutes)
Run the full pipeline (Step 1 → GNN → Step 3 feedback loop):

```bash
cd /path/to/Spatial-Reasoning/submission/03_code
export PYTHONPATH=$(pwd)/src

python src/infer.py \
    --image ../06_demo/demo_inputs/sample_image.png \
    --checkpoint ../../../checkpoints_20k/best_no_geom_no_epi.pt \
    --out_dir ../06_demo/demo_out/
```

Outputs written to `../06_demo/demo_out/`:
- `scene_graph.json` — detected objects + initial distances
- `gnn_output.json` — GNN-refined distances + per-edge residuals
- `iter_00_annotated.png` — high-residual edges annotated on the image
- `summary.json` — loop stats

Point out: GNN MAE ~2.89m vs VLM MAE ~6.19m (~53% reduction).

To skip the Step 3 feedback loop:
```bash
python src/infer.py \
    --image ../06_demo/demo_inputs/sample_image.png \
    --checkpoint ../../../checkpoints_20k/best_no_geom_no_epi.pt \
    --out_dir ../06_demo/demo_out_no_feedback/ \
    --skip_step3
```

---

## Step 3 — Show full eval results (30 seconds)
```bash
python scripts/print_results.py --results_dir ../../../results_20k/
```

This prints the 3-row comparison table (VLM baseline → GNN → GNN+feedback).

---

## Talking Points
- Why we retrained on 20k: train/test distribution mismatch (GT distances vs VLM predictions)
- Why plain GNN beats full model: noise augmentation implicitly handles uncertainty
- Why feedback loop improvement is small: GNN already corrects most error; synthetic scenes
  give VLM little new information from annotations
- Honest limitation: hallucination metrics not computed (no labels in test split)

---

## Backup
If live inference fails, show `backup_video.mp4` or the screenshots in `05_results/figures/`.
