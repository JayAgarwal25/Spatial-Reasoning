# Demo Instructions (3–5 minute live demo)

## Prerequisites
- Environment activated: `conda activate epignn`
- PYTHONPATH set: `export PYTHONPATH=/path/to/Spatial-Reasoning`
- Checkpoint present: `checkpoints_20k/best_no_geom_no_epi.pt`
- Demo inputs in: `06_demo/demo_inputs/`

---

## Step 1 — Show the problem (30 seconds)
Open a sample scene image from `demo_inputs/`. Point out the objects.
Show what Qwen2-VL-7B predicts for pairwise distances (wrong, ~6m avg error):

```bash
python src/infer.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --scene 06_demo/demo_inputs/sample_scene.json \
    --image 06_demo/demo_inputs/sample_image.png \
    --show_baseline
```

This prints: GT distances, Qwen2-VL predictions, GNN-refined predictions.

---

## Step 2 — Show GNN correction (1 minute)
The same command above also runs the GNN and prints refined distances.
Point out: GNN MAE ~2.89m vs VLM MAE ~6.19m on this scene.

---

## Step 3 — Show feedback loop (1–2 minutes)
Run the feedback loop on the demo scene:

```bash
python src/infer.py \
    --model checkpoints_20k/best_no_geom_no_epi.pt \
    --scene 06_demo/demo_inputs/sample_scene.json \
    --image 06_demo/demo_inputs/sample_image.png \
    --feedback \
    --save_annotated demo_inputs/annotated_output.png
```

Open `demo_inputs/annotated_output.png` — shows bounding boxes and distance
annotations drawn on the image for high-residual edges.

---

## Step 4 — Show full eval results (30 seconds)
```bash
python scripts/print_results.py --results_dir results_20k/
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
