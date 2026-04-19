# Demo Instructions (3–5 minute live demo)

## Prerequisites

1. Environment set up (see `03_code/README.md`):
   ```bash
   conda activate epignn
   ```
2. Navigate to the code directory and set PYTHONPATH:
   ```bash
   cd 03_code/
   export PYTHONPATH=$(pwd)/src
   ```
3. A pre-trained checkpoint is already included at `03_code/checkpoints/best_no_geom_no_epi.pt`

---

## Step 1 — Show the problem (30 seconds)

Open `06_demo/demo_inputs/sample_image.png` and point out the objects in the scene.

Explain: Qwen2-VL-7B (a state-of-the-art 7B VLM) averages **6.19m error** on pairwise
3D distances in these synthetic scenes. Our system reduces this to **2.89m** (~53% reduction).

---

## Step 2 — Run the full pipeline (2 minutes)

From inside `03_code/`, run:

```bash
python scripts/infer.py \
    --image ../06_demo/demo_inputs/sample_image.png \
    --checkpoint checkpoints/best_no_geom_no_epi.pt \
    --out_dir ../06_demo/demo_out/
```

This runs Step 1 (scene graph extraction) → Step 2 (GNN correction) → Step 3 (feedback loop).

Outputs written to `06_demo/demo_out/`:
- `scene_graph.json` — detected objects and initial distances
- `gnn_output.json` — GNN-refined distances and per-edge residuals
- `iter_00_annotated.png` — high-residual edges annotated on the image
- `summary.json` — feedback loop stats (iterations, convergence)

Point out `iter_00_annotated.png`: the GNN flagged geometrically inconsistent edges
and re-queried the VLM with bounding box annotations to correct them.

To run without the Step 3 feedback loop (GNN only):
```bash
python scripts/infer.py \
    --image ../06_demo/demo_inputs/sample_image.png \
    --checkpoint checkpoints/best_no_geom_no_epi.pt \
    --out_dir ../06_demo/demo_out_no_feedback/ \
    --skip_step3
```

---

## Step 3 — Show the results table (30 seconds)

```bash
python scripts/print_results.py --results_dir ../05_results/
```

(Pre-computed results are included — no re-evaluation needed.)

This prints the 3-row comparison table:

| Method           | MAE (m) | Reduction |
|------------------|---------|-----------|
| Qwen2-VL-7B      | 6.19    | —         |
| GNN (ours)       | 2.89    | ~53%      |
| GNN + feedback   | 2.89    | ~53%      |

---

## Backup

If live inference fails, show the pre-generated outputs in `06_demo/demo_out/`
or the figures in `05_results/figures/`.
