# Dataset Description

## Primary Dataset: Spatial457

- **Source:** HuggingFace — `RyanWW/Spatial457` (CVPR 2025)
- **Format:** SuperCLEVR synthetic scenes. Each scene has:
  - RGB image (640×480 PNG)
  - Per-object 3D coordinates (`3d_coords`, Blender world units ≈ metres)
  - Per-object pixel coordinates (`pixel_coords`) and bounding boxes (`obj_mask_box`)
  - Object attributes: size, color, shape, material
- **Test split used:** 999 scenes (scenes 20000–20998)
- **GT distances:** Computed analytically from `3d_coords` — exact, no sensor noise.
- **Hallucination labels:** Not available in this split (metrics requiring them are NaN).
- **Preprocessing:** None. Distances computed on-the-fly as `||pos_a - pos_b||₂`.

## Training Dataset: Spatial457-20k

- **Source:** HuggingFace — `RyanWW/Spatial457_20k`
- **Format:** 23,999 individual per-scene JSON files in `scenes/`
- **Train / Val split:** 95% / 5% → 22,799 train / 1,200 val (random, fixed seed)
- **Preprocessing:**
  - Label embeddings computed once using `sentence-transformers/all-MiniLM-L6-v2`
    and cached (~120 unique object labels in superCLEVR)
  - VLM noise augmentation applied at training time (log-normal, σ=0.8)
- **No images needed for GNN training** — distances computed from `3d_coords` directly.

## Scope Reduction Justification

The full Spatial457 paper reports results on a 457-question VQA benchmark with
diverse real-world and synthetic images. We evaluate on the 999-scene superCLEVR
subset (Spatial457-20k compatible format) because:
1. It provides exact GT 3D coordinates for unambiguous distance evaluation.
2. It is the only split that supports the train/test distribution matching
   required by our VLM noise augmentation approach.
3. Compute and time constraints precluded running full VQA evaluation.
