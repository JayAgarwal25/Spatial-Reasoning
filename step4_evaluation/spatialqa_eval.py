"""
spatialqa_eval.py
-----------------
Evaluation on the Spatial457 benchmark (CVPR 2025).

Dataset:  HuggingFace  RyanWW/Spatial457
GT type:  Exact 3D object coordinates (synthetic) → distances are computed
          analytically, no sensor noise.  This makes every triangle-inequality
          residual spike unambiguously attributable to VLM inconsistency,
          not depth noise.

Pipeline per scene
------------------
1. Load scene metadata (object positions + image).
2. For each ordered pair (A, B) of objects, query every baseline VLM for
   a predicted metric distance.
3. Feed the same pairs through the trained EpiGNN to get refined distances
   and per-edge residuals.
4. Compute MAE, MRA, triangle-violation rate, and (if hallucination labels
   are provided) residual–hallucination correlation.
5. Dump results to JSON.

CLI
---
python step4_evaluation/spatialqa_eval.py \
    --model   checkpoints/best.pt \
    --dataset data/spatial457 \
    --baselines gpt4o mock \
    --output  results/spatial457.json \
    --epsilon 0.3

The --dataset flag accepts either a local directory produced by
    huggingface-cli download RyanWW/Spatial457
or a HuggingFace dataset identifier (auto-downloaded if datasets is installed).
"""

import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from step4_evaluation.metrics import (
    EvalResult,
    compute_mae,
    compute_mra,
    compute_residuals_from_distances,
    residual_hallucination_correlation,
    triangle_violation_rate,
    trigger_precision_recall_f1,
    calibrate_threshold,
    print_result,
)
from step4_evaluation.baseline import build_baseline, VLMBaseline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_spatial457(dataset_path: str) -> List[Dict]:
    """
    Load Spatial457 scenes.

    Expected directory layout (after HF download):
        <dataset_path>/
            data/
                scene_0001/
                    image.jpg
                    metadata.json      ← object positions (x,y,z in metres)
                scene_0002/
                    ...

    metadata.json schema:
    {
        "objects": [
            {"id": "chair",   "position": [x, y, z]},
            {"id": "table",   "position": [x, y, z]},
            ...
        ],
        "hallucination_labels": {          # optional — only if GT labels exist
            "chair__table": 0,             # 0 = correct VLM claim, 1 = hallucinated
            ...
        }
    }

    Returns list of scene dicts with keys:
        scene_id, image_path, objects, gt_distances, hallucination_labels
    """
    dataset_path = Path(dataset_path)
    scenes = []

    # Try HuggingFace datasets library first
    if not dataset_path.exists():
        try:
            from datasets import load_dataset
            logger.info("Downloading RyanWW/Spatial457 from HuggingFace …")
            hf_ds = load_dataset("RyanWW/Spatial457", split="test")
            return _convert_hf_spatial457(hf_ds)
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset path '{dataset_path}' not found and HF download "
                f"failed: {e}"
            )

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path          # flat layout fallback

    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "metadata.json"
        img_path  = scene_dir / "image.jpg"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        objects = meta.get("objects", [])
        if len(objects) < 2:
            continue

        # Compute ground-truth pairwise distances from 3D coordinates
        gt_distances: Dict[str, float] = {}
        for obj_a, obj_b in combinations(objects, 2):
            pos_a = np.array(obj_a["position"], dtype=float)
            pos_b = np.array(obj_b["position"], dtype=float)
            dist_m = float(np.linalg.norm(pos_a - pos_b))
            key = f"{obj_a['id']}__{obj_b['id']}"
            gt_distances[key] = dist_m * 100.0   # convert to cm

        scenes.append({
            "scene_id":             scene_dir.name,
            "image_path":           str(img_path) if img_path.exists() else None,
            "objects":              objects,
            "gt_distances":         gt_distances,
            "hallucination_labels": meta.get("hallucination_labels", {}),
        })

    logger.info(f"Loaded {len(scenes)} scenes from Spatial457.")
    return scenes


def _convert_hf_spatial457(hf_ds) -> List[Dict]:
    """Convert a HuggingFace Dataset row to our internal scene format."""
    scenes = []
    for row in hf_ds:
        objects  = row.get("objects", [])
        gt_dists = {}
        for obj_a, obj_b in combinations(objects, 2):
            pos_a = np.array(obj_a["position"], dtype=float)
            pos_b = np.array(obj_b["position"], dtype=float)
            key   = f"{obj_a['id']}__{obj_b['id']}"
            gt_dists[key] = float(np.linalg.norm(pos_a - pos_b)) * 100.0

        scenes.append({
            "scene_id":             row.get("scene_id", "unknown"),
            "image_path":           None,        # PIL image stored in row["image"]
            "image_pil":            row.get("image"),
            "objects":              objects,
            "gt_distances":         gt_dists,
            "hallucination_labels": row.get("hallucination_labels", {}),
        })
    return scenes


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def load_epignn(checkpoint_path: str):
    """
    Load the trained EpiGNN from checkpoint.
    Returns the model in eval mode.
    Lazy-imports torch/torch_geometric so the script can still parse without GPU.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from step2_epistemic_gnn.epistemic_gnn import EpistemicGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt.get("config", {})

    model  = EpistemicGNN(
        sem_dim    = cfg.get("sem_dim",    384),
        hidden_dim = cfg.get("hidden_dim", 256),
        num_classes= cfg.get("num_classes", 10),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info(f"Loaded EpiGNN from {checkpoint_path} (device={device}).")
    return model, device


def build_pyg_data(scene: Dict, baseline_preds: Dict[str, float], sem_dim: int = 384):
    """
    Construct a minimal PyTorch Geometric Data object from a scene dict and
    baseline VLM distance predictions — for feeding through the EpiGNN.

    Parameters
    ----------
    scene           : scene dict from load_spatial457
    baseline_preds  : {edge_key: predicted_dist_cm}
    sem_dim         : dimensionality of semantic node embeddings

    Returns  torch_geometric.data.Data
    """
    import torch
    from torch_geometric.data import Data

    objects = scene["objects"]
    id_to_idx = {o["id"]: i for i, o in enumerate(objects)}
    N = len(objects)

    # Node features: zero-initialised semantics + bbox placeholder + depth=1
    node_sem   = torch.zeros(N, sem_dim)
    node_bbox  = torch.zeros(N, 4)
    node_depth = torch.ones(N, 1)

    src_list, tgt_list = [], []
    dist_list, conf_list, angle_list, depth_diff_list = [], [], [], []

    for obj_a, obj_b in combinations(objects, 2):
        key = f"{obj_a['id']}__{obj_b['id']}"
        if key not in baseline_preds:
            continue
        i, j = id_to_idx[obj_a["id"]], id_to_idx[obj_b["id"]]
        dist = baseline_preds[key]
        if not np.isfinite(dist):
            dist = 100.0     # fallback to 1 m when VLM refused

        # Directed edge i→j
        src_list.append(i); tgt_list.append(j)
        dist_list.append(dist)
        conf_list.append(0.5)        # unknown confidence for baselines
        angle_list.append(0.0)
        depth_diff_list.append(0.0)

    if not src_list:
        return None

    edge_index     = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_dist      = torch.tensor(dist_list, dtype=torch.float).unsqueeze(1)
    edge_conf      = torch.tensor(conf_list, dtype=torch.float).unsqueeze(1)
    edge_angle     = torch.tensor(angle_list, dtype=torch.float).unsqueeze(1)
    edge_depth_diff= torch.tensor(depth_diff_list, dtype=torch.float).unsqueeze(1)

    return Data(
        node_sem=node_sem, node_bbox=node_bbox, node_depth=node_depth,
        edge_index=edge_index, edge_dist=edge_dist, edge_conf=edge_conf,
        edge_angle=edge_angle, edge_depth_diff=edge_depth_diff,
    )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene: Dict,
    baseline: VLMBaseline,
    epignn,
    device,
    epsilon: float,
) -> Dict:
    """
    Run one scene through the baseline VLM + EpiGNN and return per-scene stats.
    """
    import torch

    objects = scene["objects"]
    pairs   = [(o_a["id"], o_b["id"]) for o_a, o_b in combinations(objects, 2)]
    gt_dists = scene["gt_distances"]

    image_path = scene.get("image_path") or "__no_image__"
    image_pil  = scene.get("image_pil")

    # --- Baseline VLM predictions ---
    if image_pil is not None:
        import tempfile, os
        from PIL import Image as PILImage
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image_pil.save(tmp.name)
        image_path = tmp.name

    baseline_preds_list = (
        baseline.predict_distances(image_path, pairs)
        if os.path.exists(image_path)
        else [float("nan")] * len(pairs)
    )
    baseline_preds = {
        f"{a}__{b}": v for (a, b), v in zip(pairs, baseline_preds_list)
    }

    # --- GT alignment ---
    gt_vals, pred_vals_baseline, pred_vals_gnn = [], [], []
    hallucination_labels, residuals_list = [], []

    pyg_data = build_pyg_data(scene, baseline_preds)
    gnn_out  = None
    if pyg_data is not None and epignn is not None:
        with torch.no_grad():
            gnn_out = epignn(pyg_data.to(device))

    edges = list(combinations(range(len(objects)), 2))
    for edge_idx, (i, j) in enumerate(edges):
        key = f"{objects[i]['id']}__{objects[j]['id']}"
        if key not in gt_dists:
            continue
        gt = gt_dists[key]
        bp = baseline_preds.get(key, float("nan"))

        gt_vals.append(gt)
        pred_vals_baseline.append(bp)

        if gnn_out is not None and edge_idx < gnn_out["pred_dist"].shape[0]:
            gnn_dist = float(gnn_out["pred_dist"][edge_idx].item())
            res      = float(gnn_out["residuals"][edge_idx].item())
        else:
            gnn_dist = bp
            res      = float("nan")

        pred_vals_gnn.append(gnn_dist)
        residuals_list.append(res)

        hl = scene["hallucination_labels"].get(key, None)
        hallucination_labels.append(hl)

    # Triangle violation on baseline predictions
    dist_dict_baseline = {}
    for (i, j), bp in zip(edges, baseline_preds_list):
        k = (i, j)
        if np.isfinite(bp):
            dist_dict_baseline[k] = bp

    tv_rate, tv_mag = triangle_violation_rate(
        dist_dict_baseline, list(range(len(objects)))
    )

    # Per-scene residuals as post-hoc filter (for ablation A1 reference)
    if pyg_data is not None:
        ei  = pyg_data.edge_index.numpy()
        ed  = pyg_data.edge_dist.squeeze().numpy()
        standalone_residuals = compute_residuals_from_distances(ei, ed)
    else:
        standalone_residuals = np.array([])

    return {
        "scene_id":               scene["scene_id"],
        "gt":                     gt_vals,
        "pred_baseline":          pred_vals_baseline,
        "pred_gnn":               pred_vals_gnn,
        "residuals":              residuals_list,
        "standalone_residuals":   standalone_residuals.tolist(),
        "hallucination_labels":   hallucination_labels,
        "triangle_violation_rate": tv_rate,
        "triangle_violation_mag":  tv_mag,
    }


def aggregate_results(
    scene_results: List[Dict],
    model_tag: str,
    epsilon: float,
) -> EvalResult:
    """Aggregate per-scene dicts into a single EvalResult."""
    all_gt   = []
    all_base = []
    all_gnn  = []
    all_res  = []
    all_hl   = []
    tv_rates = []

    for s in scene_results:
        all_gt.extend(s["gt"])
        all_base.extend(s["pred_baseline"])
        all_gnn.extend(s["pred_gnn"])
        all_res.extend(s["residuals"])
        all_hl.extend(s["hallucination_labels"])
        if np.isfinite(s["triangle_violation_rate"]):
            tv_rates.append(s["triangle_violation_rate"])

    all_gt   = np.array(all_gt,   dtype=float)
    all_base = np.array(all_base, dtype=float)
    all_gnn  = np.array(all_gnn,  dtype=float)
    all_res  = np.array(all_res,  dtype=float)

    result = EvalResult(dataset="Spatial457", model_tag=model_tag)
    result.mae                   = compute_mae(all_gnn, all_gt)
    result.mra                   = compute_mra(all_gnn, all_gt)
    result.triangle_violation_rate = float(np.nanmean(tv_rates)) if tv_rates else float("nan")
    result.mean_residual         = float(np.nanmean(all_res))

    # Hallucination correlation — only when labels are available
    valid_hl = [(r, l) for r, l in zip(all_res, all_hl) if l is not None and np.isfinite(r)]
    if valid_hl:
        res_v, hl_v = zip(*valid_hl)
        corr = residual_hallucination_correlation(np.array(res_v), np.array(hl_v))
        result.residual_hallucination_pearson = corr.get("pearson_r", float("nan"))
        result.residual_hallucination_auroc   = corr.get("auroc",     float("nan"))
        trigger = trigger_precision_recall_f1(
            np.array(res_v), np.array(hl_v), epsilon)
        result.trigger_precision = trigger["precision"]
        result.trigger_recall    = trigger["recall"]
        result.trigger_f1        = trigger["f1"]

    # Baseline MAE as extra
    result.extras["baseline_mae"] = compute_mae(all_base, all_gt)
    result.extras["gnn_mae"]      = result.mae

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EpiGNN + baselines on Spatial457.")
    parser.add_argument("--model",     required=True,
                        help="Path to EpiGNN checkpoint (.pt)")
    parser.add_argument("--dataset",   default="data/spatial457",
                        help="Local dataset path or HF identifier")
    parser.add_argument("--baselines", nargs="+", default=["mock"],
                        help="Baseline tags to evaluate (see baseline.py registry)")
    parser.add_argument("--output",    default="results/spatial457.json",
                        help="Path to write JSON results")
    parser.add_argument("--epsilon",   type=float, default=0.3,
                        help="Residual threshold ε for trigger decisions")
    parser.add_argument("--max_scenes",type=int, default=None,
                        help="Evaluate on first N scenes only (debug)")
    parser.add_argument("--no_gnn",   action="store_true",
                        help="Skip EpiGNN; evaluate VLM baselines only")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    os.makedirs(Path(args.output).parent, exist_ok=True)

    scenes = load_spatial457(args.dataset)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    logger.info(f"Evaluating on {len(scenes)} scenes.")

    epignn, device = (None, "cpu") if args.no_gnn else load_epignn(args.model)

    all_results = {}

    for bl_tag in args.baselines:
        logger.info(f"\n{'─'*60}\nBaseline: {bl_tag}\n{'─'*60}")
        baseline = build_baseline(bl_tag)

        scene_results = []
        for idx, scene in enumerate(scenes):
            logger.info(f"  Scene {idx+1}/{len(scenes)}: {scene['scene_id']}")
            try:
                sr = evaluate_scene(scene, baseline, epignn, device, args.epsilon)
                scene_results.append(sr)
            except Exception as e:
                logger.warning(f"  Scene {scene['scene_id']} failed: {e}")

        result = aggregate_results(scene_results, model_tag=bl_tag,
                                   epsilon=args.epsilon)
        print_result(result)
        all_results[bl_tag] = {
            "summary": result.to_dict(),
            "per_scene": scene_results,
        }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
