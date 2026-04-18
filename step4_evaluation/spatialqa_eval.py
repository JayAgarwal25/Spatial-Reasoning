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

    Supports two on-disk formats:

    Format A — per-scene directory layout (hand-curated or converted):
        <dataset_path>/data/scene_0001/image.jpg
                                      /metadata.json   ← {objects, hallucination_labels}
        metadata.json objects schema: [{"id": str, "position": [x, y, z]}, ...]

    Format B — superCLEVR bulk download (RyanWW/Spatial457 HF snapshot):
        <dataset_path>/spatial457_scenes_21k.json   ← all scenes + 3d_coords
        <dataset_path>/images/superCLEVR_new_*.png  ← RGB images

    Returns list of scene dicts with keys:
        scene_id, image_path, objects, gt_distances, hallucination_labels
    """
    dataset_path = Path(dataset_path)

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

    # Format B: superCLEVR bulk JSON
    bulk_json = dataset_path / "spatial457_scenes_21k.json"
    if bulk_json.exists():
        return _load_super_clevr_format(dataset_path, bulk_json)

    # Format A: per-scene subdirectories
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path

    scenes = []
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

        gt_distances: Dict[str, float] = {}
        for obj_a, obj_b in combinations(objects, 2):
            pos_a = np.array(obj_a["position"], dtype=float)
            pos_b = np.array(obj_b["position"], dtype=float)
            dist_m = float(np.linalg.norm(pos_a - pos_b))
            key = f"{obj_a['id']}__{obj_b['id']}"
            gt_distances[key] = dist_m   # metres

        scenes.append({
            "scene_id":             scene_dir.name,
            "image_path":           str(img_path) if img_path.exists() else None,
            "objects":              objects,
            "gt_distances":         gt_distances,
            "hallucination_labels": meta.get("hallucination_labels", {}),
        })

    logger.info(f"Loaded {len(scenes)} scenes from Spatial457.")
    return scenes


def _load_super_clevr_format(dataset_path: Path, bulk_json: Path) -> List[Dict]:
    """
    Parse the RyanWW/Spatial457 HuggingFace snapshot layout.

    Each scene entry in spatial457_scenes_21k.json has:
        image_filename  — PNG basename, e.g. "superCLEVR_new_020000.png"
        objects         — list of dicts with keys:
                            3d_coords [x, y, z]  (Blender world units ≈ metres)
                            size, color, shape

    Object IDs are constructed as "{size}_{color}_{shape}" with a numeric
    suffix appended when two objects in the same scene share the same label.
    """
    import os
    img_dir = dataset_path / "images"
    available_imgs = set(os.listdir(img_dir)) if img_dir.exists() else set()

    with open(bulk_json) as f:
        raw = json.load(f)

    scenes = []
    for entry in raw.get("scenes", []):
        img_fname = entry.get("image_filename", "")
        if img_fname not in available_imgs:
            continue

        raw_objects = entry.get("objects", [])
        if len(raw_objects) < 2:
            continue

        # Build unique string IDs
        label_counts: Dict[str, int] = {}
        objects = []
        mask_boxes = entry.get("obj_mask_box", {})
        for obj_idx, obj in enumerate(raw_objects):
            base_id = f"{obj.get('size','?')}_{obj.get('color','?')}_{obj.get('shape','?')}"
            count = label_counts.get(base_id, 0)
            label_counts[base_id] = count + 1
            obj_id = base_id if count == 0 else f"{base_id}_{count}"
            coords = obj.get("3d_coords", [0.0, 0.0, 0.0])
            pixel_c = obj.get("pixel_coords", [[0.0, 0.0, 0.0], [640, 480]])
            cx, cy = float(pixel_c[0][0]), float(pixel_c[0][1])
            obj_mask = mask_boxes.get(str(obj_idx), {})
            bbox = list(obj_mask.get("obj", [[0, 0, 1, 1]])[0])
            objects.append({
                "id": obj_id, "position": coords,
                "pixel_center": [cx, cy], "bbox": bbox,
            })

        gt_distances: Dict[str, float] = {}
        for obj_a, obj_b in combinations(objects, 2):
            pos_a = np.array(obj_a["position"], dtype=float)
            pos_b = np.array(obj_b["position"], dtype=float)
            dist_m = float(np.linalg.norm(pos_a - pos_b))
            key = f"{obj_a['id']}__{obj_b['id']}"
            gt_distances[key] = dist_m   # metres — matches GNN training units

        scene_id = img_fname.replace(".png", "").replace(".jpg", "")
        img_path = str(img_dir / img_fname)
        scenes.append({
            "scene_id":             scene_id,
            "image_path":           img_path,
            "objects":              objects,
            "gt_distances":         gt_distances,
            "hallucination_labels": {},
        })

    logger.info(f"Loaded {len(scenes)} scenes from superCLEVR/Spatial457 format.")
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
            gt_dists[key] = float(np.linalg.norm(pos_a - pos_b))   # metres

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

    from step2_epistemic_gnn.epistemic_gnn import QuantEpiGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    # train.py saves args under "args" key; fall back to "config" for compat
    cfg    = ckpt.get("args", ckpt.get("config", {}))

    model  = QuantEpiGNN(
        sem_dim          = cfg.get("sem_dim",          384),
        hidden_dim       = cfg.get("hidden_dim",       256),
        num_pred_classes = cfg.get("num_pred_classes", 14),
    )
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
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
            dist = 2.0     # fallback: 2 m when VLM refused

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
# Step 3: visual grounding feedback loop
# ---------------------------------------------------------------------------

def run_step3_feedback(
    scene: Dict,
    pyg_data,
    gnn_out: Dict,
    baseline: "VLMBaseline",
    model,
    device,
    epsilon: float,
    max_iters: int = 2,
) -> List[float]:
    """
    Run the visual grounding feedback loop for one scene.
    For each high-residual edge: annotate image → re-query VLM → update distances → re-run GNN.
    Returns updated per-edge predicted distances (same length as edges list).
    """
    import torch
    import cv2
    import tempfile

    from step1_scene_graph.schemas import ObjectNode, RelationEdge
    from step3_visual_agent.actions import draw_bbox, draw_line

    objects = scene["objects"]
    N = len(objects)
    edges = list(combinations(range(N), 2))

    img_path = scene.get("image_path")
    if not img_path or not os.path.exists(img_path):
        return gnn_out["pred_dist"].squeeze(1).cpu().tolist()

    current_dists = gnn_out["pred_dist"].squeeze(1).cpu().tolist()

    for _iter in range(max_iters):
        residuals = gnn_out["residuals"].squeeze(1).cpu()
        flagged = (residuals > epsilon).nonzero(as_tuple=True)[0].tolist()
        if not flagged:
            break

        img = cv2.imread(img_path)
        if img is None:
            break

        annotated = img.copy()
        for edge_idx in flagged:
            i, j = edges[edge_idx]
            oi, oj = objects[i], objects[j]

            bbox_i = oi.get("bbox", [0, 0, 10, 10])
            cx_i, cy_i = oi.get("pixel_center", [0.0, 0.0])
            node_i = ObjectNode(
                id=i, label=oi["id"], bbox=bbox_i, confidence=1.0,
                center=[cx_i, cy_i], width=float(bbox_i[2]), height=float(bbox_i[3]),
                area=float(bbox_i[2]) * float(bbox_i[3]),
            )
            bbox_j = oj.get("bbox", [0, 0, 10, 10])
            cx_j, cy_j = oj.get("pixel_center", [0.0, 0.0])
            node_j = ObjectNode(
                id=j, label=oj["id"], bbox=bbox_j, confidence=1.0,
                center=[cx_j, cy_j], width=float(bbox_j[2]), height=float(bbox_j[3]),
                area=float(bbox_j[2]) * float(bbox_j[3]),
            )
            rel_edge = RelationEdge(subject_id=i, object_id=j, predicate="near", confidence=0.5)

            r1 = draw_bbox(annotated, node_i, edge_idx, float(residuals[edge_idx]))
            r2 = draw_bbox(r1.image, node_j, edge_idx, float(residuals[edge_idx]))
            r3 = draw_line(
                r2.image, node_i, node_j, rel_edge, edge_idx,
                float(residuals[edge_idx]), float(current_dists[edge_idx]),
            )
            annotated = r3.image

        ann_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, annotated)
                ann_path = tmp.name

            flagged_pairs = [
                (objects[edges[idx][0]]["id"], objects[edges[idx][1]]["id"])
                for idx in flagged
            ]
            new_dists = baseline.predict_distances(ann_path, flagged_pairs)
            for k, edge_idx in enumerate(flagged):
                if k < len(new_dists) and np.isfinite(new_dists[k]) and new_dists[k] > 0:
                    current_dists[edge_idx] = new_dists[k]
        except Exception as exc:
            logger.warning("Step 3 VLM re-query failed at iter %d: %s", _iter, exc)
            break
        finally:
            if ann_path and os.path.exists(ann_path):
                os.unlink(ann_path)

        new_edge_dist = torch.tensor(current_dists, dtype=torch.float).unsqueeze(1)
        updated_data = pyg_data.clone()
        updated_data.edge_dist = new_edge_dist
        with torch.no_grad():
            gnn_out = model(updated_data.to(device))
        current_dists = gnn_out["pred_dist"].squeeze(1).cpu().tolist()
        pyg_data = updated_data.cpu()

    return current_dists


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene: Dict,
    baseline: VLMBaseline,
    epignn,
    device,
    epsilon: float,
    run_feedback: bool = False,
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
        import tempfile
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
        # pyg_data.to(device) mutates in-place in PyG 2.x — move back to CPU
        pyg_data = pyg_data.cpu()

    # --- Step 3: visual grounding feedback ---
    feedback_dists: Optional[List[float]] = None
    if run_feedback and gnn_out is not None and epignn is not None:
        try:
            feedback_dists = run_step3_feedback(
                scene, pyg_data, gnn_out, baseline, epignn, device, epsilon
            )
        except Exception as exc:
            logger.warning("Step 3 feedback failed for scene %s: %s",
                           scene["scene_id"], exc)

    edges = list(combinations(range(len(objects)), 2))
    pred_vals_gnn_feedback = []
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

        fb_dist = (feedback_dists[edge_idx]
                   if feedback_dists is not None and edge_idx < len(feedback_dists)
                   else gnn_dist)
        pred_vals_gnn_feedback.append(fb_dist)

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
        "pred_gnn_feedback":      pred_vals_gnn_feedback,
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
    all_gt       = []
    all_base     = []
    all_gnn      = []
    all_feedback = []
    all_res      = []
    all_hl       = []
    tv_rates     = []

    for s in scene_results:
        all_gt.extend(s["gt"])
        all_base.extend(s["pred_baseline"])
        all_gnn.extend(s["pred_gnn"])
        all_feedback.extend(s.get("pred_gnn_feedback", s["pred_gnn"]))
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
    result.extras["baseline_mae"]  = compute_mae(all_base, all_gt)
    result.extras["gnn_mae"]       = result.mae
    all_feedback = np.array(all_feedback, dtype=float)
    result.extras["feedback_mae"]  = compute_mae(all_feedback, all_gt)

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
    parser.add_argument("--feedback", action="store_true",
                        help="Run Step 3 visual grounding feedback loop after GNN")
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
                sr = evaluate_scene(scene, baseline, epignn, device, args.epsilon,
                                    run_feedback=args.feedback)
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
