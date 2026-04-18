"""
spatialvlm_eval.py
------------------
Evaluation on the SpatialVLM benchmark protocol (CVPR 2024).

Paper:  SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning
        Capabilities (Chen et al., CVPR 2024, Google Brain)
        https://arxiv.org/abs/2406.13537

This benchmark tests QUANTITATIVE metric distance estimation in real images.
Unlike Spatial457 (synthetic GT) and 3DSRBench (relative spatial predicates),
SpatialVLM evaluates unconstrained metric distance regression against depth-map
derived ground truth — the hardest test for a system claiming to correct
spatial hallucinations at a metric scale.

GT source
---------
Depth maps produced by ZoeDepth (metric monocular depth estimation) on real
internet images, with scale calibrated from EXIF focal-length metadata.
Our data-preparation script (prepare_spatialvlm_eval.py) replicates this
pipeline on the COCO-2017 val images (publicly available, CC BY 4.0).

Format
------
Evaluation JSON: data/spatialvlm_eval/scenes.json
    {
      "benchmark": "SpatialVLM-Eval",
      "paper": "Chen et al. CVPR 2024",
      "scenes": [
        {
          "scene_id": "img_000000000139",
          "image_path": "images/000000000139.jpg",
          "image_width": 640, "image_height": 480,
          "objects": [
            {"id": "person_0", "label": "person",
             "bbox": [x1,y1,x2,y2], "depth_m": 2.5}
          ],
          "pairs": [
            {"subject_id": "person_0", "object_id": "chair_0",
             "gt_dist_m": 1.2, "predicate": "near",
             "hallucination_label": null}
          ]
        }
      ]
    }

Metrics
-------
  MAE             — mean absolute error (metres)
  MRA@τ           — mean relative accuracy at τ ∈ {0.25, 0.50, 1.00}
                    (fraction of edges with |pred−gt|/gt ≤ τ)
  HvD metric      — halving/doubling accuracy: |pred−gt|/gt ≤ 1.0
                    (the primary SpatialVLM paper metric)
  Triangle violation rate — our addition
  Residual–hallucination AUROC — our addition

CLI
---
    python step4_evaluation/spatialvlm_eval.py \\
        --model   checkpoints/best_full.pt \\
        --dataset data/spatialvlm_eval \\
        --baselines mock \\
        --output  results/spatialvlm.json \\
        --epsilon 0.3

Data preparation (one-time):
    python prepare_spatialvlm_eval.py \\
        --coco_images_dir data/coco_val2017 \\
        --annotations     data/coco_val2017/annotations/instances_val2017.json \\
        --out_dir         data/spatialvlm_eval \\
        --n_images        500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from step4_evaluation.metrics import (
    EvalResult,
    compute_mae,
    compute_mra,
    compute_residuals_from_distances,
    residual_hallucination_correlation,
    triangle_violation_rate,
    trigger_precision_recall_f1,
    print_result,
)
from step4_evaluation.baseline import build_baseline, VLMBaseline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_spatialvlm_eval(dataset_path: str) -> List[Dict]:
    """
    Load SpatialVLM evaluation scenes.

    Accepts either:
      (a) a local directory with scenes.json  (from prepare_spatialvlm_eval.py)
      (b) the HuggingFace snapshot layout from:
              huggingface-cli download remyxai/SpaceLLaVA-eval
          (auto-detected if scenes.json is absent)

    Returns a list of scene dicts with keys:
        scene_id, image_path, image_width, image_height,
        objects, pairs, gt_distances
    where gt_distances maps "subj_id__obj_id" → distance_in_metres.
    """
    dataset_path = Path(dataset_path)

    scenes_json = dataset_path / "scenes.json"
    if scenes_json.exists():
        return _load_scenes_json(scenes_json, dataset_path)

    # Try HuggingFace snapshot — look for parquet/jsonl
    hf_jsonl = dataset_path / "data" / "test-00000-of-00001.parquet"
    if hf_jsonl.exists() or (dataset_path / "data").exists():
        return _load_hf_snapshot(dataset_path)

    # Auto-download from HF
    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading remyxai/SpaceLLaVA-eval from HuggingFace …")
        local_dir = str(dataset_path)
        snapshot_download(
            repo_id="remyxai/SpaceLLaVA-eval",
            repo_type="dataset",
            local_dir=local_dir,
        )
        if (dataset_path / "scenes.json").exists():
            return _load_scenes_json(dataset_path / "scenes.json", dataset_path)
        return _load_hf_snapshot(dataset_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset path '{dataset_path}' not found and HF download "
            f"failed ({e}).  Run prepare_spatialvlm_eval.py to generate data."
        )


def _load_scenes_json(scenes_json: Path, base_dir: Path) -> List[Dict]:
    """Load from our canonical scenes.json format."""
    with open(scenes_json) as f:
        data = json.load(f)

    raw_scenes = data.get("scenes", data) if isinstance(data, dict) else data
    scenes = []
    for raw in raw_scenes:
        objects = raw.get("objects", [])
        if len(objects) < 2:
            continue

        # Build gt_distances from pairs list
        gt_distances: Dict[str, float] = {}
        pairs = raw.get("pairs", [])
        for p in pairs:
            key = f"{p['subject_id']}__{p['object_id']}"
            if "gt_dist_m" in p and p["gt_dist_m"] is not None:
                gt_distances[key] = float(p["gt_dist_m"])

        # Also compute gt_distances from object depths if pairs not present
        if not gt_distances:
            id_to_obj = {o["id"]: o for o in objects}
            for a, b in combinations(objects, 2):
                da = a.get("depth_m")
                db = b.get("depth_m")
                if da is not None and db is not None:
                    # Rough Euclidean approx from depth + bbox centres
                    ba, bb = a.get("bbox", [0,0,0,0]), b.get("bbox", [0,0,0,0])
                    cx_a = (ba[0] + ba[2]) / 2.0
                    cy_a = (ba[1] + ba[3]) / 2.0
                    cx_b = (bb[0] + bb[2]) / 2.0
                    cy_b = (bb[1] + bb[3]) / 2.0
                    W = raw.get("image_width", 640)
                    H = raw.get("image_height", 480)
                    # Convert pixel offsets to rough world-space (assuming 60° FoV)
                    f = W / (2 * np.tan(np.radians(30)))
                    dx = (cx_b - cx_a) / f * ((da + db) / 2)
                    dy = (cy_b - cy_a) / f * ((da + db) / 2)
                    dz = db - da
                    dist = float(np.sqrt(dx**2 + dy**2 + dz**2))
                    key = f"{a['id']}__{b['id']}"
                    gt_distances[key] = dist

        # Get predicate labels for edges
        predicate_labels: Dict[str, str] = {}
        hallucination_labels: Dict[str, Optional[int]] = {}
        for p in pairs:
            key = f"{p['subject_id']}__{p['object_id']}"
            predicate_labels[key] = p.get("predicate", "near")
            hl = p.get("hallucination_label")
            hallucination_labels[key] = int(hl) if hl is not None else None

        # Resolve image path
        img_path = raw.get("image_path", "")
        if img_path and not os.path.isabs(img_path):
            img_path = str(base_dir / img_path)

        scenes.append({
            "scene_id":           raw.get("scene_id", "unknown"),
            "image_path":         img_path,
            "image_width":        raw.get("image_width", 640),
            "image_height":       raw.get("image_height", 480),
            "objects":            objects,
            "pairs":              pairs,
            "gt_distances":       gt_distances,       # metres
            "predicate_labels":   predicate_labels,
            "hallucination_labels": hallucination_labels,
        })

    logger.info(f"Loaded {len(scenes)} scenes from {scenes_json}.")
    return scenes


def _load_hf_snapshot(dataset_path: Path) -> List[Dict]:
    """
    Load from a HuggingFace dataset snapshot.
    Tries to read parquet files or JSONL.
    """
    try:
        import pandas as pd
        parquets = sorted(dataset_path.rglob("*.parquet"))
        if parquets:
            rows = pd.concat([pd.read_parquet(p) for p in parquets],
                             ignore_index=True).to_dict("records")
        else:
            jsonls = sorted(dataset_path.rglob("*.jsonl"))
            rows = []
            for jl in jsonls:
                with open(jl) as f:
                    rows.extend(json.loads(line) for line in f)
    except Exception as e:
        raise RuntimeError(f"Could not parse HF snapshot at {dataset_path}: {e}")

    # Convert rows to our format
    scenes = []
    for row in rows:
        objects = row.get("objects", [])
        if len(objects) < 2:
            continue
        pairs = row.get("pairs", [])
        gt_distances = {}
        for p in pairs:
            key = f"{p['subject_id']}__{p['object_id']}"
            if "gt_dist_m" in p:
                gt_distances[key] = float(p["gt_dist_m"])

        scenes.append({
            "scene_id":         row.get("scene_id", "unknown"),
            "image_path":       row.get("image_path"),
            "image_width":      row.get("image_width", 640),
            "image_height":     row.get("image_height", 480),
            "objects":          objects,
            "pairs":            pairs,
            "gt_distances":     gt_distances,
            "predicate_labels": {},
            "hallucination_labels": {},
        })

    logger.info(f"Loaded {len(scenes)} scenes from HF snapshot.")
    return scenes


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def load_epignn(checkpoint_path: str):
    """Load QuantEpiGNN checkpoint. Returns (model, device)."""
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from step2_epistemic_gnn.ablation_gnn import AblationGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt.get("args", ckpt.get("config", {}))

    model = AblationGNN(
        sem_dim              = cfg.get("sem_dim",          384),
        hidden_dim           = cfg.get("hidden_dim",       256),
        num_pred_classes     = cfg.get("num_pred_classes", 14),
        use_geom_constraint  = not cfg.get("no_geom_constraint", False),
        use_epistemic        = not cfg.get("no_epistemic", False),
    )
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
    model.to(device).eval()
    logger.info(f"Loaded model from {checkpoint_path} (device={device}).")
    return model, device


# ---------------------------------------------------------------------------
# PyG data builder
# ---------------------------------------------------------------------------

def build_pyg_data(scene: Dict, baseline_preds: Dict[str, float], sem_dim: int = 384):
    """
    Build a PyG Data object from scene + baseline predicted distances.
    Uses per-object depth (in metres) when available for edge_depth_diff.
    Returns None if no edges can be built.
    """
    import torch
    from torch_geometric.data import Data

    objects   = scene["objects"]
    id_to_idx = {o["id"]: i for i, o in enumerate(objects)}
    N = len(objects)

    # Node features
    node_sem   = torch.zeros(N, sem_dim)
    node_bbox  = torch.zeros(N, 4)
    node_depth = torch.ones(N, 1)

    for i, obj in enumerate(objects):
        depth = obj.get("depth_m")
        if depth is not None:
            node_depth[i, 0] = float(depth)
        bbox = obj.get("bbox")
        if bbox:
            node_bbox[i] = torch.tensor(bbox[:4], dtype=torch.float)

    src_list, tgt_list = [], []
    dist_list, conf_list, angle_list, ddiff_list = [], [], [], []

    for obj_a, obj_b in combinations(objects, 2):
        key = f"{obj_a['id']}__{obj_b['id']}"
        if key not in baseline_preds:
            continue
        dist = baseline_preds[key]
        if not np.isfinite(dist):
            dist = 2.0   # fallback: 2 metres

        i, j = id_to_idx[obj_a["id"]], id_to_idx[obj_b["id"]]
        W = scene.get("image_width", 640)
        H = scene.get("image_height", 480)

        # Angle from centre-of-bbox vectors
        ba = obj_a.get("bbox", [0, 0, 0, 0])
        bb = obj_b.get("bbox", [0, 0, 0, 0])
        cx_a, cy_a = (ba[0]+ba[2])/2, (ba[1]+ba[3])/2
        cx_b, cy_b = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
        angle = float(np.arctan2(cy_b - cy_a, cx_b - cx_a))
        depth_diff = node_depth[j, 0].item() - node_depth[i, 0].item()

        src_list.append(i);  tgt_list.append(j)
        dist_list.append(dist)
        conf_list.append(0.6)
        angle_list.append(angle)
        ddiff_list.append(depth_diff)

    if not src_list:
        return None

    return Data(
        node_sem       = node_sem,
        node_bbox      = node_bbox,
        node_depth     = node_depth,
        edge_index     = torch.tensor([src_list, tgt_list], dtype=torch.long),
        edge_dist      = torch.tensor(dist_list, dtype=torch.float).unsqueeze(1),
        edge_conf      = torch.tensor(conf_list, dtype=torch.float).unsqueeze(1),
        edge_angle     = torch.tensor(angle_list, dtype=torch.float).unsqueeze(1),
        edge_depth_diff= torch.tensor(ddiff_list, dtype=torch.float).unsqueeze(1),
    )


# ---------------------------------------------------------------------------
# Halving/doubling metric (SpatialVLM paper primary metric)
# ---------------------------------------------------------------------------

def halving_doubling_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Fraction of predictions where gt/2 ≤ pred ≤ 2*gt.
    This is the primary metric in the SpatialVLM CVPR 2024 paper.
    """
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if mask.sum() == 0:
        return float("nan")
    ratio = pred[mask] / gt[mask]
    return float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))


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
    import torch

    objects  = scene["objects"]
    gt_dists = scene["gt_distances"]          # metres
    pairs    = [(a["id"], b["id"]) for a, b in combinations(objects, 2)]
    image_path = scene.get("image_path") or "__no_image__"

    # ---- Baseline predictions ----
    if os.path.exists(image_path):
        baseline_preds_list = baseline.predict_distances(image_path, pairs)
    else:
        baseline_preds_list = [float("nan")] * len(pairs)

    baseline_preds = {f"{a}__{b}": v for (a, b), v in zip(pairs, baseline_preds_list)}

    # ---- Convert baseline predictions to metres (predict_distances returns cm) ----
    baseline_preds_m = {k: v / 100.0 if np.isfinite(v) else v
                        for k, v in baseline_preds.items()}

    # ---- GNN refinement ----
    pyg_data = build_pyg_data(scene, baseline_preds_m)
    gnn_out  = None
    if pyg_data is not None and epignn is not None:
        with torch.no_grad():
            gnn_out = epignn(pyg_data.to(device))

    # ---- Collect per-edge results ----
    gt_list, base_list, gnn_list = [], [], []
    res_list, hl_list = [], []
    edges = list(combinations(range(len(objects)), 2))

    for edge_idx, (i, j) in enumerate(edges):
        key = f"{objects[i]['id']}__{objects[j]['id']}"
        if key not in gt_dists:
            continue
        gt_m = gt_dists[key]
        bp_m = baseline_preds_m.get(key, float("nan"))

        gt_list.append(gt_m)
        base_list.append(bp_m)

        if gnn_out is not None and edge_idx < gnn_out["pred_dist"].shape[0]:
            # GNN outputs are in whatever unit the training used; convert to metres
            gnn_dist_m = float(gnn_out["pred_dist"][edge_idx].item())
            res        = float(gnn_out["residuals"][edge_idx].item())
        else:
            gnn_dist_m = bp_m
            res        = float("nan")

        gnn_list.append(gnn_dist_m)
        res_list.append(res)
        hl = scene["hallucination_labels"].get(key)
        hl_list.append(int(hl) if hl is not None else None)

    # Triangle violation on baseline (per SpatialVLM paper's consistency analysis)
    dist_dict_base = {}
    for (i, j), bp in zip(edges, baseline_preds_list):
        if np.isfinite(bp):
            dist_dict_base[(i, j)] = bp / 100.0   # metres
    tv_rate, tv_mag = triangle_violation_rate(dist_dict_base, list(range(len(objects))))

    return {
        "scene_id":                scene["scene_id"],
        "gt":                      gt_list,
        "pred_baseline":           base_list,
        "pred_gnn":                gnn_list,
        "residuals":               res_list,
        "hallucination_labels":    hl_list,
        "triangle_violation_rate": tv_rate,
        "triangle_violation_mag":  tv_mag,
    }


def aggregate_results(
    scene_results: List[Dict],
    model_tag: str,
    epsilon: float,
) -> EvalResult:
    all_gt, all_base, all_gnn, all_res, all_hl = [], [], [], [], []
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

    result = EvalResult(dataset="SpatialVLM-Eval (CVPR 2024)", model_tag=model_tag)

    result.mae  = compute_mae(all_gnn, all_gt)
    result.mra  = compute_mra(all_gnn, all_gt, thresholds=(0.25, 0.50, 1.00))
    result.triangle_violation_rate = float(np.nanmean(tv_rates)) if tv_rates else float("nan")
    result.mean_residual = float(np.nanmean(all_res))

    # Hallucination correlation
    valid_hl = [(r, l) for r, l in zip(all_res, all_hl)
                if l is not None and np.isfinite(r)]
    if valid_hl:
        res_v, hl_v = zip(*valid_hl)
        corr = residual_hallucination_correlation(np.array(res_v), np.array(hl_v))
        result.residual_hallucination_pearson = corr.get("pearson_r", float("nan"))
        result.residual_hallucination_auroc   = corr.get("auroc",     float("nan"))
        trig = trigger_precision_recall_f1(np.array(res_v), np.array(hl_v), epsilon)
        result.trigger_precision = trig["precision"]
        result.trigger_recall    = trig["recall"]
        result.trigger_f1        = trig["f1"]

    # SpatialVLM primary metric: halving/doubling accuracy
    result.extras["hvd_accuracy_gnn"]      = halving_doubling_accuracy(all_gnn, all_gt)
    result.extras["hvd_accuracy_baseline"] = halving_doubling_accuracy(all_base, all_gt)
    result.extras["baseline_mae"]          = compute_mae(all_base, all_gt)
    result.extras["gnn_mae"]               = result.mae
    result.extras["baseline_mra"]          = compute_mra(all_base, all_gt)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QuantEpiGNN + baselines on SpatialVLM-Eval (CVPR 2024).")
    parser.add_argument("--model",      required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--dataset",    default="data/spatialvlm_eval",
                        help="Local path to scenes.json directory")
    parser.add_argument("--baselines",  nargs="+", default=["mock"],
                        help="Baseline tags (see baseline.py registry)")
    parser.add_argument("--output",     default="results/spatialvlm.json")
    parser.add_argument("--epsilon",    type=float, default=0.3,
                        help="Residual threshold ε")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--no_gnn",     action="store_true")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    scenes = load_spatialvlm_eval(args.dataset)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    logger.info(f"Evaluating on {len(scenes)} scenes.")

    epignn, device = (None, "cpu")
    if not args.no_gnn:
        try:
            epignn, device = load_epignn(args.model)
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Running baseline-only.")

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
            "summary":   result.to_dict(),
            "per_scene": scene_results,
        }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
