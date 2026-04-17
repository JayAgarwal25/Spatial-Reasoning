"""
nuscenes_eval.py
----------------
NOTE ON NAMING: Despite the filename (matching the repo structure from the
README), this script evaluates on **3DSRBench** (ICCV 2025) — the primary
real-world RGB-D benchmark.  The filename mirrors the repository layout;
the dataset argument defaults to 3DSRBench.

Dataset:  HuggingFace  ccvl/3DSRBench  (CC BY 4.0)
GT type:  Real RGB-D depth sensor → tests the system under realistic
          depth noise, directly probing the source-ambiguity limitation
          described in the README.

Key difference from spatialqa_eval.py
--------------------------------------
3DSRBench uses two evaluation protocols:
    CircularEval — model must answer both original and rephrased QA correctly
    FlipEval     — model must answer both original and semantically negated QA

The critical publication-strength experiment here is:
    Do residuals correlate more with VLM hallucinations than with depth noise?
    (Section: "Source ambiguity in the consistency residual", README)

We split scenes by depth-noise level (estimated from depth-map variance) and
report residual–hallucination correlation separately for low-noise and
high-noise groups.

CLI
---
python step4_evaluation/nuscenes_eval.py \
    --model   checkpoints/best.pt \
    --dataset data/3dsrbench \
    --baselines gpt4o mock \
    --output  results/3dsrbench.json \
    --epsilon 0.3
"""

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
    circular_eval,
    flip_eval,
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

def load_3dsrbench(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load 3DSRBench QA pairs.

    Returns
    -------
    distance_items : items that have metric distance GT (for MAE)
    qa_items       : all items (for CircularEval / FlipEval)

    Expected directory layout (after HF download):
        <dataset_path>/
            data/
                <split>/
                    <item_id>/
                        rgb.jpg
                        depth.png          ← 16-bit depth map (mm)
                        metadata.json
                        qa_pairs.json

    metadata.json schema:
    {
        "objects": [
            {"id": "chair", "bbox2d": [x,y,w,h], "depth_mean_mm": 1500.0},
            ...
        ],
        "depth_noise_variance": 0.04        # scalar, estimated from sensor
    }

    qa_pairs.json schema  (list of items):
    [
        {
            "qid":         "q001",
            "type":        "distance" | "relation",
            "question":    "How far is the chair from the table?",
            "choices":     ["A", "B", "C", "D"],
            "answer":      "A",
            "gt_dist_cm":  142.5,            # only for type=distance
            "circular_pair_qid": "q001c",    # circular rephrasing
            "flip_pair_qid":     "q001f",    # negated version
            "hallucination_label": 0         # 1 if the VLM known to hallucinate
        },
        ...
    ]
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        try:
            from datasets import load_dataset
            logger.info("Downloading ccvl/3DSRBench from HuggingFace …")
            hf_ds = load_dataset("ccvl/3DSRBench", split="test")
            return _convert_hf_3dsrbench(hf_ds)
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset path '{dataset_path}' not found and HF download "
                f"failed: {e}"
            )

    distance_items: List[Dict] = []
    qa_items:       List[Dict] = []

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for item_dir in sorted(split_dir.iterdir()):
            if not item_dir.is_dir():
                continue
            meta_path = item_dir / "metadata.json"
            qa_path   = item_dir / "qa_pairs.json"
            rgb_path  = item_dir / "rgb.jpg"
            dep_path  = item_dir / "depth.png"

            if not meta_path.exists() or not qa_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            with open(qa_path) as f:
                qa_pairs = json.load(f)

            # Estimate depth noise from depth map variance (in normalised units)
            depth_noise = meta.get("depth_noise_variance", 0.0)

            base_item = {
                "item_id":      item_dir.name,
                "image_path":   str(rgb_path) if rgb_path.exists() else None,
                "depth_path":   str(dep_path) if dep_path.exists() else None,
                "objects":      meta.get("objects", []),
                "depth_noise":  depth_noise,
            }

            for qa in qa_pairs:
                item = {**base_item, **qa}
                qa_items.append(item)
                if qa.get("type") == "distance" and qa.get("gt_dist_cm") is not None:
                    distance_items.append(item)

    logger.info(
        f"Loaded {len(qa_items)} QA pairs "
        f"({len(distance_items)} with metric GT) from 3DSRBench."
    )
    return distance_items, qa_items


def _convert_hf_3dsrbench(hf_ds) -> Tuple[List[Dict], List[Dict]]:
    """Convert a HuggingFace Dataset to our internal format."""
    distance_items, qa_items = [], []
    for row in hf_ds:
        base = {
            "item_id":     row.get("id", "unknown"),
            "image_path":  None,
            "image_pil":   row.get("image"),
            "depth_path":  None,
            "depth_array": row.get("depth"),        # numpy array if available
            "objects":     row.get("objects", []),
            "depth_noise": row.get("depth_noise_variance", 0.0),
        }
        qa = {
            "qid":                row.get("qid", ""),
            "type":               row.get("type", "relation"),
            "question":           row.get("question", ""),
            "choices":            row.get("choices", []),
            "answer":             row.get("answer", ""),
            "gt_dist_cm":         row.get("gt_dist_cm"),
            "circular_pair_qid":  row.get("circular_pair_qid"),
            "flip_pair_qid":      row.get("flip_pair_qid"),
            "hallucination_label":row.get("hallucination_label"),
        }
        item = {**base, **qa}
        qa_items.append(item)
        if qa["type"] == "distance" and qa["gt_dist_cm"] is not None:
            distance_items.append(item)
    return distance_items, qa_items


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def _ensure_image(item: Dict) -> Optional[str]:
    """Return a valid image path, writing PIL image to tmp if needed."""
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]
    if item.get("image_pil") is not None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        item["image_pil"].save(tmp.name)
        return tmp.name
    return None


def evaluate_distance_items(
    distance_items: List[Dict],
    baseline: VLMBaseline,
    epignn,
    device,
) -> Dict:
    """
    Compute MAE for distance items.
    Returns dict with gt, pred_baseline, pred_gnn, residuals arrays.
    """
    import torch

    gt_list, base_list, gnn_list, res_list, hl_list, noise_list = \
        [], [], [], [], [], []

    for item in distance_items:
        gt  = float(item["gt_dist_cm"])
        objs = item.get("objects", [])
        image_path = _ensure_image(item)

        # Baseline prediction — ask for distance between first two objects if available
        if image_path and len(objs) >= 2:
            preds = baseline.predict_distances(
                image_path, [(objs[0]["id"], objs[1]["id"])]
            )
            bp = preds[0]
        else:
            bp = float("nan")

        # GNN prediction — build minimal single-edge graph
        gnn_dist, res = bp, float("nan")
        if epignn is not None and np.isfinite(bp) and len(objs) >= 2:
            try:
                from torch_geometric.data import Data
                sem_dim = 384
                node_sem   = torch.zeros(2, sem_dim)
                node_bbox  = torch.zeros(2, 4)
                node_depth = torch.ones(2, 1)
                if len(objs) >= 2 and "depth_mean_mm" in objs[0]:
                    node_depth[0, 0] = objs[0]["depth_mean_mm"] / 1000.0
                    node_depth[1, 0] = objs[1]["depth_mean_mm"] / 1000.0
                edge_index = torch.tensor([[0], [1]], dtype=torch.long)
                data = Data(
                    node_sem=node_sem, node_bbox=node_bbox, node_depth=node_depth,
                    edge_index=edge_index,
                    edge_dist=torch.tensor([[bp]], dtype=torch.float),
                    edge_conf=torch.tensor([[0.5]], dtype=torch.float),
                    edge_angle=torch.zeros(1, 1),
                    edge_depth_diff=torch.tensor(
                        [[abs(node_depth[0,0].item() - node_depth[1,0].item())]],
                        dtype=torch.float
                    ),
                )
                with torch.no_grad():
                    out = epignn(data.to(device))
                gnn_dist = float(out["pred_dist"][0].item())
                res      = float(out["residuals"][0].item())
            except Exception as e:
                logger.debug(f"GNN inference failed for item {item.get('qid')}: {e}")

        gt_list.append(gt)
        base_list.append(bp)
        gnn_list.append(gnn_dist)
        res_list.append(res)
        hl_list.append(item.get("hallucination_label"))
        noise_list.append(float(item.get("depth_noise", 0.0)))

    return {
        "gt":                 np.array(gt_list),
        "pred_baseline":      np.array(base_list),
        "pred_gnn":           np.array(gnn_list),
        "residuals":          np.array(res_list),
        "hallucination_labels": hl_list,
        "depth_noise":        np.array(noise_list),
    }


def evaluate_qa_items(
    qa_items: List[Dict],
    baseline: VLMBaseline,
) -> Dict:
    """
    Run CircularEval and FlipEval on all QA items.
    Builds paired arrays for the two protocols.
    """
    # Build a lookup by qid for pair resolution
    qid_to_item = {item["qid"]: item for item in qa_items if "qid" in item}

    circular_pred, circular_gt = [], []
    flip_pred,     flip_gt     = [], []

    for item in qa_items:
        image_path = _ensure_image(item)
        if not image_path:
            continue
        question = item.get("question", "")
        choices  = item.get("choices", [])
        answer   = item.get("answer", "")
        if not choices or not answer:
            continue

        # Predict on original question
        pred_orig = baseline.predict_relation(image_path, question, choices)
        gt_idx_orig = choices.index(answer) if answer in choices else 0

        # --- CircularEval ---
        cpqid = item.get("circular_pair_qid")
        if cpqid and cpqid in qid_to_item:
            cp = qid_to_item[cpqid]
            cp_img = _ensure_image(cp) or image_path
            pred_c = baseline.predict_relation(cp_img, cp["question"], cp["choices"])
            gt_c   = cp["choices"].index(cp["answer"]) if cp["answer"] in cp["choices"] else 0
            circular_pred.extend([
                choices.index(pred_orig) if pred_orig in choices else 0,
                cp["choices"].index(pred_c) if pred_c in cp["choices"] else 0,
            ])
            circular_gt.extend([gt_idx_orig, gt_c])

        # --- FlipEval ---
        fpqid = item.get("flip_pair_qid")
        if fpqid and fpqid in qid_to_item:
            fp = qid_to_item[fpqid]
            fp_img = _ensure_image(fp) or image_path
            pred_f = baseline.predict_relation(fp_img, fp["question"], fp["choices"])
            gt_f   = fp["choices"].index(fp["answer"]) if fp["answer"] in fp["choices"] else 0
            flip_pred.extend([
                choices.index(pred_orig) if pred_orig in choices else 0,
                fp["choices"].index(pred_f) if pred_f in fp["choices"] else 0,
            ])
            flip_gt.extend([gt_idx_orig, gt_f])

    circ_acc = (circular_eval(np.array(circular_pred), np.array(circular_gt))
                if circular_pred else float("nan"))
    flip_acc = (flip_eval(np.array(flip_pred), np.array(flip_gt))
                if flip_pred else float("nan"))

    return {"circular_acc": circ_acc, "flip_acc": flip_acc}


# ---------------------------------------------------------------------------
# Source-ambiguity experiment (publication-strength)
# ---------------------------------------------------------------------------

def source_ambiguity_analysis(
    residuals: np.ndarray,
    hallucination_labels: List,
    depth_noise: np.ndarray,
    noise_percentile: float = 50.0,
) -> Dict:
    """
    Key experiment from README §"Known Limitations":
    Split items by depth-noise level and compare residual–hallucination
    correlation in each group.

    Returns
    -------
    {
        "low_noise_auroc":  float,
        "high_noise_auroc": float,
        "low_noise_pearson": float,
        "high_noise_pearson": float,
        "interpretation":   str,
    }
    """
    hl = np.array([l if l is not None else -1 for l in hallucination_labels])
    mask_valid = (hl >= 0) & np.isfinite(residuals)
    if mask_valid.sum() < 4:
        return {"note": "insufficient labeled samples for source-ambiguity analysis"}

    threshold_noise = np.percentile(depth_noise[mask_valid], noise_percentile)
    low_mask  = mask_valid & (depth_noise <= threshold_noise)
    high_mask = mask_valid & (depth_noise >  threshold_noise)

    def _corr(m):
        if m.sum() < 2:
            return {"pearson_r": float("nan"), "auroc": float("nan")}
        return residual_hallucination_correlation(residuals[m], hl[m])

    low  = _corr(low_mask)
    high = _corr(high_mask)

    # Interpretation heuristic:
    # If AUROC drops significantly in the high-noise group, depth noise
    # confounds the residual signal.
    auroc_drop = low.get("auroc", float("nan")) - high.get("auroc", float("nan"))
    if np.isfinite(auroc_drop):
        if auroc_drop > 0.10:
            interp = ("Depth noise significantly degrades residual signal as a "
                      "hallucination detector (AUROC drops {:.2f}). "
                      "Source-ambiguity is a real concern.".format(auroc_drop))
        else:
            interp = ("Residual signal robust to depth noise level "
                      "(AUROC drop {:.2f}). "
                      "Residuals primarily detect VLM hallucinations.".format(auroc_drop))
    else:
        interp = "Could not compute interpretation (insufficient data)."

    return {
        "low_noise_auroc":   low.get("auroc",     float("nan")),
        "high_noise_auroc":  high.get("auroc",    float("nan")),
        "low_noise_pearson": low.get("pearson_r", float("nan")),
        "high_noise_pearson":high.get("pearson_r",float("nan")),
        "auroc_drop":        float(auroc_drop) if np.isfinite(auroc_drop) else float("nan"),
        "interpretation":    interp,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EpiGNN + baselines on 3DSRBench.")
    parser.add_argument("--model",     required=True,
                        help="Path to EpiGNN checkpoint (.pt)")
    parser.add_argument("--dataset",   default="data/3dsrbench",
                        help="Local dataset path or HF identifier")
    parser.add_argument("--baselines", nargs="+", default=["mock"],
                        help="Baseline tags (see baseline.py)")
    parser.add_argument("--output",    default="results/3dsrbench.json",
                        help="Path to write JSON results")
    parser.add_argument("--epsilon",   type=float, default=0.3,
                        help="Residual threshold ε for trigger decisions")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Evaluate on first N items only (debug)")
    parser.add_argument("--no_gnn",   action="store_true",
                        help="Skip EpiGNN; evaluate VLM baselines only")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    distance_items, qa_items = load_3dsrbench(args.dataset)
    if args.max_items:
        distance_items = distance_items[:args.max_items]
        qa_items       = qa_items[:args.max_items]

    epignn, device = None, "cpu"
    if not args.no_gnn:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from step2_epistemic_gnn.ablation_gnn import AblationGNN
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(args.model, map_location=device)
            cfg  = ckpt.get("args", ckpt.get("config", {}))
            epignn = AblationGNN(
                sem_dim             = cfg.get("sem_dim",          384),
                hidden_dim          = cfg.get("hidden_dim",       256),
                num_pred_classes    = cfg.get("num_pred_classes", 14),
                use_geom_constraint = not cfg.get("no_geom_constraint", False),
                use_epistemic       = not cfg.get("no_epistemic",       False),
            )
            epignn.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
            epignn.to(device).eval()
            logger.info(f"Loaded EpiGNN from {args.model}")
        except Exception as e:
            logger.warning(f"Could not load EpiGNN: {e}. Continuing baseline-only.")
            epignn = None

    all_results = {}

    for bl_tag in args.baselines:
        logger.info(f"\n{'─'*60}\nBaseline: {bl_tag}\n{'─'*60}")
        baseline = build_baseline(bl_tag)

        # --- Distance MAE ---
        logger.info("  Running distance evaluation …")
        dist_out = evaluate_distance_items(distance_items, baseline, epignn, device)

        # --- QA protocols ---
        logger.info("  Running CircularEval / FlipEval …")
        qa_out = evaluate_qa_items(qa_items, baseline)

        # --- Hallucination correlation ---
        hl_labels = dist_out["hallucination_labels"]
        residuals = dist_out["residuals"]
        valid_hl  = [(r, l) for r, l in zip(residuals, hl_labels)
                     if l is not None and np.isfinite(r)]
        corr = {}
        if valid_hl:
            r_arr, l_arr = map(np.array, zip(*valid_hl))
            corr = residual_hallucination_correlation(r_arr, l_arr)

        # --- Source-ambiguity analysis ---
        ambig = source_ambiguity_analysis(
            residuals, hl_labels, dist_out["depth_noise"]
        )

        result = EvalResult(dataset="3DSRBench", model_tag=bl_tag)
        result.mae          = compute_mae(dist_out["pred_gnn"], dist_out["gt"])
        result.circular_acc = qa_out["circular_acc"]
        result.flip_acc     = qa_out["flip_acc"]
        result.mean_residual = float(np.nanmean(residuals))
        result.residual_hallucination_pearson = corr.get("pearson_r", float("nan"))
        result.residual_hallucination_auroc   = corr.get("auroc",     float("nan"))
        result.extras["baseline_mae"]    = compute_mae(dist_out["pred_baseline"], dist_out["gt"])
        result.extras["source_ambiguity"]= str(ambig)

        if valid_hl:
            r_arr, l_arr = map(np.array, zip(*valid_hl))
            trig = trigger_precision_recall_f1(r_arr, l_arr, args.epsilon)
            result.trigger_precision = trig["precision"]
            result.trigger_recall    = trig["recall"]
            result.trigger_f1        = trig["f1"]

        print_result(result)
        logger.info(f"\n  Source-ambiguity analysis:")
        for k, v in ambig.items():
            logger.info(f"    {k}: {v}")

        all_results[bl_tag] = {
            "summary": result.to_dict(),
            "source_ambiguity": ambig,
        }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
