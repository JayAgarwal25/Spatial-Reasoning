"""
infer.py
--------
Full inference pipeline: Steps 1 → 2 → 3.

Step 1  — Scene graph extraction (object detection + depth + relation parsing)
Step 2  — QuantEpiGNN: predicts spatial predicates + metric distances +
          per-edge geometric consistency residuals
Step 3  — Visual grounding feedback loop: flags high-residual edges,
          annotates the image (bboxes + lines), re-runs Step 1 on the
          annotated image, iterates until convergence or budget exhausted

Usage
-----
    python infer.py \\
        --image      path/to/image.jpg \\
        --checkpoint checkpoints/best_full.pt \\
        --device     cuda:1 \\
        --out_dir    results/my_run \\
        --epsilon    0.3 \\
        --max_iters  3

Outputs
-------
    <out_dir>/
        scene_graph.json          — initial Step 1 output
        scene_graph_final.json    — scene graph after Step 3 iterations
        gnn_output.json           — per-edge GNN predictions + residuals
        iter_00_annotated.png     — annotated image from iteration 0
        iter_01_annotated.png     — annotated image from iteration 1 (if any)
        iter_final.png            — final annotated image
        summary.json              — loop stats + convergence info
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# SceneGraph → PyG Data conversion (avoids disk I/O during feedback loop)
# ---------------------------------------------------------------------------

def scene_graph_to_pyg(scene_graph, embed_model=None):
    """
    Convert a step1_scene_graph.schemas.SceneGraph object directly into a
    PyTorch Geometric Data object suitable for QuantEpiGNN.

    Mirrors the logic in scene_graph_to_pyg.py but operates on the in-memory
    SceneGraph dataclass rather than a JSON file.
    """
    import math
    from torch_geometric.data import Data
    from step2_epistemic_gnn.scene_graph_to_pyg import PRED_TO_IDX, SEM_DIM, GT_DIST_KEY

    nodes  = scene_graph.nodes
    edges  = scene_graph.edges
    W      = scene_graph.image_width
    H      = scene_graph.image_height
    N      = len(nodes)
    image_diag = math.sqrt(W**2 + H**2)

    # ---- node_sem ----
    if embed_model is not None and N > 0:
        labels   = [n.label for n in nodes]
        node_sem = torch.tensor(
            embed_model.encode(labels, show_progress_bar=False),
            dtype=torch.float32,
        )
    else:
        node_sem = torch.zeros(N, SEM_DIM)

    # ---- node_bbox: [x1,y1,x2,y2] → [x,y,w,h] ----
    bboxes = []
    for n in nodes:
        x1, y1, x2, y2 = n.bbox[:4]
        bboxes.append([x1, y1, x2 - x1, y2 - y1])
    node_bbox = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros(N, 4)

    # ---- node_depth ----
    node_depth = torch.tensor(
        [[n.depth if n.depth is not None else 0.0] for n in nodes],
        dtype=torch.float32,
    )

    id_to_idx = {n.id: i for i, n in enumerate(nodes)}

    # Deduplicate directed edges (keep first per pair)
    seen: set = set()
    deduped = []
    for e in edges:
        key = (e.subject_id, e.object_id)
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    if not deduped:
        return Data(
            node_sem=node_sem, node_bbox=node_bbox, node_depth=node_depth,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_dist=torch.zeros(0, 1), edge_conf=torch.zeros(0, 1),
            edge_angle=torch.zeros(0, 1), edge_depth_diff=torch.zeros(0, 1),
            num_nodes=N,
        )

    src_list, dst_list = [], []
    dist_list, conf_list, angle_list, ddiff_list = [], [], [], []

    for e in deduped:
        if e.subject_id not in id_to_idx or e.object_id not in id_to_idx:
            continue
        i = id_to_idx[e.subject_id]
        j = id_to_idx[e.object_id]
        feats = e.features or {}

        src_list.append(i)
        dst_list.append(j)

        if GT_DIST_KEY in feats:
            dist_list.append(float(feats[GT_DIST_KEY]))
        else:
            cd = feats.get("center_distance", 0.0)
            dist_list.append(cd / image_diag if image_diag > 0 else 0.0)

        conf_list.append(float(e.confidence))
        angle_list.append(float(feats.get("angle_rad", 0.0)))
        depth_diff = feats.get("depth_diff")
        ddiff_list.append(float(depth_diff) if depth_diff is not None else 0.0)

    return Data(
        node_sem       = node_sem,
        node_bbox      = node_bbox,
        node_depth     = node_depth,
        edge_index     = torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_dist      = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1),
        edge_conf      = torch.tensor(conf_list, dtype=torch.float32).unsqueeze(1),
        edge_angle     = torch.tensor(angle_list, dtype=torch.float32).unsqueeze(1),
        edge_depth_diff= torch.tensor(ddiff_list, dtype=torch.float32).unsqueeze(1),
        num_nodes      = N,
    )


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    from step2_epistemic_gnn.ablation_gnn import AblationGNN

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("args", ckpt.get("config", {}))
    model = AblationGNN(
        sem_dim             = cfg.get("sem_dim",          384),
        hidden_dim          = cfg.get("hidden_dim",       256),
        num_pred_classes    = cfg.get("num_pred_classes", 14),
        use_geom_constraint = not cfg.get("no_geom_constraint", False),
        use_epistemic       = not cfg.get("no_epistemic",       False),
    )
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
    model.to(device).eval()
    variant = ckpt.get("args", {}).get("variant", checkpoint_path)
    logger.info(f"Loaded model '{variant}' from {checkpoint_path} (device={device})")
    return model


# ---------------------------------------------------------------------------
# Step 1 wrapper (callable for feedback loop)
# ---------------------------------------------------------------------------

def make_run_step1(
    detector_backend: str,
    depth_backend: str,
    relation_backend: str,
    device: str,
    embed_model,
    min_confidence: float = 0.12,
    edge_threshold: float = 0.48,
    max_pairs: int = 24,
    max_objects: int = 20,
    owlvit_labels=None,
):
    """
    Returns a callable that runs Step 1 on a numpy BGR image.
    The callable saves the image to a temp file, calls run_pipeline, and
    returns both the SceneGraph and a PyG Data object.
    """
    from step1_scene_graph.run_pipeline import run_pipeline

    def run_step1(image_bgr: np.ndarray, **kwargs):
        # Write the annotated image to a temp file so run_pipeline can read it
        tmp_img  = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_viz  = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            cv2.imwrite(tmp_img.name, image_bgr)
            run_pipeline(
                image_path       = tmp_img.name,
                json_output_path = tmp_json.name,
                viz_output_path  = tmp_viz.name,
                detector_backend = detector_backend,
                depth_backend    = depth_backend,
                relation_backend = relation_backend,
                use_vlm_relations= (relation_backend not in ("heuristic",)),
                device           = device,
                min_confidence   = min_confidence,
                edge_threshold   = edge_threshold,
                max_pairs        = max_pairs,
                max_objects      = max_objects,
                owlvit_labels    = owlvit_labels,
                gdino_text_prompt= None,
                florence2_task_prompt = "<OD>",
                annotation_path  = None,
            )
            # Reload the SceneGraph from JSON
            from step1_scene_graph.graph_builder import build_scene_graph
            from step1_scene_graph.schemas import SceneGraph, ObjectNode, RelationEdge
            with open(tmp_json.name) as f:
                sg_dict = json.load(f)
            nodes = [ObjectNode(**n) for n in sg_dict["nodes"]]
            edges = [RelationEdge(**e) for e in sg_dict["edges"]]
            scene_graph = SceneGraph(
                image_path   = sg_dict["image_path"],
                image_width  = sg_dict["image_width"],
                image_height = sg_dict["image_height"],
                nodes        = nodes,
                edges        = edges,
                stats        = sg_dict.get("stats", {}),
            )
        finally:
            for f in (tmp_img, tmp_json, tmp_viz):
                try:
                    os.unlink(f.name)
                except OSError:
                    pass

        return scene_graph

    return run_step1


# ---------------------------------------------------------------------------
# Step 2 wrapper (callable for feedback loop)
# ---------------------------------------------------------------------------

def make_run_step2(model, device: torch.device, embed_model):
    """Returns a callable that runs Step 2 (GNN) on a SceneGraph."""

    def run_step2(scene_graph):
        pyg_data = scene_graph_to_pyg(scene_graph, embed_model=embed_model)
        pyg_data = pyg_data.to(device)
        with torch.no_grad():
            out = model(pyg_data)
        return out

    return run_step2


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------

def gnn_output_to_dict(gnn_out: Dict[str, torch.Tensor], scene_graph) -> Dict:
    """Convert GNN tensor output to JSON-serialisable dict."""
    from step2_epistemic_gnn.scene_graph_to_pyg import PREDICATE_VOCAB

    edges = scene_graph.edges
    pred_classes = gnn_out["pred_classes"].cpu().tolist()
    pred_dist    = gnn_out["pred_dist"].squeeze(1).cpu().tolist()
    residuals    = gnn_out["residuals"].squeeze(1).cpu().tolist() \
                   if gnn_out["residuals"].dim() > 1 \
                   else gnn_out["residuals"].cpu().tolist()

    result = []
    for i, edge in enumerate(edges):
        if i >= len(pred_classes):
            break
        result.append({
            "edge_idx":     i,
            "subject_id":   edge.subject_id,
            "object_id":    edge.object_id,
            "pred_class":   PREDICATE_VOCAB[pred_classes[i]] if pred_classes[i] < len(PREDICATE_VOCAB) else "unknown",
            "pred_dist_m":  float(pred_dist[i]) if i < len(pred_dist) else None,
            "residual":     float(residuals[i]) if i < len(residuals) else None,
            "gt_predicate": edge.predicate,
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full inference: Step 1 → Step 2 → Step 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",       required=True, help="Input image path")
    p.add_argument("--checkpoint",  required=True, help="GNN checkpoint (.pt)")
    p.add_argument("--out_dir",     default="results/infer", help="Output directory")
    p.add_argument("--device",      default=None,
                   help="CUDA device (e.g. cuda:1) — auto-detected if not set")

    p.add_argument("--detector_backend", default="contour",
                   choices=["annotation", "contour", "owlvit", "detr",
                            "groundingdino", "florence2"])
    p.add_argument("--depth_backend", default="pseudo",
                   choices=["none", "pseudo", "dpt"])
    p.add_argument("--relation_backend", default="heuristic",
                   choices=["heuristic", "blip2", "llava"])
    p.add_argument("--owlvit_labels", type=str, default=None,
                   help="CSV of class labels for OWL-ViT detector")

    p.add_argument("--epsilon",    type=float, default=0.3,
                   help="Residual threshold for Step 3 trigger")
    p.add_argument("--max_iters",  type=int,   default=3,
                   help="Max Step 3 feedback iterations")
    p.add_argument("--skip_step3", action="store_true",
                   help="Skip feedback loop (Step 1+2 only)")
    p.add_argument("--no_embed",   action="store_true",
                   help="Skip sentence-transformer embedding (zero vectors)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load GNN model ----
    model = load_model(args.checkpoint, device)

    # ---- Load sentence-transformer (optional) ----
    embed_model = None
    if not args.no_embed:
        try:
            from step2_epistemic_gnn.scene_graph_to_pyg import load_embed_model
            embed_model = load_embed_model()
            logger.info("Loaded sentence-transformer for node embeddings.")
        except Exception as e:
            logger.warning(f"Could not load embed model: {e} — using zero vectors.")

    # ---- Step 1: initial scene graph ----
    logger.info(f"=== Step 1: extracting scene graph from {args.image} ===")
    from step1_scene_graph.run_pipeline import run_pipeline
    from step1_scene_graph.schemas import SceneGraph, ObjectNode, RelationEdge

    sg_json_path = os.path.join(args.out_dir, "scene_graph.json")
    sg_viz_path  = os.path.join(args.out_dir, "scene_graph_viz.png")
    owlvit_labels = [x.strip() for x in args.owlvit_labels.split(",")
                     if x.strip()] if args.owlvit_labels else None

    run_pipeline(
        image_path       = args.image,
        json_output_path = sg_json_path,
        viz_output_path  = sg_viz_path,
        detector_backend = args.detector_backend,
        depth_backend    = args.depth_backend,
        relation_backend = args.relation_backend,
        use_vlm_relations= (args.relation_backend not in ("heuristic",)),
        device           = str(device),
        min_confidence   = 0.12,
        edge_threshold   = 0.48,
        max_pairs        = 24,
        max_objects      = 20,
        owlvit_labels    = owlvit_labels,
        gdino_text_prompt= None,
        florence2_task_prompt= "<OD>",
        annotation_path  = None,
    )

    with open(sg_json_path) as f:
        sg_dict = json.load(f)

    nodes = [ObjectNode(**n) for n in sg_dict["nodes"]]
    edges = [RelationEdge(**e) for e in sg_dict["edges"]]
    scene_graph = SceneGraph(
        image_path   = sg_dict["image_path"],
        image_width  = sg_dict["image_width"],
        image_height = sg_dict["image_height"],
        nodes        = nodes,
        edges        = edges,
        stats        = sg_dict.get("stats", {}),
    )
    logger.info(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    # ---- Step 2: GNN forward pass ----
    logger.info("=== Step 2: running QuantEpiGNN ===")
    run_step2 = make_run_step2(model, device, embed_model)
    gnn_output = run_step2(scene_graph)

    gnn_json_path = os.path.join(args.out_dir, "gnn_output.json")
    with open(gnn_json_path, "w") as f:
        json.dump(gnn_output_to_dict(gnn_output, scene_graph), f, indent=2)
    logger.info(f"  GNN output saved → {gnn_json_path}")

    residuals = gnn_output["residuals"].flatten()
    if residuals.numel() > 0:
        logger.info(
            f"  Residuals: max={residuals.max().item():.4f}  "
            f"mean={residuals.mean().item():.4f}  "
            f"flagged (>{args.epsilon}): {(residuals > args.epsilon).sum().item()}"
        )
    else:
        logger.info("  No edges — skipping residual stats.")

    if args.skip_step3:
        logger.info("Step 3 skipped (--skip_step3).")
        _write_summary(args, scene_graph, gnn_output, loop_result=None)
        return

    # ---- Step 3: feedback loop ----
    logger.info(f"=== Step 3: visual grounding feedback loop (max_iters={args.max_iters}) ===")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")

    run_step1 = make_run_step1(
        detector_backend = args.detector_backend,
        depth_backend    = args.depth_backend,
        relation_backend = args.relation_backend,
        device           = str(device),
        embed_model      = embed_model,
        owlvit_labels    = owlvit_labels,
    )

    from step3_visual_agent.feedback_loop import run_feedback_loop, save_loop_images
    loop_result = run_feedback_loop(
        image       = image_bgr,
        scene_graph = scene_graph,
        gnn_output  = gnn_output,
        run_step1   = run_step1,
        run_step2   = run_step2,
        epsilon     = args.epsilon,
        max_iters   = args.max_iters,
    )

    # Save all iteration images
    save_loop_images(loop_result, output_dir=args.out_dir)

    # Save final scene graph
    final_sg_path = os.path.join(args.out_dir, "scene_graph_final.json")
    with open(final_sg_path, "w") as f:
        json.dump(loop_result.final_scene_graph.to_dict(), f, indent=2)

    # Save final GNN output
    final_gnn_path = os.path.join(args.out_dir, "gnn_output_final.json")
    with open(final_gnn_path, "w") as f:
        json.dump(
            gnn_output_to_dict(loop_result.final_gnn_output, loop_result.final_scene_graph),
            f, indent=2,
        )

    logger.info(
        f"Step 3 complete — converged={loop_result.converged}  "
        f"iterations={loop_result.iterations_run}"
    )

    _write_summary(args, scene_graph, gnn_output, loop_result)


def _write_summary(args, initial_sg, initial_gnn_out, loop_result):
    summary = {
        "image":                args.image,
        "checkpoint":           args.checkpoint,
        "epsilon":              args.epsilon,
        "initial_num_nodes":    len(initial_sg.nodes),
        "initial_num_edges":    len(initial_sg.edges),
        "initial_residual_max": float(initial_gnn_out["residuals"].max().item()),
        "initial_residual_mean":float(initial_gnn_out["residuals"].mean().item()),
    }
    if loop_result is not None:
        final_r = loop_result.final_gnn_output["residuals"].flatten()
        summary.update({
            "converged":             loop_result.converged,
            "iterations_run":        loop_result.iterations_run,
            "final_residual_max":    float(final_r.max().item()),
            "final_residual_mean":   float(final_r.mean().item()),
            "final_num_nodes":       len(loop_result.final_scene_graph.nodes),
            "final_num_edges":       len(loop_result.final_scene_graph.edges),
        })

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
