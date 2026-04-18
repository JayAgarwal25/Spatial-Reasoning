"""
Step 3 — Feedback Loop: Recursive Iteration Manager

Orchestrates the full regrounding cycle:

    Step 1 → Step 2 → [Step 3: trigger → actions → annotated image] → Step 1 → …

The loop runs until either:
  (a) no edge residual exceeds epsilon  (convergence), or
  (b) max_iters is reached              (budget exhausted).

Interface contract
------------------
The loop requires two callable hooks that the user supplies:

    run_step1(image: np.ndarray, **kwargs) -> SceneGraph
        Re-runs scene graph extraction on an annotated image.
        This is a thin wrapper around the Step 1 pipeline.

    run_step2(scene_graph: SceneGraph) -> dict
        Re-runs the GNN on the updated scene graph and returns
        QuantEpiGNN's output dict (with "residuals").

These are injected rather than imported to keep Step 3 decoupled from
Step 1 and Step 2 implementation details.

LoopResult
----------
Returned after the loop terminates. Contains the final scene graph, the
final GNN output, a full per-iteration history, and a convergence flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

from step1_scene_graph.schemas import ObjectNode, RelationEdge, SceneGraph
from step3_visual_agent.trigger import TriggerResult, residual_trigger, rank_flagged_edges
from step3_visual_agent.actions import ActionResult, ground_flagged_edge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-iteration record
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """
    Snapshot of a single feedback loop iteration.

    Attributes
    ----------
    iteration          : 0-based iteration index.
    trigger_result     : TriggerResult from this iteration's threshold check.
    action_results     : all ActionResults produced in this iteration
                         (one list per flagged edge, flattened).
    annotated_image    : the image passed back to Step 1 — the draw_line
                         composite for all flagged edges applied in sequence.
    residual_max       : max residual in this iteration (for convergence tracking).
    residual_mean      : mean residual in this iteration.
    num_flagged        : number of edges flagged in this iteration.
    """
    iteration:       int
    trigger_result:  TriggerResult
    action_results:  List[ActionResult]
    annotated_image: np.ndarray
    residual_max:    float
    residual_mean:   float
    num_flagged:     int


# ---------------------------------------------------------------------------
# LoopResult
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """
    Final output of run_feedback_loop().

    Attributes
    ----------
    converged          : True if loop exited because needs_regrounding=False.
    iterations_run     : total number of iterations executed (≥ 1).
    final_scene_graph  : SceneGraph after the last Step 1 re-run.
    final_gnn_output   : GNN output dict after the last Step 2 re-run.
    final_trigger      : TriggerResult from the last iteration.
    history            : list of IterationRecord, one per iteration.
    final_image        : the annotated image from the final iteration
                         (original image if loop never ran a visual action).
    """
    converged:          bool
    iterations_run:     int
    final_scene_graph:  SceneGraph
    final_gnn_output:   Dict[str, torch.Tensor]
    final_trigger:      TriggerResult
    history:            List[IterationRecord]
    final_image:        np.ndarray


# ---------------------------------------------------------------------------
# Node lookup helper
# ---------------------------------------------------------------------------

def _build_node_index(scene_graph: SceneGraph) -> Dict[int, ObjectNode]:
    """Return a dict mapping node.id → ObjectNode for O(1) lookup."""
    return {node.id: node for node in scene_graph.nodes}


# ---------------------------------------------------------------------------
# Single-iteration action application
# ---------------------------------------------------------------------------

def _apply_actions_for_iteration(
    image:        np.ndarray,
    trigger:      TriggerResult,
    scene_graph:  SceneGraph,
    gnn_output:   Dict[str, torch.Tensor],
    ranked_indices: List[int],
) -> tuple[np.ndarray, List[ActionResult]]:
    """
    For every flagged edge (in residual-descending order), apply all three
    grounding actions. The draw_line composite image is accumulated — each
    flagged edge's annotations are layered onto the previous result.

    Depth crops are collected separately (not layered) so they can be passed
    to the VLM for targeted re-query.

    Returns
    -------
    annotated_image : composite image with all flagged edges annotated.
    all_results     : flat list of every ActionResult produced.
    """
    node_index = _build_node_index(scene_graph)
    annotated  = image.copy()
    all_results: List[ActionResult] = []

    r_flat     = gnn_output["residuals"].squeeze(1)   # (E,)
    pred_dist  = gnn_output["pred_dist"].squeeze(1)   # (E,)

    for edge_idx in ranked_indices:
        edge: RelationEdge = scene_graph.edges[edge_idx]

        src_node = node_index.get(edge.subject_id)
        dst_node = node_index.get(edge.object_id)

        if src_node is None or dst_node is None:
            logger.warning(
                "Edge %d references unknown node id(s): subject=%d object=%d — skipping.",
                edge_idx, edge.subject_id, edge.object_id,
            )
            continue

        residual  = r_flat[edge_idx].item()
        dist_pred = pred_dist[edge_idx].item()

        results = ground_flagged_edge(
            image      = annotated,   # accumulate onto the growing annotation
            edge       = edge,
            edge_index = edge_idx,
            src_node   = src_node,
            dst_node   = dst_node,
            residual   = residual,
            pred_dist  = dist_pred,
        )

        # r3 (index 2) is the draw_line composite — use as the new base
        if len(results) >= 3:
            annotated = results[2].image

        all_results.extend(results)

        logger.debug(
            "Iter action: edge %d  (%s →%s→ %s)  r=%.4f  pred_dist=%.3fm",
            edge_idx, src_node.label, edge.predicate, dst_node.label,
            residual, dist_pred,
        )

    return annotated, all_results


# ---------------------------------------------------------------------------
# Main feedback loop
# ---------------------------------------------------------------------------

def run_feedback_loop(
    image:          np.ndarray,
    scene_graph:    SceneGraph,
    gnn_output:     Dict[str, torch.Tensor],
    run_step1:      Callable[..., SceneGraph],
    run_step2:      Callable[[SceneGraph], Dict[str, torch.Tensor]],
    epsilon:        float = 0.3,
    max_iters:      int   = 3,
    step1_kwargs:   Optional[Dict[str, Any]] = None,
) -> LoopResult:
    """
    Run the full visual grounding feedback loop.

    Parameters
    ----------
    image        : original input image (uint8 BGR numpy array).
    scene_graph  : SceneGraph from the *initial* Step 1 run.
    gnn_output   : GNN output dict from the *initial* Step 2 run.
    run_step1    : callable — re-runs Step 1 on an annotated image.
                   Signature: run_step1(image: np.ndarray, **step1_kwargs) -> SceneGraph
    run_step2    : callable — re-runs Step 2 on a SceneGraph.
                   Signature: run_step2(scene_graph: SceneGraph) -> dict
    epsilon      : residual threshold for regrounding. Default 0.3.
    max_iters    : maximum number of feedback loop iterations. Default 3.
    step1_kwargs : extra kwargs forwarded to run_step1 (e.g., image_path).

    Returns
    -------
    LoopResult
        .converged         — True if loop exited due to no flagged edges.
        .iterations_run    — how many iterations were executed.
        .final_scene_graph — scene graph after last Step 1 re-run.
        .final_gnn_output  — GNN output after last Step 2 re-run.
        .final_trigger     — TriggerResult from the last check.
        .history           — per-iteration records for ablation / logging.
        .final_image       — last annotated image passed to Step 1.
    """
    if step1_kwargs is None:
        step1_kwargs = {}

    history: List[IterationRecord] = []
    current_image       = image.copy()
    current_scene_graph = scene_graph
    current_gnn_output  = gnn_output
    converged           = False

    for iteration in range(max_iters):
        logger.info("Feedback loop — iteration %d / %d", iteration + 1, max_iters)

        # ---- Trigger -------------------------------------------------------
        trigger = residual_trigger(current_gnn_output, current_scene_graph, epsilon)

        logger.info(
            "  Flagged %d / %d edges  (max_r=%.4f  mean_r=%.4f)",
            trigger.stats["num_flagged"],
            trigger.stats["num_edges"],
            trigger.stats["residual_max"],
            trigger.stats["residual_mean"],
        )

        if not trigger.needs_regrounding:
            logger.info("  No edges exceed epsilon=%.3f — converged.", epsilon)
            converged = True
            # Record a zero-action iteration for completeness
            history.append(IterationRecord(
                iteration       = iteration,
                trigger_result  = trigger,
                action_results  = [],
                annotated_image = current_image,
                residual_max    = trigger.stats["residual_max"],
                residual_mean   = trigger.stats["residual_mean"],
                num_flagged     = 0,
            ))
            break

        # ---- Rank flagged edges by descending residual ---------------------
        ranked = rank_flagged_edges(trigger)

        # ---- Apply visual grounding actions --------------------------------
        annotated_image, action_results = _apply_actions_for_iteration(
            image         = current_image,
            trigger       = trigger,
            scene_graph   = current_scene_graph,
            gnn_output    = current_gnn_output,
            ranked_indices= ranked,
        )

        r_flat = current_gnn_output["residuals"].squeeze(1)

        history.append(IterationRecord(
            iteration       = iteration,
            trigger_result  = trigger,
            action_results  = action_results,
            annotated_image = annotated_image,
            residual_max    = r_flat.max().item(),
            residual_mean   = r_flat.mean().item(),
            num_flagged     = trigger.stats["num_flagged"],
        ))

        # ---- Re-run Step 1 on the annotated image --------------------------
        logger.info("  Re-running Step 1 on annotated image …")
        try:
            current_scene_graph = run_step1(annotated_image, **step1_kwargs)
        except Exception as exc:
            logger.error("Step 1 re-run failed at iteration %d: %s", iteration, exc)
            break

        # ---- Re-run Step 2 on the updated scene graph ----------------------
        logger.info("  Re-running Step 2 on updated scene graph …")
        try:
            current_gnn_output = run_step2(current_scene_graph)
        except Exception as exc:
            logger.error("Step 2 re-run failed at iteration %d: %s", iteration, exc)
            break

        current_image = annotated_image

    else:
        # max_iters exhausted without convergence — run final trigger check
        logger.info("Max iterations (%d) reached — running final trigger check.", max_iters)
        trigger = residual_trigger(current_gnn_output, current_scene_graph, epsilon)
        converged = not trigger.needs_regrounding

    final_trigger = residual_trigger(current_gnn_output, current_scene_graph, epsilon)

    logger.info(
        "Feedback loop complete — converged=%s  iterations_run=%d  "
        "final_max_r=%.4f  final_flagged=%d",
        converged, len(history),
        final_trigger.stats["residual_max"],
        final_trigger.stats["num_flagged"],
    )

    return LoopResult(
        converged          = converged,
        iterations_run     = len(history),
        final_scene_graph  = current_scene_graph,
        final_gnn_output   = current_gnn_output,
        final_trigger      = final_trigger,
        history            = history,
        final_image        = current_image,
    )


# ---------------------------------------------------------------------------
# Convenience: save annotated images from a LoopResult
# ---------------------------------------------------------------------------

def save_loop_images(
    loop_result: LoopResult,
    output_dir:  str,
    prefix:      str = "iter",
) -> List[str]:
    """
    Save the annotated image from each iteration to `output_dir`.

    Returns a list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []

    for rec in loop_result.history:
        fname = os.path.join(output_dir, f"{prefix}_{rec.iteration:02d}_annotated.png")
        cv2.imwrite(fname, rec.annotated_image)
        paths.append(fname)
        logger.info("Saved iteration %d annotated image → %s", rec.iteration, fname)

    final_path = os.path.join(output_dir, f"{prefix}_final.png")
    cv2.imwrite(final_path, loop_result.final_image)
    paths.append(final_path)
    logger.info("Saved final image → %s", final_path)

    return paths
