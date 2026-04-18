"""
Step 3 — Trigger: Residual Thresholding

Reads the `residuals` tensor from Step 2's QuantEpiGNN output dict and decides
which edges need visual regrounding.

Interface contract
------------------
Input:
    gnn_output  : dict returned by QuantEpiGNN.forward()
                  Must contain:
                    "residuals"    (E, 1)  FloatTensor
                    "pred_classes" (E,)    LongTensor
                    "pred_dist"    (E, 1)  FloatTensor
    scene_graph : SceneGraph (from schemas.py)
    epsilon     : float — regrounding threshold.
                  Edges with r > epsilon are flagged.
                  Calibrate per dataset on a validation precision-recall curve;
                  0.3 is a reasonable starting point (see README).

Output:
    TriggerResult dataclass:
        flagged_edge_indices : List[int]    — indices into edge_index / RelationEdge list
        flagged_edges        : List[RelationEdge]
        residuals            : FloatTensor (E, 1)
        needs_regrounding    : bool         — False when no edge exceeds epsilon
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import torch

from step1_scene_graph.schemas import RelationEdge, SceneGraph


@dataclass
class TriggerResult:
    """
    Output of residual_trigger().

    Attributes
    ----------
    flagged_edge_indices : indices into the original edge list (and edge_index)
                           that exceeded the threshold.
    flagged_edges        : the corresponding RelationEdge objects for easy
                           downstream access (subject_id, object_id, predicate).
    residuals            : full (E, 1) residual tensor from Step 2 — kept here
                           so Step 3 actions can read per-edge r values directly.
    needs_regrounding    : True iff at least one edge was flagged.
    epsilon              : the threshold value used (stored for logging / ablations).
    """
    flagged_edge_indices: List[int]
    flagged_edges:        List[RelationEdge]
    residuals:            torch.Tensor          # (E, 1)
    needs_regrounding:    bool
    epsilon:              float
    stats:                Dict[str, Any] = field(default_factory=dict)


def residual_trigger(
    gnn_output:   Dict[str, torch.Tensor],
    scene_graph:  SceneGraph,
    epsilon:      float = 0.3,
) -> TriggerResult:
    """
    Threshold the geometric consistency residuals from Step 2 and identify
    which edges require targeted visual regrounding in Step 3.

    Parameters
    ----------
    gnn_output   : output dict from QuantEpiGNN.forward().
                   Must contain "residuals" (E, 1).
    scene_graph  : SceneGraph produced by Step 1. Its `.edges` list must be
                   index-aligned with the GNN's edge_index (same order).
    epsilon      : regrounding threshold. Edges with r > epsilon are flagged.

    Returns
    -------
    TriggerResult
        .flagged_edge_indices  — list of integer indices for flagged edges
        .flagged_edges         — corresponding RelationEdge objects
        .residuals             — full (E, 1) residual tensor
        .needs_regrounding     — True if any edge was flagged
        .epsilon               — stored threshold value
        .stats                 — summary stats for logging

    Raises
    ------
    KeyError  if "residuals" is absent from gnn_output.
    ValueError if edge count mismatches between gnn_output and scene_graph.
    """
    if "residuals" not in gnn_output:
        raise KeyError(
            "gnn_output must contain 'residuals'. "
            "Ensure QuantEpiGNN.forward() was called and its output passed here."
        )

    residuals: torch.Tensor = gnn_output["residuals"]   # (E, 1)
    E_gnn = residuals.size(0)
    E_sg  = len(scene_graph.edges)

    if E_gnn != E_sg:
        raise ValueError(
            f"Edge count mismatch: GNN produced residuals for {E_gnn} edges "
            f"but SceneGraph has {E_sg} edges. "
            "Ensure edge_index in Step 2 was built from the same SceneGraph."
        )

    r_flat = residuals.squeeze(1)                        # (E,)
    flag_mask = r_flat > epsilon                         # (E,) bool

    flagged_edge_indices: List[int] = flag_mask.nonzero(as_tuple=True)[0].tolist()
    flagged_edges: List[RelationEdge] = [
        scene_graph.edges[i] for i in flagged_edge_indices
    ]

    stats = {
        "num_edges":            E_gnn,
        "num_flagged":          len(flagged_edge_indices),
        "flagged_fraction":     len(flagged_edge_indices) / max(E_gnn, 1),
        "residual_mean":        r_flat.mean().item(),
        "residual_max":         r_flat.max().item(),
        "residual_min":         r_flat.min().item(),
        "epsilon":              epsilon,
    }

    return TriggerResult(
        flagged_edge_indices=flagged_edge_indices,
        flagged_edges=flagged_edges,
        residuals=residuals,
        needs_regrounding=bool(flag_mask.any()),
        epsilon=epsilon,
        stats=stats,
    )


def rank_flagged_edges(trigger_result: TriggerResult) -> List[int]:
    """
    Return flagged_edge_indices sorted by descending residual magnitude.

    Useful when the feedback loop has a budget (max_iters) and should
    prioritise the most inconsistent edges first.

    Parameters
    ----------
    trigger_result : output of residual_trigger()

    Returns
    -------
    List of flagged edge indices sorted by r descending.
    """
    if not trigger_result.flagged_edge_indices:
        return []

    r_flat = trigger_result.residuals.squeeze(1)
    flagged_r = [
        (idx, r_flat[idx].item())
        for idx in trigger_result.flagged_edge_indices
    ]
    flagged_r.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in flagged_r]
