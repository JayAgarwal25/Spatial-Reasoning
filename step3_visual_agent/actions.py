"""
Step 3 — Visual Grounding Actions

Typed action primitives used by the Visual Grounding Agent to annotate an image
before it is fed back into Step 1 for regrounding.

Actions
-------
draw_bbox   : Draw a bounding box around a node, labelled with its id + label.
              Used to reground a node that is party to a high-residual edge.

draw_line   : Draw a line between two node centroids, annotated with the VLM's
              predicted distance and the consistency residual.
              Used to make the geometry of a flagged edge visually explicit.

depth_crop  : Crop the image to a node's bounding box and return it as a
              separate array, to be re-queried by the VLM for a fresh depth /
              distance estimate on the flagged edge.

All actions are pure functions — they do not mutate the input image.
They return a new numpy array (uint8 BGR, OpenCV convention).

ActionResult
------------
Returned by every action. Carries the annotated image plus metadata so the
feedback loop in feedback_loop.py can log exactly what was done and pass
the image back to Step 1.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from step1_scene_graph.schemas import ObjectNode, RelationEdge


# ---------------------------------------------------------------------------
# Colour palette — one colour per action type for visual clarity
# ---------------------------------------------------------------------------

_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "bbox":       (0,   200, 50),    # green
    "line":       (0,   120, 255),   # orange
    "depth_crop": (255, 60,  60),    # blue-ish red (OpenCV BGR)
    "text_bg":    (20,  20,  20),    # near-black background for labels
}

_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE   = 0.55
_FONT_THICKNESS = 1
_BOX_THICKNESS  = 2
_LINE_THICKNESS = 2


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """
    Returned by every action function.

    Attributes
    ----------
    image        : annotated image (uint8 BGR numpy array, same size as input
                   unless action == "depth_crop").
    action_type  : one of "draw_bbox" | "draw_line" | "depth_crop"
    edge_index   : index of the flagged edge that triggered this action.
    metadata     : action-specific payload for logging and Step 1 re-ingestion.
    """
    image:       np.ndarray
    action_type: str
    edge_index:  int
    metadata:    Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_label(
    img:   np.ndarray,
    text:  str,
    x:     int,
    y:     int,
    colour: Tuple[int, int, int],
) -> np.ndarray:
    """
    Draw `text` at (x, y) with a dark background rectangle for readability.
    Returns a copy of img.
    """
    img = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    # background rectangle
    cv2.rectangle(
        img,
        (x, y - th - baseline - 2),
        (x + tw + 4, y + baseline),
        _COLOURS["text_bg"],
        cv2.FILLED,
    )
    cv2.putText(img, text, (x + 2, y), _FONT, _FONT_SCALE, colour, _FONT_THICKNESS, cv2.LINE_AA)
    return img


def _node_bbox_ints(node: ObjectNode) -> Tuple[int, int, int, int]:
    """Return bbox as (x1, y1, x2, y2) integer pixel coords."""
    x, y, w, h = node.bbox
    return int(x), int(y), int(x + w), int(y + h)


def _node_centre_ints(node: ObjectNode) -> Tuple[int, int]:
    cx, cy = node.center
    return int(cx), int(cy)


# ---------------------------------------------------------------------------
# Action 1: draw_bbox
# ---------------------------------------------------------------------------

def draw_bbox(
    image:      np.ndarray,
    node:       ObjectNode,
    edge_index: int,
    residual:   float,
    colour:     Optional[Tuple[int, int, int]] = None,
) -> ActionResult:
    """
    Draw a bounding box around `node` with a label showing its id, label,
    and the residual of the edge that caused it to be flagged.

    Parameters
    ----------
    image      : input image (uint8 BGR numpy array). Not mutated.
    node       : the ObjectNode to highlight.
    edge_index : index of the flagged edge (for ActionResult metadata).
    residual   : consistency residual r of the flagged edge (for label).
    colour     : override box colour (defaults to _COLOURS["bbox"]).

    Returns
    -------
    ActionResult with annotated image and metadata.
    """
    img = image.copy()
    col = colour or _COLOURS["bbox"]
    x1, y1, x2, y2 = _node_bbox_ints(node)

    cv2.rectangle(img, (x1, y1), (x2, y2), col, _BOX_THICKNESS)

    label = f"[{node.id}] {node.label}  r={residual:.3f}"
    img = _put_label(img, label, x1, y1 - 4, col)

    return ActionResult(
        image=img,
        action_type="draw_bbox",
        edge_index=edge_index,
        metadata={
            "node_id":   node.id,
            "node_label": node.label,
            "bbox":      [x1, y1, x2, y2],
            "residual":  residual,
        },
    )


# ---------------------------------------------------------------------------
# Action 2: draw_line
# ---------------------------------------------------------------------------

def draw_line(
    image:       np.ndarray,
    src_node:    ObjectNode,
    dst_node:    ObjectNode,
    edge:        RelationEdge,
    edge_index:  int,
    residual:    float,
    pred_dist:   float,
    colour:      Optional[Tuple[int, int, int]] = None,
) -> ActionResult:
    """
    Draw a line between the centroids of `src_node` and `dst_node`, annotated
    with the predicate, VLM-predicted distance, and consistency residual.

    This makes the geometric inconsistency spatially explicit so that when the
    annotated image is fed back to the VLM, the problematic edge is highlighted.

    Parameters
    ----------
    image      : input image (uint8 BGR). Not mutated.
    src_node   : subject node of the edge.
    dst_node   : object node of the edge.
    edge       : RelationEdge — carries predicate and confidence.
    edge_index : index of this edge in the scene graph edge list.
    residual   : consistency residual r for this edge.
    pred_dist  : GNN-refined metric distance prediction (metres).
    colour     : override line colour.

    Returns
    -------
    ActionResult with annotated image and metadata.
    """
    img = image.copy()
    col = colour or _COLOURS["line"]

    cx_src, cy_src = _node_centre_ints(src_node)
    cx_dst, cy_dst = _node_centre_ints(dst_node)

    # Draw endpoint markers
    cv2.circle(img, (cx_src, cy_src), 5, col, -1)
    cv2.circle(img, (cx_dst, cy_dst), 5, col, -1)

    # Draw line
    cv2.line(img, (cx_src, cy_src), (cx_dst, cy_dst), col, _LINE_THICKNESS, cv2.LINE_AA)

    # Label at midpoint
    mx = (cx_src + cx_dst) // 2
    my = (cy_src + cy_dst) // 2
    label = f"{edge.predicate}  d={pred_dist:.2f}m  r={residual:.3f}"
    img = _put_label(img, label, mx, my, col)

    # Also draw bboxes for both nodes at reduced opacity via weighted blend
    overlay = img.copy()
    x1s, y1s, x2s, y2s = _node_bbox_ints(src_node)
    x1d, y1d, x2d, y2d = _node_bbox_ints(dst_node)
    cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), col, 1)
    cv2.rectangle(overlay, (x1d, y1d), (x2d, y2d), col, 1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    return ActionResult(
        image=img,
        action_type="draw_line",
        edge_index=edge_index,
        metadata={
            "src_node_id":  src_node.id,
            "dst_node_id":  dst_node.id,
            "predicate":    edge.predicate,
            "pred_dist_m":  pred_dist,
            "residual":     residual,
            "confidence":   edge.confidence,
            "src_centre":   [cx_src, cy_src],
            "dst_centre":   [cx_dst, cy_dst],
        },
    )


# ---------------------------------------------------------------------------
# Action 3: depth_crop
# ---------------------------------------------------------------------------

def depth_crop(
    image:      np.ndarray,
    node:       ObjectNode,
    edge_index: int,
    residual:   float,
    pad_px:     int = 16,
) -> ActionResult:
    """
    Crop the image to the bounding box of `node` (with optional padding) and
    return it as a separate image.

    The crop is passed back to the VLM for a fresh depth / distance estimate
    on the flagged edge, giving the grounding agent a zoomed-in view of the
    specific region that caused the inconsistency.

    Parameters
    ----------
    image      : input image (uint8 BGR). Not mutated.
    node       : node whose bounding box defines the crop region.
    edge_index : index of the flagged edge.
    residual   : consistency residual r (stored in metadata).
    pad_px     : pixel padding added around the bbox (clamped to image bounds).

    Returns
    -------
    ActionResult whose .image is the cropped sub-image (may be smaller than
    the original). metadata includes the crop rect and original image size for
    coordinate remapping in Step 1.
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = _node_bbox_ints(node)

    # Apply padding, clamp to image bounds
    x1c = max(0, x1 - pad_px)
    y1c = max(0, y1 - pad_px)
    x2c = min(W, x2 + pad_px)
    y2c = min(H, y2 + pad_px)

    crop = image[y1c:y2c, x1c:x2c].copy()

    # Draw a border on the crop to indicate it's a regrounding crop
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1),
                  _COLOURS["depth_crop"], 2)

    label = f"crop [{node.id}] {node.label}  r={residual:.3f}"
    crop = _put_label(crop, label, 4, 16, _COLOURS["depth_crop"])

    return ActionResult(
        image=crop,
        action_type="depth_crop",
        edge_index=edge_index,
        metadata={
            "node_id":         node.id,
            "node_label":      node.label,
            "crop_rect":       [x1c, y1c, x2c, y2c],
            "original_size":   [W, H],
            "pad_px":          pad_px,
            "residual":        residual,
        },
    )


# ---------------------------------------------------------------------------
# Composite: apply all three actions for a single flagged edge
# ---------------------------------------------------------------------------

def ground_flagged_edge(
    image:      np.ndarray,
    edge:       RelationEdge,
    edge_index: int,
    src_node:   ObjectNode,
    dst_node:   ObjectNode,
    residual:   float,
    pred_dist:  float,
) -> List[ActionResult]:
    """
    Apply all three grounding actions for one flagged edge and return
    them as a list.

    Execution order:
        1. draw_bbox on src_node
        2. draw_bbox on dst_node   (applied to the result of step 1)
        3. draw_line between them  (applied to the result of step 2)
        4. depth_crop of src_node  (from original image — independent)
        5. depth_crop of dst_node  (from original image — independent)

    Steps 1-3 produce a single accumulated annotated image for Step 1 re-ingestion.
    Steps 4-5 produce separate crops for targeted VLM depth re-query.

    Returns
    -------
    List[ActionResult] — caller (feedback_loop.py) decides which images to
    pass back to Step 1 and which to pass to the VLM crop-requery pathway.
    """
    results: List[ActionResult] = []

    # Accumulate bbox annotations on one canvas
    r1 = draw_bbox(image,       src_node, edge_index, residual)
    r2 = draw_bbox(r1.image,    dst_node, edge_index, residual)
    r3 = draw_line(r2.image, src_node, dst_node, edge, edge_index, residual, pred_dist)

    results.append(r1)
    results.append(r2)
    results.append(r3)   # r3.image is the fully annotated image for Step 1

    # Independent depth crops from the original image
    results.append(depth_crop(image, src_node, edge_index, residual))
    results.append(depth_crop(image, dst_node, edge_index, residual))

    return results
