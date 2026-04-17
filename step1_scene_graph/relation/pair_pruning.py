from __future__ import annotations
import math
from typing import Dict, List, Tuple
from step1_scene_graph.schemas import ObjectNode
from step1_scene_graph.utils.image_utils import image_diagonal

SEMANTIC_PRIORS = {
    ('pillow', 'sofa'), ('cushion', 'sofa'), ('chair', 'table'), ('coffee table', 'sofa'),
    ('plant', 'window'), ('lamp', 'table'), ('rug', 'table'), ('table', 'rug'),
    ('book', 'bookshelf'), ('bookshelf', 'book'), ('chair', 'desk'), ('monitor', 'desk'),
}


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def contains(box_a, box_b, tol: float = 0.0) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 - tol <= bx1 and ay1 - tol <= by1 and ax2 + tol >= bx2 and ay2 + tol >= by2


def center_distance(a: ObjectNode, b: ObjectNode) -> float:
    dx = a.center[0] - b.center[0]
    dy = a.center[1] - b.center[1]
    return math.sqrt(dx * dx + dy * dy)


def semantic_compatibility(a: ObjectNode, b: ObjectNode) -> float:
    key = (a.label.lower(), b.label.lower())
    rev = (b.label.lower(), a.label.lower())
    if key in SEMANTIC_PRIORS or rev in SEMANTIC_PRIORS:
        return 1.0
    # generic furniture prior
    furniture = {'chair', 'table', 'desk', 'sofa', 'couch', 'lamp', 'rug', 'bookshelf', 'window', 'plant'}
    if a.label.lower() in furniture and b.label.lower() in furniture:
        return 0.6
    return 0.2


def prune_pairs(nodes: List[ObjectNode], image_w: int, image_h: int, max_pairs: int = 24) -> List[Tuple[ObjectNode, ObjectNode, Dict]]:
    diag = image_diagonal(image_w, image_h)
    candidates = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            a, b = nodes[i], nodes[j]
            iou = bbox_iou(a.bbox, b.bbox)
            dist = center_distance(a, b)
            dist_norm = dist / max(diag, 1e-6)
            contain = contains(a.bbox, b.bbox, tol=6.0) or contains(b.bbox, a.bbox, tol=6.0)
            depth_close = (a.depth is not None and b.depth is not None and abs(a.depth - b.depth) < 0.18 * max(abs(a.depth), abs(b.depth), 1.0))
            size_ratio = min(a.area, b.area) / max(a.area, b.area, 1.0)
            sem = semantic_compatibility(a, b)
            strong = int(iou > 0.08) + int(contain) + int(dist_norm < 0.28) + int(depth_close)
            weak = int(sem >= 0.6) + int(size_ratio > 0.04)
            if strong >= 1 and weak >= 1:
                score = 1.5 * iou + 1.2 * (1 - dist_norm) + 0.9 * sem + 0.5 * size_ratio + (0.4 if contain else 0.0)
                candidates.append((a, b, {
                    'iou': iou,
                    'center_distance': dist,
                    'distance_norm': dist_norm,
                    'contains': contain,
                    'depth_close': depth_close,
                    'semantic_prior': sem,
                    'size_ratio': size_ratio,
                    'pair_score': score,
                }))
    candidates.sort(key=lambda x: x[2]['pair_score'], reverse=True)
    return candidates[:max_pairs]
