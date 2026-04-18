from __future__ import annotations
import math
from typing import Dict, Optional
from step1_scene_graph.schemas import ObjectNode
from step1_scene_graph.relation.pair_pruning import bbox_iou, contains


def geometry_relation(a: ObjectNode, b: ObjectNode) -> Dict:
    dx = b.center[0] - a.center[0]
    dy = b.center[1] - a.center[1]
    iou = bbox_iou(a.bbox, b.bbox)
    dist = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    depth_diff = None if a.depth is None or b.depth is None else b.depth - a.depth
    predicate = 'none'
    gconf = 0.2
    if contains(a.bbox, b.bbox, tol=4):
        predicate = 'contains'
        gconf = 0.95
    elif contains(b.bbox, a.bbox, tol=4):
        predicate = 'inside'
        gconf = 0.95
    elif iou > 0.25:
        predicate = 'overlapping'
        gconf = 0.85
    elif depth_diff is not None and abs(depth_diff) > 0.12 * max(abs(a.depth), abs(b.depth), 1.0):
        predicate = 'in_front_of' if a.depth < b.depth else 'behind'
        gconf = 0.65
    else:
        if abs(dx) > abs(dy):
            predicate = 'left_of' if a.center[0] < b.center[0] else 'right_of'
        else:
            predicate = 'above' if a.center[1] < b.center[1] else 'below'
        gconf = 0.55
    return {
        'predicate': predicate,
        'geometry_confidence': gconf,
        'dx': dx,
        'dy': dy,
        'center_distance': dist,
        'iou': iou,
        'depth_diff': depth_diff,
        'angle_rad': angle,
    }
