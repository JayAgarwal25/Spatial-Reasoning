"""
prepare_spatial457.py
---------------------
Converts the RyanWW/Spatial457 (superCLEVR) bulk JSON into individual
Step-1-compatible scene graph JSONs with ground-truth labels.

What this produces
------------------
One JSON per scene in data/scene_graphs/, named after the image stem.
Each JSON is fully compatible with scene_graph_to_pyg.py and the
existing Step-1 output schema.  Three GT fields are attached:

  node.depth              — camera-space Z from pixel_coords (true metric depth)
  edge.predicate          — spatial predicate computed via geometry_relation()
                            with GT depth (accurate in_front_of / behind labels)
  edge.features['gt_metric_dist_m']  — Euclidean distance in 3-D world coords (m)

The target_dist used during GNN training will be pulled from
gt_metric_dist_m (see scene_graph_to_pyg.py, GT_DIST_KEY).

Usage
-----
    python prepare_spatial457.py \
        --scenes_json data/spatial457/spatial457_scenes_21k.json \
        --images_dir  data/spatial457/images \
        --out_dir     data/scene_graphs \
        --workers     8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Predicate vocabulary & geometry helpers (mirrors scene_graph_to_pyg.py)
# ---------------------------------------------------------------------------

NEAR_THRESH_M = 2.0   # 3D world-space distance (m) to label an edge "near"


def _bbox_iou(ba: List[float], bb: List[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(ba[0], bb[0]); iy1 = max(ba[1], bb[1])
    ix2 = min(ba[2], bb[2]); iy2 = min(ba[3], bb[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ba[2]-ba[0]) * max(0.0, ba[3]-ba[1])
    area_b = max(0.0, bb[2]-bb[0]) * max(0.0, bb[3]-bb[1])
    denom  = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _contains(outer: List[float], inner: List[float], tol: float = 4) -> bool:
    return (outer[0] - tol <= inner[0] and inner[2] <= outer[2] + tol and
            outer[1] - tol <= inner[1] and inner[3] <= outer[3] + tol)


def compute_gt_predicate(
    center_a: Tuple[float, float],
    bbox_a:   List[float],          # [x1,y1,x2,y2]
    depth_a:  Optional[float],      # camera-space Z (metres)
    center_b: Tuple[float, float],
    bbox_b:   List[float],
    depth_b:  Optional[float],
    dist_3d_m: float,               # Euclidean 3-D world distance (m)
) -> Tuple[str, float]:
    """
    Compute GT predicate for directed edge A→B.

    Priority (mirrors geometry_relation in relation/geometry.py but uses
    true GT depth for in_front_of / behind instead of pseudo-depth):
        1. bbox containment  → contains / inside
        2. high IoU          → overlapping
        3. near (3D)         → near
        4. depth dominates   → in_front_of / behind
        5. horizontal pixel  → left_of / right_of
        6. vertical pixel    → above / below
    """
    iou = _bbox_iou(bbox_a, bbox_b)

    if _contains(bbox_a, bbox_b, tol=4):
        return 'contains', 0.95
    if _contains(bbox_b, bbox_a, tol=4):
        return 'inside', 0.95
    if iou > 0.25:
        return 'overlapping', 0.85
    if dist_3d_m < NEAR_THRESH_M:
        return 'near', 0.80

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    if depth_a is not None and depth_b is not None:
        ref_depth = max(abs(depth_a), abs(depth_b), 1e-6)
        if abs(depth_b - depth_a) > 0.12 * ref_depth:
            pred = 'in_front_of' if depth_a < depth_b else 'behind'
            return pred, 0.75

    if abs(dx) > abs(dy):
        pred = 'left_of' if center_a[0] < center_b[0] else 'right_of'
    else:
        pred = 'above' if center_a[1] < center_b[1] else 'below'
    return pred, 0.65


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def _process_scene(scene: Dict[str, Any], img_dir: str) -> Optional[Dict[str, Any]]:
    """
    Convert one superCLEVR scene dict into a Step-1-compatible scene graph dict.
    Returns None if the scene should be skipped.
    """
    img_fname = scene.get('image_filename', '')
    img_path  = os.path.join(img_dir, img_fname)
    if not os.path.exists(img_path):
        return None

    objects  = scene.get('objects', [])
    mask_box = scene.get('obj_mask_box', {})
    N = len(objects)
    if N < 2:
        return None

    W, H = 640, 480   # superCLEVR canonical resolution

    # ---- Build nodes -------------------------------------------------------
    nodes = []
    for i, obj in enumerate(objects):
        mb = mask_box.get(str(i), {}).get('obj', None)
        if mb is None or not mb:
            return None   # skip scenes with missing mask data

        x, y, w, h = mb[0]          # [x, y, w, h]  pixel coords
        # Guard against degenerate boxes
        if w <= 0 or h <= 0:
            return None

        x1, y1, x2, y2 = float(x), float(y), float(x+w), float(y+h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        pcoords = obj.get('pixel_coords', None)
        if pcoords is None:
            camera_z = float(cy)   # pseudo-depth fallback
        else:
            camera_z = float(pcoords[0][2])   # true camera-space Z

        label = f"{obj.get('size','?')}_{obj.get('color','?')}_{obj.get('shape','?')}"

        nodes.append({
            'id':         i,
            'label':      label,
            'bbox':       [x1, y1, x2, y2],
            'confidence': 1.0,
            'center':     [cx, cy],
            'width':      float(w),
            'height':     float(h),
            'area':       float(w * h),
            'depth':      camera_z,
            'backend':    'gt_annotation',
        })

    if len(nodes) < 2:
        return None

    # ---- Build edges (all ordered pairs i→j, i≠j) -------------------------
    image_diag = math.sqrt(W**2 + H**2)
    edges = []
    seen: set = set()

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i, j) in seen:
                continue
            seen.add((i, j))

            na = nodes[i]; nb = nodes[j]
            oa = objects[i]; ob = objects[j]

            # 3-D world-space distance (metres)
            c3a = np.array(oa.get('3d_coords', [0, 0, 0]), dtype=float)
            c3b = np.array(ob.get('3d_coords', [0, 0, 0]), dtype=float)
            dist_3d_m = float(np.linalg.norm(c3a - c3b))

            # Pixel-space geometry
            ca = tuple(na['center']); cb = tuple(nb['center'])
            dx = cb[0] - ca[0]; dy = cb[1] - ca[1]
            pixel_dist = math.sqrt(dx*dx + dy*dy)
            angle_rad  = math.atan2(dy, dx)
            depth_diff = nb['depth'] - na['depth']

            predicate, gconf = compute_gt_predicate(
                center_a  = ca,   bbox_a = na['bbox'],
                depth_a   = na['depth'],
                center_b  = cb,   bbox_b = nb['bbox'],
                depth_b   = nb['depth'],
                dist_3d_m = dist_3d_m,
            )

            edges.append({
                'subject_id':        i,
                'object_id':         j,
                'predicate':         predicate,
                'confidence':        gconf,
                'features': {
                    'center_distance':  pixel_dist,
                    'angle_rad':        angle_rad,
                    'depth_diff':       depth_diff,
                    'gt_metric_dist_m': dist_3d_m,   # true 3-D distance in metres
                    'iou':              _bbox_iou(na['bbox'], nb['bbox']),
                },
                'backend': 'gt_geometry',
            })

    return {
        'image_path':   img_path,
        'image_width':  W,
        'image_height': H,
        'nodes':        nodes,
        'edges':        edges,
        'stats': {
            'num_nodes':        N,
            'num_edges':        len(edges),
            'detector_backend': 'gt_annotation',
            'depth_backend':    'gt_pixel_coords',
            'relation_backend': 'gt_geometry',
        },
    }


def _worker(args):
    scene, img_dir, out_dir = args
    try:
        sg = _process_scene(scene, img_dir)
        if sg is None:
            return None, None
        stem = Path(scene['image_filename']).stem
        out_path = os.path.join(out_dir, f"{stem}.json")
        with open(out_path, 'w') as f:
            json.dump(sg, f)
        return stem, len(sg['edges'])
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate GT-labelled scene graph JSONs from Spatial457")
    ap.add_argument('--scenes_json', default='data/spatial457/spatial457_scenes_21k.json')
    ap.add_argument('--images_dir',  default='data/spatial457/images')
    ap.add_argument('--out_dir',     default='data/scene_graphs')
    ap.add_argument('--workers',     type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.scenes_json) as f:
        raw = json.load(f)

    avail = set(os.listdir(args.images_dir))
    scenes = [s for s in raw['scenes'] if s.get('image_filename', '') in avail]
    print(f"Found {len(scenes)} matching scenes.")

    tasks = [(s, args.images_dir, args.out_dir) for s in scenes]

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            stem, info = fut.result()
            if stem is None:
                if isinstance(info, str):
                    errors += 1
                    print(f"  ERROR: {info}")
                else:
                    skipped += 1
            else:
                done += 1

    print(f"\nDone: {done} scene graphs written to {args.out_dir}")
    print(f"Skipped: {skipped}  Errors: {errors}")

    # Quick sanity check on one output
    jsons = sorted(Path(args.out_dir).glob('*.json'))
    if jsons:
        with open(jsons[0]) as f:
            sg = json.load(f)
        print(f"\nSample ({jsons[0].name}):")
        print(f"  nodes={len(sg['nodes'])}  edges={len(sg['edges'])}")
        if sg['edges']:
            e0 = sg['edges'][0]
            print(f"  edge[0]: {e0['predicate']}  "
                  f"gt_dist={e0['features']['gt_metric_dist_m']:.3f}m")


if __name__ == '__main__':
    main()
