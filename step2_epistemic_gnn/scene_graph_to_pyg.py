"""
Adapter: Step 1 JSON → Step 2 PyG Data

Reads the JSON produced by step1_scene_graph.run_pipeline and converts it
into the named-field PyG Data object that QuantEpiGNN expects.

Node semantic embeddings are produced by encoding node labels with a frozen
sentence-transformer (all-MiniLM-L6-v2, 384-dim).  Pass embed_model=None to
use zero vectors instead (useful for unit tests without GPU/model weights).

Edge metric distance is approximated as center_distance / image_diagonal,
giving a dimensionless [0, 1] ratio.  Swap in real metric distances via the
annotation pipeline once GT data is available.

Predicate vocabulary (14 classes) mirrors Step 1's VALID_RELATIONS exactly.
Set num_pred_classes=14 in QuantEpiGNN and train.py to match.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Predicate vocabulary — must stay in sync with step1_scene_graph/relation/parser.py
# ---------------------------------------------------------------------------

PREDICATE_VOCAB: List[str] = sorted([
    'above', 'behind', 'below', 'contains', 'in_front_of',
    'inside', 'left_of', 'near', 'none', 'on',
    'overlapping', 'right_of', 'surrounding', 'under',
])
NUM_PRED_CLASSES = len(PREDICATE_VOCAB)          # 14
PRED_TO_IDX = {p: i for i, p in enumerate(PREDICATE_VOCAB)}

# Embedding dimension produced by all-MiniLM-L6-v2
SEM_DIM = 384


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

def load_embed_model(model_name: str = "all-MiniLM-L6-v2"):
    """Returns a SentenceTransformer for encoding node label strings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for label embedding. "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def scene_graph_json_to_pyg(
    json_path: str,
    embed_model=None,
    add_labels: bool = True,
) -> Data:
    """
    Convert a Step 1 JSON scene graph file to a PyG Data object for Step 2.

    Args:
        json_path   : path to JSON produced by step1_scene_graph/run_pipeline.py
        embed_model : SentenceTransformer instance; None → zero node_sem vectors
        add_labels  : attach data.target_classes (predicate indices) and
                      data.target_dist (normalized distance proxy) for training

    Returns:
        PyG Data satisfying the Step 1 → Step 2 interface contract.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        sg = json.load(f)

    nodes = sg["nodes"]
    edges = sg["edges"]
    image_diag = math.sqrt(sg["image_width"] ** 2 + sg["image_height"] ** 2)

    N = len(nodes)

    # ---- node_sem: encode label strings → (N, SEM_DIM) ----
    if embed_model is not None:
        labels = [n["label"] for n in nodes]
        node_sem = torch.tensor(
            embed_model.encode(labels, show_progress_bar=False),
            dtype=torch.float32,
        )
    else:
        node_sem = torch.zeros(N, SEM_DIM)

    # ---- node_bbox: [x1,y1,x2,y2] → [x,y,w,h] ----
    bboxes = []
    for n in nodes:
        x1, y1, x2, y2 = n["bbox"]
        bboxes.append([x1, y1, x2 - x1, y2 - y1])
    node_bbox = torch.tensor(bboxes, dtype=torch.float32)   # (N, 4)

    # ---- node_depth: scalar → (N, 1), default 0.0 if missing ----
    node_depth = torch.tensor(
        [[n["depth"] if n["depth"] is not None else 0.0] for n in nodes],
        dtype=torch.float32,
    )   # (N, 1)

    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    # ---- Deduplicate directed edges (keep first occurrence per pair) ----
    seen: set = set()
    deduped = []
    for e in edges:
        key = (e["subject_id"], e["object_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    E = len(deduped)

    if E == 0:
        data = Data(
            node_sem=node_sem,
            node_bbox=node_bbox,
            node_depth=node_depth,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_dist=torch.zeros(0, 1),
            edge_conf=torch.zeros(0, 1),
            edge_angle=torch.zeros(0, 1),
            edge_depth_diff=torch.zeros(0, 1),
            num_nodes=N,
        )
        if add_labels:
            data.target_classes = torch.zeros(0, dtype=torch.long)
            data.target_dist = torch.zeros(0, 1)
        return data

    src_list, dst_list = [], []
    dist_list, conf_list, angle_list, ddiff_list = [], [], [], []
    cls_list = []

    for e in deduped:
        feats = e.get("features", {})

        if e["subject_id"] not in id_to_idx or e["object_id"] not in id_to_idx:
            print(f"Warning: skipping edge ({e['subject_id']}→{e['object_id']}) — node ID not in graph")
            continue

        src_list.append(id_to_idx[e["subject_id"]])
        dst_list.append(id_to_idx[e["object_id"]])

        # Metric distance proxy: normalize pixel center_distance by image diagonal
        center_dist = feats.get("center_distance", 0.0)
        dist_list.append(center_dist / image_diag if image_diag > 0 else 0.0)

        conf_list.append(float(e["confidence"]))

        # angle_rad and depth_diff come from geometry_relation() stored in features
        angle_list.append(float(feats.get("angle_rad", 0.0)))
        depth_diff = feats.get("depth_diff", None)
        ddiff_list.append(float(depth_diff) if depth_diff is not None else 0.0)

        if add_labels:
            cls_list.append(PRED_TO_IDX.get(e["predicate"], PRED_TO_IDX["none"]))

    edge_index     = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_dist      = torch.tensor(dist_list,  dtype=torch.float32).unsqueeze(1)
    edge_conf      = torch.tensor(conf_list,  dtype=torch.float32).unsqueeze(1)
    edge_angle     = torch.tensor(angle_list, dtype=torch.float32).unsqueeze(1)
    edge_depth_diff = torch.tensor(ddiff_list, dtype=torch.float32).unsqueeze(1)

    data = Data(
        node_sem=node_sem,
        node_bbox=node_bbox,
        node_depth=node_depth,
        edge_index=edge_index,
        edge_dist=edge_dist,
        edge_conf=edge_conf,
        edge_angle=edge_angle,
        edge_depth_diff=edge_depth_diff,
        num_nodes=N,
    )

    if add_labels:
        data.target_classes = torch.tensor(cls_list, dtype=torch.long)
        # target_dist placeholder: use normalized center_distance until GT is available
        data.target_dist = edge_dist.clone()

    return data


# ---------------------------------------------------------------------------
# Directory-level loader
# ---------------------------------------------------------------------------

def load_scene_graph_dataset(json_dir: str, embed_model=None) -> List[Data]:
    """
    Load all Step 1 JSON files in a directory as a list of PyG Data objects.
    Files that fail to parse are skipped with a warning.
    """
    graphs = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        try:
            graphs.append(scene_graph_json_to_pyg(path, embed_model=embed_model))
        except Exception as exc:
            print(f"Warning: skipping {fname}: {exc}")
    return graphs
