from __future__ import annotations
import argparse
import os
import cv2
from step1_scene_graph.backends.detectors import DetectionConfig, get_detector
from step1_scene_graph.backends.depth import get_depth_backend
from step1_scene_graph.relation.pair_pruning import prune_pairs
from step1_scene_graph.relation.backends import get_relation_model
from step1_scene_graph.relation.edge_builder import build_sparse_edges
from step1_scene_graph.graph_builder import build_scene_graph
from step1_scene_graph.visualize import draw_scene_graph
from step1_scene_graph.utils.io import save_json
from step1_scene_graph.utils.image_utils import load_image_rgb
from step1_scene_graph.utils.logging_utils import log_info


def _split_csv(text: str | None):
    if not text:
        return None
    return [x.strip() for x in text.split(',') if x.strip()]


def run_pipeline(
    image_path: str,
    json_output_path: str,
    viz_output_path: str,
    detector_backend: str,
    depth_backend: str,
    relation_backend: str,
    use_vlm_relations: bool,
    device: str,
    min_confidence: float,
    edge_threshold: float,
    max_pairs: int,
    max_objects: int,
    owlvit_labels=None,
    gdino_text_prompt=None,
    florence2_task_prompt='<OD>',
    annotation_path=None,
):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f'Could not read image: {image_path}')
    image_h, image_w = image_bgr.shape[:2]
    image_rgb = load_image_rgb(image_path)

    dcfg = DetectionConfig(
        backend=detector_backend,
        device=device,
        min_confidence=min_confidence,
        max_objects=max_objects,
        owlvit_labels=owlvit_labels,
        gdino_text_prompt=gdino_text_prompt,
        florence2_task_prompt=florence2_task_prompt,
        annotation_path=annotation_path,
    )
    detector = get_detector(dcfg)
    nodes = detector.detect(image_path)
    log_info(f'Detected {len(nodes)} nodes via {detector_backend}')

    depth_model = get_depth_backend(depth_backend, device=device)
    nodes = depth_model.attach_depth(image_path, nodes)
    log_info(f'Attached depth via {depth_backend}')

    pruned_pairs = prune_pairs(nodes, image_w=image_w, image_h=image_h, max_pairs=max_pairs)
    log_info(f'Pruned to {len(pruned_pairs)} candidate pairs')

    relation_model = get_relation_model(relation_backend, device=device)
    edges = build_sparse_edges(
        image_rgb=image_rgb,
        pruned_pairs=pruned_pairs,
        relation_model=relation_model,
        edge_threshold=edge_threshold,
        use_vlm_relations=use_vlm_relations,
    )
    log_info(f'Generated {len(edges)} sparse edges via {relation_backend}')

    stats = {
        'num_nodes': len(nodes),
        'num_pruned_pairs': len(pruned_pairs),
        'num_edges': len(edges),
        'detector_backend': detector_backend,
        'depth_backend': depth_backend,
        'relation_backend': relation_backend,
        'use_vlm_relations': use_vlm_relations,
        'edge_threshold': edge_threshold,
        'max_pairs': max_pairs,
    }

    graph = build_scene_graph(image_path, image_w, image_h, nodes, edges, stats=stats)
    save_json(graph.to_dict(), json_output_path)
    draw_scene_graph(graph, viz_output_path)
    log_info(f'Saved graph JSON to {json_output_path}')
    log_info(f'Saved visualization to {viz_output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image_path', required=True)
    p.add_argument('--json_output_path', required=True)
    p.add_argument('--viz_output_path', required=True)
    p.add_argument('--detector_backend', default='owlvit', choices=['annotation', 'contour', 'owlvit', 'detr', 'groundingdino', 'florence2'])
    p.add_argument('--depth_backend', default='dpt', choices=['none', 'pseudo', 'dpt'])
    p.add_argument('--relation_backend', default='blip2', choices=['heuristic', 'blip2', 'llava'])
    p.add_argument('--use_vlm_relations', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--min_confidence', type=float, default=0.12)
    p.add_argument('--edge_threshold', type=float, default=0.48)
    p.add_argument('--max_pairs', type=int, default=24)
    p.add_argument('--max_objects', type=int, default=20)
    p.add_argument('--owlvit_labels', type=str, default=None)
    p.add_argument('--gdino_text_prompt', type=str, default=None)
    p.add_argument('--florence2_task_prompt', type=str, default='<OD>')
    p.add_argument('--annotation_path', type=str, default=None)
    args = p.parse_args()

    run_pipeline(
        image_path=args.image_path,
        json_output_path=args.json_output_path,
        viz_output_path=args.viz_output_path,
        detector_backend=args.detector_backend,
        depth_backend=args.depth_backend,
        relation_backend=args.relation_backend,
        use_vlm_relations=args.use_vlm_relations,
        device=args.device,
        min_confidence=args.min_confidence,
        edge_threshold=args.edge_threshold,
        max_pairs=args.max_pairs,
        max_objects=args.max_objects,
        owlvit_labels=_split_csv(args.owlvit_labels),
        gdino_text_prompt=args.gdino_text_prompt,
        florence2_task_prompt=args.florence2_task_prompt,
        annotation_path=args.annotation_path,
    )
