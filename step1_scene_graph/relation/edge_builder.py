from __future__ import annotations
from typing import List
from step1_scene_graph.schemas import RelationEdge
from step1_scene_graph.relation.geometry import geometry_relation
from step1_scene_graph.utils.image_utils import crop_union_region


def build_sparse_edges(image_rgb, pruned_pairs, relation_model, edge_threshold: float = 0.48, use_vlm_relations: bool = True, max_outgoing_per_node: int = 4) -> List[RelationEdge]:
    edges = []
    outgoing = {}
    for a, b, meta in pruned_pairs:
        if outgoing.get(a.id, 0) >= max_outgoing_per_node:
            continue
        geom = geometry_relation(a, b)
        final_pred = geom['predicate']
        raw_text = None
        rel_backend = 'geometry'
        model_conf = 0.0
        if use_vlm_relations:
            crop = crop_union_region(image_rgb, a.bbox, b.bbox)
            rr = relation_model.infer(crop, a.label, b.label)
            raw_text = rr.raw_text
            rel_backend = rr.backend
            model_conf = rr.model_confidence
            if rr.relation != 'none':
                final_pred = rr.relation
            elif geom['geometry_confidence'] < 0.6:
                final_pred = 'none'
        sem = meta['semantic_prior']
        if use_vlm_relations:
            conf = 0.35 * geom['geometry_confidence'] + 0.45 * model_conf + 0.20 * sem
        else:
            conf = 0.65 * geom['geometry_confidence'] + 0.35 * sem
        if final_pred != 'none' and conf >= edge_threshold:
            edges.append(RelationEdge(
                subject_id=a.id,
                object_id=b.id,
                predicate=final_pred,
                confidence=float(conf),
                backend=rel_backend,
                raw_relation_text=raw_text,
                features={**meta, **geom, 'model_confidence': model_conf}
            ))
            outgoing[a.id] = outgoing.get(a.id, 0) + 1
    return edges
