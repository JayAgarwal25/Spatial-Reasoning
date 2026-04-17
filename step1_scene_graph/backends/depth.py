from __future__ import annotations
from typing import List
import numpy as np
import torch
from transformers import pipeline
from step1_scene_graph.schemas import ObjectNode


class DepthBackend:
    def attach_depth(self, image_path: str, nodes: List[ObjectNode]) -> List[ObjectNode]:
        raise NotImplementedError


class NoDepthBackend(DepthBackend):
    def attach_depth(self, image_path: str, nodes: List[ObjectNode]) -> List[ObjectNode]:
        return nodes


class PseudoDepthBackend(DepthBackend):
    def attach_depth(self, image_path: str, nodes: List[ObjectNode]) -> List[ObjectNode]:
        for n in nodes:
            n.depth = float((n.bbox[1] + n.bbox[3]) / 2.0)
        return nodes


class DptDepthBackend(DepthBackend):
    def __init__(self, device: str = 'cpu'):
        device_id = 0 if device.startswith('cuda') else -1
        self.pipe = pipeline('depth-estimation', model='Intel/dpt-large', device=device_id)

    def attach_depth(self, image_path: str, nodes: List[ObjectNode]) -> List[ObjectNode]:
        out = self.pipe(image_path)
        pred = out['predicted_depth']
        if torch.is_tensor(pred):
            depth_map = pred.squeeze().detach().cpu().numpy()
        else:
            depth_map = np.asarray(pred)
        h, w = depth_map.shape[:2]
        for n in nodes:
            x1, y1, x2, y2 = [int(max(0, v)) for v in n.bbox]
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            patch = depth_map[y1:y2, x1:x2]
            n.depth = float(np.median(patch)) if patch.size else None
        return nodes


def get_depth_backend(name: str, device: str = 'cpu') -> DepthBackend:
    if name == 'none':
        return NoDepthBackend()
    if name == 'pseudo':
        return PseudoDepthBackend()
    if name == 'dpt':
        return DptDepthBackend(device=device)
    raise ValueError(f'Unknown depth backend: {name}')
