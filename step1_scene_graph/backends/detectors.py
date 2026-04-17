from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForCausalLM,
)
from step1_scene_graph.schemas import ObjectNode
from step1_scene_graph.utils.image_utils import load_image_bgr, load_image_rgb


@dataclass
class DetectionConfig:
    backend: str
    device: str = "cpu"
    min_confidence: float = 0.1
    max_objects: int = 20
    owlvit_labels: Optional[List[str]] = None
    gdino_text_prompt: Optional[str] = None
    florence2_task_prompt: str = "<OD>"
    annotation_path: Optional[str] = None


class BaseDetector:
    def detect(self, image_path: str) -> List[ObjectNode]:
        raise NotImplementedError


class AnnotationDetector(BaseDetector):
    def __init__(self, annotation_path: str):
        import json
        self.data = json.load(open(annotation_path, 'r', encoding='utf-8'))

    def detect(self, image_path: str) -> List[ObjectNode]:
        nodes = []
        for i, item in enumerate(self.data.get('objects', [])):
            x1, y1, x2, y2 = item['bbox']
            w, h = x2 - x1, y2 - y1
            nodes.append(ObjectNode(
                id=i, label=item['label'], bbox=[x1, y1, x2, y2], confidence=float(item.get('confidence', 1.0)),
                center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0], width=w, height=h, area=w * h, backend='annotation'
            ))
        return nodes


class ContourDetector(BaseDetector):
    def __init__(self, min_confidence: float = 0.1, max_objects: int = 20):
        self.min_confidence = min_confidence
        self.max_objects = max_objects

    def detect(self, image_path: str) -> List[ObjectNode]:
        image = load_image_bgr(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nodes = []
        idx = 0
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 2500:
                continue
            nodes.append(ObjectNode(
                id=idx, label='object', bbox=[x, y, x + w, y + h], confidence=0.5,
                center=[x + w / 2.0, y + h / 2.0], width=w, height=h, area=area, backend='contour'
            ))
            idx += 1
            if idx >= self.max_objects:
                break
        return nodes


class OwlViTDetector(BaseDetector):
    def __init__(self, labels: List[str], device: str = 'cpu', min_confidence: float = 0.1, max_objects: int = 20):
        self.labels = labels
        self.min_confidence = min_confidence
        self.max_objects = max_objects
        device_id = 0 if device.startswith('cuda') else -1
        self.pipe = pipeline('zero-shot-object-detection', model='google/owlvit-base-patch32', device=device_id)

    def detect(self, image_path: str) -> List[ObjectNode]:
        raw = self.pipe(image_path, candidate_labels=self.labels)
        nodes = []
        idx = 0
        for r in sorted(raw, key=lambda x: x['score'], reverse=True):
            if r['score'] < self.min_confidence:
                continue
            box = r['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            w, h = x2 - x1, y2 - y1
            nodes.append(ObjectNode(
                id=idx, label=r['label'], bbox=[x1, y1, x2, y2], confidence=float(r['score']),
                center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0], width=w, height=h, area=w * h, backend='owlvit'
            ))
            idx += 1
            if idx >= self.max_objects:
                break
        return nodes


class DetrDetector(BaseDetector):
    def __init__(self, device: str = 'cpu', min_confidence: float = 0.1, max_objects: int = 20):
        self.min_confidence = min_confidence
        self.max_objects = max_objects
        device_id = 0 if device.startswith('cuda') else -1
        self.pipe = pipeline('object-detection', model='facebook/detr-resnet-50', device=device_id)

    def detect(self, image_path: str) -> List[ObjectNode]:
        raw = self.pipe(image_path)
        nodes = []
        idx = 0
        for r in sorted(raw, key=lambda x: x['score'], reverse=True):
            if r['score'] < self.min_confidence:
                continue
            box = r['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            w, h = x2 - x1, y2 - y1
            nodes.append(ObjectNode(
                id=idx, label=r['label'], bbox=[x1, y1, x2, y2], confidence=float(r['score']),
                center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0], width=w, height=h, area=w * h, backend='detr'
            ))
            idx += 1
            if idx >= self.max_objects:
                break
        return nodes


class GroundingDinoDetector(BaseDetector):
    def __init__(self, device: str = 'cpu', text_prompt: str = 'chair . table . sofa .'):
        self.processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
        self.device = device
        if device.startswith('cuda'):
            self.model = self.model.to(device)
        self.text_prompt = text_prompt

    def detect(self, image_path: str) -> List[ObjectNode]:
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors='pt')
        if self.device.startswith('cuda'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]],
        )[0]
        nodes = []
        for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            nodes.append(ObjectNode(
                id=i, label=str(label), bbox=[x1, y1, x2, y2], confidence=float(score),
                center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0], width=w, height=h, area=w * h, backend='groundingdino'
            ))
        return nodes


class Florence2Detector(BaseDetector):
    def __init__(self, device: str = 'cpu', task_prompt: str = '<OD>'):
        model_id = 'microsoft/Florence-2-base'
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.device = device
        if device.startswith('cuda'):
            self.model = self.model.to(device)
        self.task_prompt = task_prompt

    def detect(self, image_path: str) -> List[ObjectNode]:
        image = Image.open(image_path).convert('RGB')
        prompt = self.task_prompt
        inputs = self.processor(text=prompt, images=image, return_tensors='pt')
        if self.device.startswith('cuda'):
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        od = parsed.get(prompt, {})
        bboxes = od.get('bboxes', [])
        labels = od.get('labels', [])
        nodes = []
        for i, (box, label) in enumerate(zip(bboxes, labels)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            nodes.append(ObjectNode(
                id=i, label=str(label), bbox=[x1, y1, x2, y2], confidence=0.9,
                center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0], width=w, height=h, area=w * h, backend='florence2'
            ))
        return nodes


def get_detector(cfg: DetectionConfig) -> BaseDetector:
    if cfg.backend == 'annotation':
        if not cfg.annotation_path:
            raise ValueError('annotation backend requires --annotation_path')
        return AnnotationDetector(cfg.annotation_path)
    if cfg.backend == 'contour':
        return ContourDetector(cfg.min_confidence, cfg.max_objects)
    if cfg.backend == 'owlvit':
        labels = cfg.owlvit_labels or ['chair', 'table', 'sofa', 'plant', 'window']
        return OwlViTDetector(labels=labels, device=cfg.device, min_confidence=cfg.min_confidence, max_objects=cfg.max_objects)
    if cfg.backend == 'detr':
        return DetrDetector(device=cfg.device, min_confidence=cfg.min_confidence, max_objects=cfg.max_objects)
    if cfg.backend == 'groundingdino':
        return GroundingDinoDetector(device=cfg.device, text_prompt=cfg.gdino_text_prompt or 'chair . table . sofa . plant . window .')
    if cfg.backend == 'florence2':
        return Florence2Detector(device=cfg.device, task_prompt=cfg.florence2_task_prompt)
    raise ValueError(f'Unknown detector backend: {cfg.backend}')
