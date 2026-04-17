from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration
from step1_scene_graph.relation.parser import parse_relation_text


PROMPT_TEMPLATE = """You are given an image crop containing two referenced objects.\nObject A: {label_a}\nObject B: {label_b}\nReturn ONLY one relation from this set:\n[left_of, right_of, above, below, on, under, inside, surrounding, overlapping, in_front_of, behind, near, contains, none]\nOne token or one phrase only."""


@dataclass
class RelationResult:
    relation: str
    raw_text: str
    model_confidence: float
    backend: str


class BaseRelationModel:
    def infer(self, crop_rgb, label_a: str, label_b: str) -> RelationResult:
        raise NotImplementedError


class HeuristicRelationModel(BaseRelationModel):
    def infer(self, crop_rgb, label_a: str, label_b: str) -> RelationResult:
        return RelationResult(relation='none', raw_text='none', model_confidence=0.0, backend='heuristic')


class Blip2RelationModel(BaseRelationModel):
    def __init__(self, device: str = 'cpu', model_id: str = 'Salesforce/blip2-opt-2.7b'):
        self.processor = Blip2Processor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        self.device = device
        self.model = self.model.to(device)

    def infer(self, crop_rgb, label_a: str, label_b: str) -> RelationResult:
        image = Image.fromarray(crop_rgb)
        prompt = PROMPT_TEMPLATE.format(label_a=label_a, label_b=label_b)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=8)
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        rel = parse_relation_text(raw)
        conf = 0.9 if rel != 'none' else 0.35
        return RelationResult(relation=rel, raw_text=raw, model_confidence=conf, backend='blip2')


class LlavaRelationModel(BaseRelationModel):
    def __init__(self, device: str = 'cpu', model_id: str = 'llava-hf/llava-1.5-7b-hf'):
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.device = device
        self.model = self.model.to(device)

    def infer(self, crop_rgb, label_a: str, label_b: str) -> RelationResult:
        image = Image.fromarray(crop_rgb)
        prompt = 'USER: <image>\n' + PROMPT_TEMPLATE.format(label_a=label_a, label_b=label_b) + '\nASSISTANT:'
        inputs = self.processor(images=image, text=prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=12)
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if 'ASSISTANT:' in raw:
            raw = raw.split('ASSISTANT:')[-1].strip()
        rel = parse_relation_text(raw)
        conf = 0.92 if rel != 'none' else 0.3
        return RelationResult(relation=rel, raw_text=raw, model_confidence=conf, backend='llava')


def get_relation_model(name: str, device: str = 'cpu') -> BaseRelationModel:
    if name == 'heuristic':
        return HeuristicRelationModel()
    if name == 'blip2':
        return Blip2RelationModel(device=device)
    if name == 'llava':
        return LlavaRelationModel(device=device)
    raise ValueError(f'Unknown relation backend: {name}')
