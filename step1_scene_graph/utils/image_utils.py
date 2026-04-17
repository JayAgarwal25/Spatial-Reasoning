from __future__ import annotations
import math
from typing import Tuple
import cv2
import numpy as np


def load_image_bgr(path: str):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def load_image_rgb(path: str):
    image = load_image_bgr(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_diagonal(width: int, height: int) -> float:
    return math.sqrt(width * width + height * height)


def clamp_box(box, w: int, h: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return [x1, y1, x2, y2]


def crop_union_region(image_rgb: np.ndarray, box_a, box_b, padding_ratio: float = 0.08) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    crop_box = clamp_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], w, h)
    x1, y1, x2, y2 = crop_box
    return image_rgb[y1:y2, x1:x2].copy()
