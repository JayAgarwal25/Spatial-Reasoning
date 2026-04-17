"""
baseline.py
-----------
Zero-shot VLM baseline evaluation for quantitative spatial reasoning.

Supports:
  - GPT-4o  (OpenAI)
  - Gemini 1.5 Pro  (Google)
  - InternVL2-8B  (open-weights, local)
  - LLaVA-NeXT  (open-weights, local)

Each baseline follows the same interface:
    predict_distances(image_path, object_pairs) -> List[float]
    predict_relation(image_path, qa_pair) -> str

All baseline classes inherit from VLMBaseline and are registered in
BASELINE_REGISTRY so the eval scripts can instantiate them by name.

Usage (from eval scripts):
    from step4_evaluation.baseline import build_baseline
    model = build_baseline("gpt4o", api_key=os.environ["OPENAI_API_KEY"])
    pred_dist = model.predict_distances(img_path, [("chair", "table")])
"""

import os
import re
import json
import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMBaseline(ABC):
    """
    Abstract baseline model.  All subclasses must implement:
        predict_distances   — metric distance estimation
        predict_relation    — categorical spatial QA (for 3DSRBench)
        tag                 — short identifier string
    """

    @property
    @abstractmethod
    def tag(self) -> str:
        ...

    @abstractmethod
    def predict_distances(
        self,
        image_path: str,
        object_pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """
        For each (object_A, object_B) pair in `object_pairs`, return a
        predicted metric distance in the same unit as the dataset GT.

        Returns list of floats, same length as object_pairs.
        nan signals refusal / parse failure.
        """
        ...

    @abstractmethod
    def predict_relation(
        self,
        image_path: str,
        question: str,
        choices: List[str],
    ) -> str:
        """
        Multiple-choice spatial QA.  Returns the chosen option string.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _parse_distance(text: str) -> float:
        """
        Extract the first numeric token from a model response.
        Returns nan on failure.
        """
        matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", "."))
        if matches:
            return float(matches[0])
        return float("nan")

    @staticmethod
    def _distance_prompt(obj_a: str, obj_b: str) -> str:
        return (
            f"Look at the image carefully. Estimate the real-world metric "
            f"distance in centimetres between the '{obj_a}' and the '{obj_b}'. "
            f"Reply with a single number only (no units, no explanation)."
        )

    @staticmethod
    def _relation_prompt(question: str, choices: List[str]) -> str:
        choices_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
        return (
            f"{question}\n\nChoose exactly one:\n{choices_str}\n\n"
            f"Reply with the letter only (A, B, C, …)."
        )


# ---------------------------------------------------------------------------
# GPT-4o baseline
# ---------------------------------------------------------------------------

class GPT4oBaseline(VLMBaseline):
    """
    OpenAI GPT-4o vision baseline.
    Requires:  pip install openai
    """

    tag = "gpt4o"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self._client = OpenAI(api_key=api_key)
        self._model  = model

    def _call(self, image_path: str, prompt: str) -> str:
        b64 = self._encode_image_b64(image_path)
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png"}.get(ext, "image/jpeg")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def predict_distances(self, image_path, object_pairs):
        results = []
        for (a, b) in object_pairs:
            try:
                raw = self._call(image_path, self._distance_prompt(a, b))
                results.append(self._parse_distance(raw))
            except Exception as e:
                logger.warning(f"GPT-4o distance call failed: {e}")
                results.append(float("nan"))
        return results

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call(image_path, self._relation_prompt(question, choices))
            letter = raw.strip()[0].upper() if raw.strip() else "A"
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception as e:
            logger.warning(f"GPT-4o relation call failed: {e}")
        return choices[0]


# ---------------------------------------------------------------------------
# Gemini 1.5 Pro baseline
# ---------------------------------------------------------------------------

class GeminiBaseline(VLMBaseline):
    """
    Google Gemini 1.5 Pro vision baseline.
    Requires:  pip install google-generativeai
    """

    tag = "gemini15pro"

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-1.5-pro-002"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def _call(self, image_path: str, prompt: str) -> str:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pip install Pillow")

        img = Image.open(image_path)
        response = self._model.generate_content(
            [prompt, img],
            generation_config={"temperature": 0.0, "max_output_tokens": 64},
        )
        return response.text.strip()

    def predict_distances(self, image_path, object_pairs):
        results = []
        for (a, b) in object_pairs:
            try:
                raw = self._call(image_path, self._distance_prompt(a, b))
                results.append(self._parse_distance(raw))
            except Exception as e:
                logger.warning(f"Gemini distance call failed: {e}")
                results.append(float("nan"))
        return results

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call(image_path, self._relation_prompt(question, choices))
            letter = raw.strip()[0].upper() if raw.strip() else "A"
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception as e:
            logger.warning(f"Gemini relation call failed: {e}")
        return choices[0]


# ---------------------------------------------------------------------------
# InternVL2-8B baseline (local, open-weights)
# ---------------------------------------------------------------------------

class InternVL2Baseline(VLMBaseline):
    """
    InternVL2-8B local inference baseline.
    Requires:  pip install transformers torch torchvision Pillow
    Model:     OpenGVLab/InternVL2-8B  (auto-downloaded from HuggingFace)
    """

    tag = "internvl2-8b"

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-8B",
                 device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)
        self._device = device

    def _call(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")
        # InternVL2 pixel_values preprocessing
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(img).unsqueeze(0).to(
            torch.bfloat16).to(self._device)

        generation_config = {"max_new_tokens": 64, "do_sample": False}
        response = self._model.chat(
            self._tokenizer, pixel_values,
            question=prompt,
            generation_config=generation_config,
        )
        return response.strip()

    def predict_distances(self, image_path, object_pairs):
        results = []
        for (a, b) in object_pairs:
            try:
                raw = self._call(image_path, self._distance_prompt(a, b))
                results.append(self._parse_distance(raw))
            except Exception as e:
                logger.warning(f"InternVL2 distance call failed: {e}")
                results.append(float("nan"))
        return results

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call(image_path, self._relation_prompt(question, choices))
            letter = raw.strip()[0].upper() if raw.strip() else "A"
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception as e:
            logger.warning(f"InternVL2 relation call failed: {e}")
        return choices[0]


# ---------------------------------------------------------------------------
# LLaVA-NeXT baseline (local, open-weights)
# ---------------------------------------------------------------------------

class LLaVANextBaseline(VLMBaseline):
    """
    LLaVA-NeXT (LLaVA-1.6) local inference baseline.
    Requires:  pip install transformers torch Pillow
    Model:     llava-hf/llava-v1.6-mistral-7b-hf
    """

    tag = "llava-next-7b"

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 device: str = "cuda"):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch

        self._processor = LlavaNextProcessor.from_pretrained(model_name)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self._device = device

    def _call(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")
        # LLaVA-NeXT expects prompt wrapped in conversation template
        conversation = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = self._processor(formatted, img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = self._processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1]
        return decoded.strip()

    def predict_distances(self, image_path, object_pairs):
        results = []
        for (a, b) in object_pairs:
            try:
                raw = self._call(image_path, self._distance_prompt(a, b))
                results.append(self._parse_distance(raw))
            except Exception as e:
                logger.warning(f"LLaVA-NeXT distance call failed: {e}")
                results.append(float("nan"))
        return results

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call(image_path, self._relation_prompt(question, choices))
            letter = raw.strip()[0].upper() if raw.strip() else "A"
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception as e:
            logger.warning(f"LLaVA-NeXT relation call failed: {e}")
        return choices[0]


# ---------------------------------------------------------------------------
# Mock baseline (for unit-testing without API keys / GPUs)
# ---------------------------------------------------------------------------

class MockBaseline(VLMBaseline):
    """
    Deterministic mock that returns random-but-reproducible predictions.
    Use for integration testing the eval pipeline without API access.
    """

    tag = "mock"

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def predict_distances(self, image_path, object_pairs):
        # Simulate plausible but mildly noisy predictions (0–300 cm)
        return [float(self._rng.uniform(10, 300)) for _ in object_pairs]

    def predict_relation(self, image_path, question, choices):
        return self._rng.choice(choices)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: Dict[str, type] = {
    "gpt4o":          GPT4oBaseline,
    "gemini15pro":    GeminiBaseline,
    "internvl2-8b":   InternVL2Baseline,
    "llava-next-7b":  LLaVANextBaseline,
    "mock":           MockBaseline,
}


def build_baseline(name: str, **kwargs) -> VLMBaseline:
    """
    Instantiate a baseline by its registry name.

    Parameters
    ----------
    name   : one of "gpt4o", "gemini15pro", "internvl2-8b",
                    "llava-next-7b", "mock"
    kwargs : forwarded to the class constructor (e.g. api_key, device)
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name](**kwargs)
