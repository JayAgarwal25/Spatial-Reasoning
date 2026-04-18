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
            f"distance in metres between the '{obj_a}' and the '{obj_b}'. "
            f"Reply with a single number only (no units, no explanation)."
        )

    @staticmethod
    def _batch_distance_prompt(pairs: List[Tuple[str, str]]) -> str:
        lines = "\n".join(
            f"{i+1}. '{a}' to '{b}'" for i, (a, b) in enumerate(pairs)
        )
        return (
            "Look at the image carefully. For each numbered pair of objects below, "
            "estimate the real-world 3D distance between them in metres.\n\n"
            f"{lines}\n\n"
            "Reply with ONLY a JSON array of numbers in the same order, e.g. "
            "[1.2, 3.4, 0.8]. No explanation, no units, just the JSON array."
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
    Requires:  pip install google-genai Pillow
    """

    tag = "gemini15pro"

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash"):
        try:
            from google import genai as _genai
            from google.genai import types as _gtypes
        except ImportError:
            raise ImportError("pip install google-genai")

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self._client = _genai.Client(api_key=api_key)
        self._model  = model
        self._gtypes = _gtypes

    def _call(self, image_path: str, prompt: str) -> str:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pip install Pillow")

        img = Image.open(image_path).convert("RGB")
        response = self._client.models.generate_content(
            model   = self._model,
            contents= [img, prompt],
            config  = self._gtypes.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,   # thinking model needs headroom
            ),
        )
        return (response.text or "").strip()

    def _call_with_retry(self, image_path: str, prompt: str,
                         max_retries: int = 5) -> str:
        import time
        delay = 15
        for attempt in range(max_retries):
            try:
                return self._call(image_path, prompt)
            except Exception as e:
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    if attempt < max_retries - 1:
                        logger.info(f"Gemini transient error (attempt {attempt+1}), "
                                    f"retrying in {delay}s …")
                        time.sleep(delay)
                        delay = min(delay * 2, 120)
                        continue
                raise
        return ""

    def predict_distances(self, image_path, object_pairs):
        """
        Single batched API call for all pairs in one image.
        Accepts partial arrays (NaN-pads if Gemini returns fewer values than pairs).
        Adds a 13-second inter-scene sleep to stay within ~5 RPM free-tier limit.
        """
        import time
        if not object_pairs:
            return []

        # --- Single batched call ---
        try:
            raw = self._call_with_retry(
                image_path, self._batch_distance_prompt(object_pairs)
            )
            # Extract JSON array from response
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) > 0:
                    results = []
                    for v in parsed:
                        try:
                            results.append(float(v))
                        except (TypeError, ValueError):
                            results.append(float("nan"))
                    # Pad with NaN if Gemini returned fewer values
                    if len(results) < len(object_pairs):
                        logger.warning(
                            f"Batch response: got {len(results)} values for "
                            f"{len(object_pairs)} pairs — NaN-padding remainder"
                        )
                        results += [float("nan")] * (len(object_pairs) - len(results))
                    else:
                        results = results[:len(object_pairs)]
                    logger.info(f"Batch call OK: {len(object_pairs)} pairs, {sum(1 for r in results if r==r)} valid")
                    time.sleep(13)   # ~5 RPM
                    return results
            logger.warning(f"Batch call: could not parse JSON array from response: {raw[:120]!r}")
        except Exception as e:
            logger.warning(f"Gemini batched call failed: {e}")

        # --- Fallback: return all NaN (don't do per-pair to avoid quota exhaustion) ---
        logger.warning(f"Batch call failed for {len(object_pairs)} pairs — returning NaN")
        time.sleep(13)
        return [float("nan")] * len(object_pairs)

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call_with_retry(image_path,
                                        self._relation_prompt(question, choices))
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
    InternVL2-8B local inference baseline — no rate limits.
    Matches 'InternVL2-8B' (58.11) in the Spatial457 paper comparison table.
    Requires:  pip install transformers torch torchvision Pillow timm einops
    Model:     OpenGVLab/InternVL2-8B  (~16 GB VRAM, bfloat16)
    """

    tag = "internvl2-8b"

    # Official InternVL2 preprocessing constants
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    _INPUT_SIZE    = 448

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-8B",
                 device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel
        import transformers.modeling_utils as _mu
        import torch

        # InternVL2 uses transformers 4.x API; patch for compatibility with 5.x
        def _compat_getattr(self, name):
            if name == "all_tied_weights_keys":
                return getattr(self, "_tied_weights_keys", [])
            if name == "language_model":
                return None
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        _mu.PreTrainedModel.__getattr__ = _compat_getattr

        logger.info(f"Loading InternVL2 from {model_name} …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        self._device = device
        logger.info("InternVL2 loaded.")

    def _preprocess(self, image_path: str):
        from PIL import Image
        import torch
        from torchvision import transforms

        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])
        return transform(img).unsqueeze(0).to(torch.bfloat16).to(self._device)

    def _call(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        pixel_values = self._preprocess(image_path)
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question=prompt,
            generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
        )
        return (response or "").strip()

    @staticmethod
    def _batch_distance_prompt(pairs):
        lines = "\n".join(f"{i+1}. '{a}' to '{b}'" for i, (a, b) in enumerate(pairs))
        return (
            "Look at the image carefully. For each numbered pair of objects below, "
            "estimate the real-world 3D distance between them in metres.\n\n"
            f"{lines}\n\n"
            "Reply with ONLY a JSON array of numbers in the same order, e.g. "
            "[1.2, 3.4, 0.8]. No explanation, no units, just the JSON array."
        )

    def predict_distances(self, image_path, object_pairs):
        if not object_pairs:
            return []
        try:
            raw = self._call(image_path, self._batch_distance_prompt(object_pairs),
                             max_new_tokens=512)
            start = raw.find("["); end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) > 0:
                    results = [float(v) if v == v else float("nan") for v in parsed]
                    if len(results) < len(object_pairs):
                        logger.warning(
                            f"InternVL2 batch: got {len(results)}/{len(object_pairs)} "
                            f"values — NaN-padding"
                        )
                        results += [float("nan")] * (len(object_pairs) - len(results))
                    return results[:len(object_pairs)]
            logger.warning(f"InternVL2 batch: no JSON array in: {raw[:100]!r}")
        except Exception as e:
            logger.warning(f"InternVL2 batch call failed: {e}")
        return [float("nan")] * len(object_pairs)

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
    LLaVA-NeXT (LLaVA-1.6 / vicuna-7b) local inference baseline — no rate limits.
    Matches the 'LLaVA-NeXT-vicuna-7B' row in the Spatial457 paper comparison table.
    Requires:  pip install transformers torch Pillow accelerate
    Model:     llava-hf/llava-v1.6-vicuna-7b-hf  (~14 GB VRAM)
    """

    tag = "llava-next-7b"

    def __init__(self,
                 model_name: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
                 device: str = "cuda"):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch

        logger.info(f"Loading LLaVA-NeXT from {model_name} …")
        self._processor = LlavaNextProcessor.from_pretrained(model_name)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self._model.eval()
        self._device = device
        logger.info("LLaVA-NeXT loaded.")

    def _call(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        formatted = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = self._processor(
            text=formatted, images=img, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        # Decode only the generated tokens (skip input)
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _batch_distance_prompt(pairs):
        lines = "\n".join(f"{i+1}. '{a}' to '{b}'" for i, (a, b) in enumerate(pairs))
        return (
            "Look at the image carefully. For each numbered pair of objects below, "
            "estimate the real-world 3D distance between them in metres.\n\n"
            f"{lines}\n\n"
            "Reply with ONLY a JSON array of numbers in the same order, e.g. "
            "[1.2, 3.4, 0.8]. No explanation, no units, just the JSON array."
        )

    def predict_distances(self, image_path, object_pairs):
        if not object_pairs:
            return []
        try:
            # All pairs in one forward pass
            raw = self._call(
                image_path,
                self._batch_distance_prompt(object_pairs),
                max_new_tokens=512,
            )
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) > 0:
                    results = []
                    for v in parsed:
                        try:
                            results.append(float(v))
                        except (TypeError, ValueError):
                            results.append(float("nan"))
                    if len(results) < len(object_pairs):
                        logger.warning(
                            f"LLaVA batch: got {len(results)} values for "
                            f"{len(object_pairs)} pairs — NaN-padding"
                        )
                        results += [float("nan")] * (len(object_pairs) - len(results))
                    return results[:len(object_pairs)]
            logger.warning(f"LLaVA batch: could not parse JSON: {raw[:100]!r}")
        except Exception as e:
            logger.warning(f"LLaVA-NeXT batch call failed: {e}")
        return [float("nan")] * len(object_pairs)

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
# Qwen2-VL-7B-Instruct baseline (local, open-weights)
# ---------------------------------------------------------------------------

class Qwen2VLBaseline(VLMBaseline):
    """
    Qwen2-VL-7B-Instruct local inference baseline — no rate limits.
    Qwen2-VL is explicitly designed for spatial/metric understanding.
    Requires:  pip install transformers torch Pillow qwen-vl-utils
    Model:     Qwen/Qwen2-VL-7B-Instruct  (~15 GB VRAM, bfloat16)
    """

    tag = "qwen2-vl-7b"

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = "cuda"):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        logger.info(f"Loading Qwen2-VL from {model_name} …")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self._device = device
        logger.info("Qwen2-VL loaded.")

    def _call(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[img], return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Strip input tokens
        gen_ids = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
        return self._processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @staticmethod
    def _batch_distance_prompt(pairs):
        lines = "\n".join(f"{i+1}. '{a}' to '{b}'" for i, (a, b) in enumerate(pairs))
        return (
            "Look at the image carefully. For each numbered pair of objects below, "
            "estimate the real-world 3D distance between them in metres.\n\n"
            f"{lines}\n\n"
            "Reply with ONLY a JSON array of numbers in the same order, e.g. "
            "[1.2, 3.4, 0.8]. No explanation, no units, just the JSON array."
        )

    def predict_distances(self, image_path, object_pairs):
        if not object_pairs:
            return []
        try:
            raw = self._call(image_path, self._batch_distance_prompt(object_pairs),
                             max_new_tokens=512)
            start = raw.find("["); end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list) and len(parsed) > 0:
                    results = [float(v) if v == v else float("nan") for v in parsed]
                    if len(results) < len(object_pairs):
                        logger.warning(
                            f"Qwen2-VL batch: got {len(results)}/{len(object_pairs)} "
                            f"values — NaN-padding"
                        )
                        results += [float("nan")] * (len(object_pairs) - len(results))
                    return results[:len(object_pairs)]
            logger.warning(f"Qwen2-VL batch: no JSON array in: {raw[:100]!r}")
        except Exception as e:
            logger.warning(f"Qwen2-VL batch call failed: {e}")
        return [float("nan")] * len(object_pairs)

    def predict_relation(self, image_path, question, choices):
        try:
            raw = self._call(image_path, self._relation_prompt(question, choices))
            letter = raw.strip()[0].upper() if raw.strip() else "A"
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception as e:
            logger.warning(f"Qwen2-VL relation call failed: {e}")
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
        # Plausible metre-scale predictions (0.5–8 m, matching GNN training units)
        return [float(self._rng.uniform(0.5, 8.0)) for _ in object_pairs]

    def predict_relation(self, image_path, question, choices):
        return self._rng.choice(choices)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: Dict[str, type] = {
    "gpt4o":          GPT4oBaseline,
    "gemini15pro":    GeminiBaseline,
    "internvl2-8b":   InternVL2Baseline,
    "qwen2-vl-7b":    Qwen2VLBaseline,
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
