"""
Generic black-box adapter for any OpenAI-compatible VLM API.

Works with:
    - llama-server (Qwen, LLaVA, etc.)
    - vLLM serving endpoint
    - LocalAI
    - Any OpenAI-compatible /v1/chat/completions endpoint

No gradient access — used for SPSA, Square Attack, Genetic algorithms.
"""

import io
import base64
import time
import requests
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from models.base import BlackBoxVLMAdapter


DEFAULT_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]

SUCCESS_SCORE = 200.0


class APIVLMAdapter(BlackBoxVLMAdapter):
    """
    Black-box adapter for any OpenAI-compatible VLM API.

    Supports images via base64 data URLs in chat completions.
    """

    def __init__(self, name: str, host: str, model_name: str,
                 image_size: int = 378,
                 api_type: str = "openai",
                 question: str = "What do you see in this image?",
                 image_format: str = "JPEG",
                 image_quality: int = 90):
        super().__init__(
            name=name, host=host, model_name=model_name,
            device="cpu", image_size=image_size, api_type=api_type,
        )
        self.question = question
        self.image_format = image_format
        self.image_quality = image_quality

    def _encode_image(self, pil_image: Image.Image) -> str:
        """Encode PIL image as base64 data URL."""
        buf = io.BytesIO()
        pil_image.save(buf, format=self.image_format, quality=self.image_quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = self.image_format.lower()
        if mime == "jpg":
            mime = "jpeg"
        return f"data:image/{mime};base64,{b64}"

    def query(self, pil_image: Image.Image,
              question: Optional[str] = None,
              max_tokens: int = 50,
              temperature: float = 0.1) -> str:
        """Send image + question to API, return text response."""
        q = question or self.question
        img_b64 = self._encode_image(pil_image)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url",
                     "image_url": {"url": img_b64}},
                ]}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.host}/v1/chat/completions",
                    json=payload, timeout=120,
                )
                if resp.status_code == 200:
                    self.query_count += 1
                    return resp.json()["choices"][0]["message"]["content"].strip()
                elif resp.status_code == 503:
                    time.sleep(5)
                    continue
                else:
                    return f"<error {resp.status_code}>"
            except Exception as e:
                if attempt == 2:
                    return f"<error: {e}>"
                time.sleep(2)
        return "<error: max retries>"

    def score(self, pil_image: Image.Image,
              keywords: Optional[List[str]] = None,
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        """
        Character-position scoring.
        Higher = better for attacker (dog keyword pushed later or absent).
        """
        if keywords is None:
            keywords = DEFAULT_KEYWORDS

        text = self.query(pil_image, max_tokens=max_tokens)
        text_lower = text.lower().strip()

        if not text_lower:
            return SUCCESS_SCORE

        min_pos = len(text_lower)
        for kw in keywords:
            pos = text_lower.find(kw)
            if 0 <= pos < min_pos:
                min_pos = pos

        dog_count = sum(text_lower.count(kw) for kw in keywords)
        target_bonus = 50.0 if target_keyword and target_keyword in text_lower else 0.0

        if min_pos >= len(text_lower):
            score = SUCCESS_SCORE + len(text_lower) * 0.5 + target_bonus
        else:
            score = float(min_pos) - dog_count * 5.0 + target_bonus

        return score

    def describe(self, pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 100) -> str:
        """Generate a description."""
        return self.query(pil_image, question=question, max_tokens=max_tokens)

    def classify(self, pil_image: Image.Image,
                 source_text: str, target_text: str) -> Dict[str, Any]:
        """A/B classification based on keywords."""
        desc = self.describe(pil_image)
        desc_lower = desc.lower()

        has_source = source_text.lower() in desc_lower
        has_target = target_text.lower() in desc_lower

        return {
            "description": desc,
            "source_match": has_source,
            "target_match": has_target,
            "classification": "target" if has_target and not has_source
                            else ("source" if has_source else "neither"),
        }


class OllamaVLMAdapter2(BlackBoxVLMAdapter):
    """
    Black-box adapter for Ollama-served VLMs.
    Uses Ollama's native /api/chat endpoint.
    """

    def __init__(self, name: str = "ollama-moondream",
                 host: str = "http://127.0.0.1:11435",
                 model_name: str = "moondream",
                 image_size: int = 378):
        super().__init__(
            name=name, host=host, model_name=model_name,
            device="cpu", image_size=image_size, api_type="ollama",
        )

    def _encode_image(self, pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def query(self, pil_image: Image.Image,
              question: Optional[str] = None,
              max_tokens: int = 50,
              temperature: float = 0.1) -> str:
        """Query Ollama /api/chat endpoint."""
        q = question or "What do you see in this image?"
        img_b64 = self._encode_image(pil_image)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": q, "images": [img_b64]}
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat", json=payload, timeout=120,
                )
                if resp.status_code == 200:
                    self.query_count += 1
                    return resp.json()["message"]["content"].strip()
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    return f"<error: {e}>"
                time.sleep(2)
        return "<error: max retries>"

    def score(self, pil_image: Image.Image,
              keywords: Optional[List[str]] = None,
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        """Character-position scoring."""
        if keywords is None:
            keywords = DEFAULT_KEYWORDS

        text = self.query(pil_image, max_tokens=max_tokens)
        text_lower = text.lower().strip()

        if not text_lower:
            return SUCCESS_SCORE

        min_pos = len(text_lower)
        for kw in keywords:
            pos = text_lower.find(kw)
            if 0 <= pos < min_pos:
                min_pos = pos

        dog_count = sum(text_lower.count(kw) for kw in keywords)
        target_bonus = 50.0 if target_keyword and target_keyword in text_lower else 0.0

        if min_pos >= len(text_lower):
            score = SUCCESS_SCORE + len(text_lower) * 0.5 + target_bonus
        else:
            score = float(min_pos) - dog_count * 5.0 + target_bonus

        return score

    def describe(self, pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 100) -> str:
        return self.query(pil_image, question=question, max_tokens=max_tokens)

    def classify(self, pil_image: Image.Image,
                 source_text: str, target_text: str) -> Dict[str, Any]:
        desc = self.describe(pil_image)
        desc_lower = desc.lower()
        has_source = source_text.lower() in desc_lower
        has_target = target_text.lower() in desc_lower
        return {
            "description": desc,
            "source_match": has_source,
            "target_match": has_target,
            "classification": "target" if has_target and not has_source
                            else ("source" if has_source else "neither"),
        }
