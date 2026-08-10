"""
Black-box adapters for frontier VLMs via their hosted APIs.

All adapters:
  - Read API keys from environment variables
  - Use plain ``requests`` (no extra SDKs)
  - Share a common ``score()`` interface so the transfer
    test experiment can use any of them interchangeably

Supported providers:
    GLMAdapter     - Zhipu BigModel (GLM-4.5V / GLM-4V-Plus)
    GLM5Adapter    - Zhipu BigModel (GLM-5V-Turbo)
    OpenAIAdapter  - OpenAI (gpt-4o, gpt-4o-mini)
    GeminiAdapter  - Google AI Studio (Gemini 2.5 Pro / Flash)
"""

import base64
import io
import os
import time
from typing import List, Optional

import requests
from PIL import Image

from models.base import BlackBoxVLMAdapter
from models.api_adapter import DEFAULT_KEYWORDS, SUCCESS_SCORE


# ============================================================
# SHARED HELPERS
# ============================================================

def encode_image_b64(pil_image: Image.Image,
                     fmt: str = "JPEG",
                     quality: int = 90) -> str:
    """Encode PIL image as a base64 string (no data URL prefix)."""
    buf = io.BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image_data_url(pil_image: Image.Image,
                          fmt: str = "JPEG",
                          quality: int = 90) -> str:
    """Encode PIL image as a base64 data URL (data:image/jpeg;base64,...)."""
    b64 = encode_image_b64(pil_image, fmt=fmt, quality=quality)
    mime = "jpeg" if fmt.upper() == "JPG" else fmt.lower()
    return f"data:image/{mime};base64,{b64}"


def character_score(text: str,
                    keywords: Optional[List[str]] = None,
                    target_keyword: Optional[str] = None) -> float:
    """Higher = better for attacker (no dog keyword, target keyword present)."""
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    text_lower = (text or "").lower().strip()

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
        return SUCCESS_SCORE + len(text_lower) * 0.5 + target_bonus
    return float(min_pos) - dog_count * 5.0 + target_bonus


# ============================================================
# ZHIPU BIGMODEL - GLM-4.5V
# ============================================================

class GLMAdapter(BlackBoxVLMAdapter):
    """Zhipu BigModel adapter (GLM-4.5V / GLM-4V-Plus).

    API docs: https://bigmodel.cn/dev/api/vision/glm-4v
    Endpoint: OpenAI-compatible ``/api/paas/v4/chat/completions``
    Auth:     ``ZHIPU_API_KEY`` env var (Bearer token)
    """

    DEFAULT_HOST = "https://open.bigmodel.cn"
    DEFAULT_MODEL = "glm-4.5v"

    def __init__(self,
                 name: str = "glm-4.5v",
                 host: Optional[str] = None,
                 model_name: Optional[str] = None,
                 image_size: int = 378,
                 api_key: Optional[str] = None,
                 question: str = "What do you see in this image?",
                 timeout: int = 120):
        api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ZHIPU_API_KEY env var is required for GLMAdapter"
            )
        super().__init__(
            name=name,
            host=(host or self.DEFAULT_HOST).rstrip("/"),
            model_name=model_name or self.DEFAULT_MODEL,
            device="cpu",
            image_size=image_size,
            api_type="zhipu",
        )
        self.api_key = api_key
        self.question = question
        self.timeout = timeout

    def query(self,
              pil_image: Image.Image,
              question: Optional[str] = None,
              max_tokens: int = 100,
              temperature: float = 0.1) -> str:
        q = question or self.question
        img_b64 = encode_image_data_url(pil_image)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": img_b64}},
                        {"type": "text", "text": q},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.host}/api/paas/v4/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    self.query_count += 1
                    return resp.json()["choices"][0]["message"]["content"].strip()
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return f"<error {resp.status_code}: {resp.text[:200]}>"
            except Exception as exc:
                if attempt == 2:
                    return f"<error: {exc}>"
                time.sleep(2)
        return "<error: max retries>"

    def score(self,
              pil_image: Image.Image,
              keywords: Optional[List[str]] = None,
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        text = self.query(pil_image, max_tokens=max_tokens)
        return character_score(text, keywords=keywords, target_keyword=target_keyword)

    def describe(self,
                 pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 200) -> str:
        return self.query(pil_image, question=question, max_tokens=max_tokens)

    def classify(self,
                 pil_image: Image.Image,
                 source_text: str,
                 target_text: str) -> dict:
        """A/B classification via open-ended describe + keyword match."""
        desc = self.describe(pil_image, max_tokens=200)
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

    def classify(self,
                 pil_image: Image.Image,
                 source_text: str,
                 target_text: str) -> dict:
        """A/B classification via open-ended describe + keyword match."""
        desc = self.describe(pil_image, max_tokens=200)
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


# ============================================================
# ZHIPU BIGMODEL - GLM-5V-Turbo
# ============================================================

class GLM5Adapter(GLMAdapter):
    """Zhipu BigModel adapter for GLM-5V-Turbo.

    Same endpoint, transport, and auth as :class:`GLMAdapter`; only the
    default model id differs. GLM-5V-Turbo is Zhipu's multimodal coding
    base model in the GLM-5 family (vision-language; ``glm-5`` itself
    is text-only).

    API docs: https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo
    Endpoint: OpenAI-compatible ``/api/paas/v4/chat/completions``
    Auth:     ``ZHIPU_API_KEY`` env var (Bearer token)
    """

    DEFAULT_MODEL = "glm-5v-turbo"

    def __init__(self,
                 name: str = "glm-5v-turbo",
                 host: Optional[str] = None,
                 model_name: Optional[str] = None,
                 image_size: int = 378,
                 api_key: Optional[str] = None,
                 question: str = "What do you see in this image?",
                 timeout: int = 120):
        super().__init__(
            name=name,
            host=host,
            model_name=model_name,
            image_size=image_size,
            api_key=api_key,
            question=question,
            timeout=timeout,
        )


# ============================================================
# OPENAI - GPT-4o
# ============================================================

class OpenAIAdapter(BlackBoxVLMAdapter):
    """OpenAI Chat Completions adapter (gpt-4o / gpt-4o-mini).

    API docs: https://platform.openai.com/docs/guides/vision
    Endpoint: ``https://api.openai.com/v1/chat/completions``
    Auth:     ``OPENAI_API_KEY`` env var (Bearer token)
    """

    DEFAULT_HOST = "https://api.openai.com"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self,
                 name: str = "gpt-4o",
                 host: Optional[str] = None,
                 model_name: Optional[str] = None,
                 image_size: int = 378,
                 api_key: Optional[str] = None,
                 question: str = "What do you see in this image?",
                 timeout: int = 120):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY env var is required for OpenAIAdapter"
            )
        super().__init__(
            name=name,
            host=(host or self.DEFAULT_HOST).rstrip("/"),
            model_name=model_name or self.DEFAULT_MODEL,
            device="cpu",
            image_size=image_size,
            api_type="openai",
        )
        self.api_key = api_key
        self.question = question
        self.timeout = timeout

    def query(self,
              pil_image: Image.Image,
              question: Optional[str] = None,
              max_tokens: int = 100,
              temperature: float = 0.1) -> str:
        q = question or self.question
        img_b64 = encode_image_data_url(pil_image)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {"type": "image_url",
                         "image_url": {"url": img_b64}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.host}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    self.query_count += 1
                    return resp.json()["choices"][0]["message"]["content"].strip()
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return f"<error {resp.status_code}: {resp.text[:200]}>"
            except Exception as exc:
                if attempt == 2:
                    return f"<error: {exc}>"
                time.sleep(2)
        return "<error: max retries>"

    def score(self,
              pil_image: Image.Image,
              keywords: Optional[List[str]] = None,
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        text = self.query(pil_image, max_tokens=max_tokens)
        return character_score(text, keywords=keywords, target_keyword=target_keyword)

    def describe(self,
                 pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 200) -> str:
        return self.query(pil_image, question=question, max_tokens=max_tokens)

    def classify(self,
                 pil_image: Image.Image,
                 source_text: str,
                 target_text: str) -> dict:
        """A/B classification via open-ended describe + keyword match."""
        desc = self.describe(pil_image, max_tokens=200)
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

    def classify(self,
                 pil_image: Image.Image,
                 source_text: str,
                 target_text: str) -> dict:
        """A/B classification via open-ended describe + keyword match."""
        desc = self.describe(pil_image, max_tokens=200)
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


# ============================================================
# GOOGLE AI STUDIO - GEMINI
# ============================================================

class GeminiAdapter(BlackBoxVLMAdapter):
    """Google AI Studio adapter (Gemini 2.5 Pro / Flash).

    API docs: https://ai.google.dev/gemini-api/docs/vision
    Endpoint: ``/v1beta/models/{model}:generateContent?key=...``
    Auth:     ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) env var
               passed as a query-string parameter
    """

    DEFAULT_HOST = "https://generativelanguage.googleapis.com"
    DEFAULT_MODEL = "gemini-2.5-pro"

    def __init__(self,
                 name: str = "gemini-2.5-pro",
                 host: Optional[str] = None,
                 model_name: Optional[str] = None,
                 image_size: int = 378,
                 api_key: Optional[str] = None,
                 question: str = "What do you see in this image?",
                 timeout: int = 120):
        api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) env var is "
                "required for GeminiAdapter"
            )
        super().__init__(
            name=name,
            host=(host or self.DEFAULT_HOST).rstrip("/"),
            model_name=model_name or self.DEFAULT_MODEL,
            device="cpu",
            image_size=image_size,
            api_type="gemini",
        )
        self.api_key = api_key
        self.question = question
        self.timeout = timeout

    def query(self,
              pil_image: Image.Image,
              question: Optional[str] = None,
              max_tokens: int = 100,
              temperature: float = 0.1) -> str:
        q = question or self.question
        img_b64 = encode_image_b64(pil_image)

        url = (
            f"{self.host}/v1beta/models/{self.model_name}"
            f":generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"text": q},
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    url, json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.query_count += 1
                    try:
                        return (
                            data["candidates"][0]["content"]["parts"][0]["text"]
                            .strip()
                        )
                    except (KeyError, IndexError, TypeError):
                        return "<empty response>"
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return f"<error {resp.status_code}: {resp.text[:200]}>"
            except Exception as exc:
                if attempt == 2:
                    return f"<error: {exc}>"
                time.sleep(2)
        return "<error: max retries>"

    def score(self,
              pil_image: Image.Image,
              keywords: Optional[List[str]] = None,
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        text = self.query(pil_image, max_tokens=max_tokens)
        return character_score(text, keywords=keywords, target_keyword=target_keyword)

    def describe(self,
                 pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 200) -> str:
        return self.query(pil_image, question=question, max_tokens=max_tokens)

    def classify(self,
                 pil_image: Image.Image,
                 source_text: str,
                 target_text: str) -> dict:
        """A/B classification via open-ended describe + keyword match."""
        desc = self.describe(pil_image, max_tokens=200)
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
