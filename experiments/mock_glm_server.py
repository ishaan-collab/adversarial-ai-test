"""
Mock GLM-4.5V server for offline transfer testing.

Exposes the same OpenAI-compatible endpoint that the real Zhipu
BigModel API exposes, but answers every image question by running
a locally cached ResNet50 (torchvision) and converting the top-1
prediction into a short text description.

Why ResNet50?
    - Already cached at ~/.cache/torch/hub/checkpoints/resnet50-*.pth
    - Runs in ~50ms per image on CPU
    - ImageNet-1k has 118 dog breeds in classes 151-268, so we can
      legitimately check whether the perturbation fooled the model
      into predicting something other than a dog breed.

This is "virtual" only in the sense that it does not call Zhipu's
commercial API. The vision model is real, the transfer evaluation
is real, only the chat-completions transport is replaced.

Run:
    python experiments/mock_glm_server.py --port 11880

Then point the experiment at it:
    ZHIPU_API_KEY=mock python experiments/frontier_transfer_test.py \\
        --models glm --glm-host http://127.0.0.1:11880
"""

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image


# ============================================================
# IMAGENET DOG BREEDS (classes 151-268)
# ============================================================

DOG_CLASS_IDS = set(range(151, 269))  # 118 dog breeds

# Subset of names we actually need to format descriptions nicely.
# Full list is fine to print — pulled from torchvision weights metadata.
def _load_imagenet_labels() -> list:
    """Try to load ImageNet class labels from torchvision weights metadata."""
    try:
        weights = tvm.ResNet50_Weights.DEFAULT
        categories = weights.meta.get("categories", [])
        if categories:
            return categories
    except Exception:
        pass
    # Fallback: empty list. We will still know which IDs are dogs.
    return [f"class_{i}" for i in range(1000)]


IMAGENET_LABELS = _load_imagenet_labels()


# ============================================================
# MODEL
# ============================================================

_MODEL_LOCK = threading.Lock()
_MODEL = None
_TRANSFORM = None


def _load_model():
    global _MODEL, _TRANSFORM
    if _MODEL is not None:
        return _MODEL, _TRANSFORM
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _TRANSFORM
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
        model.eval()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _MODEL = model
        _TRANSFORM = transform
    return _MODEL, _TRANSFORM


def _classify(pil_image: Image.Image):
    """Return (top1_class_id, top1_name, top5_ids, top5_names, confidence)."""
    model, transform = _load_model()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    with torch.no_grad():
        x = transform(pil_image).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, k=5)
        top5_ids = top5.indices.tolist()
        top5_probs = top5.values.tolist()
    top5_names = [
        IMAGENET_LABELS[i] if i < len(IMAGENET_LABELS) else f"class_{i}"
        for i in top5_ids
    ]
    return (
        top5_ids[0],
        top5_names[0],
        top5_ids,
        top5_names,
        top5_probs,
    )


def _describe(pil_image: Image.Image, question: str) -> str:
    """Turn a ResNet50 prediction into a free-form text answer."""
    top1_id, top1_name, top5_ids, top5_names, top5_probs = _classify(pil_image)
    is_dog = top1_id in DOG_CLASS_IDS

    if is_dog:
        return (
            f"I see a {top1_name.replace('_', ' ')} (a dog breed). "
            f"It is clearly a canine. "
            f"Top-5: {', '.join(top5_names)}."
        )
    return (
        f"I see a {top1_name.replace('_', ' ')}. "
        f"It does not appear to be a dog. "
        f"Top-5: {', '.join(top5_names)}."
    )


# ============================================================
# HTTP HANDLER
# ============================================================

class GLMMockHandler(BaseHTTPRequestHandler):
    """Handles POST /api/paas/v4/chat/completions in OpenAI-compatible format."""

    def log_message(self, fmt, *args):
        sys.stderr.write(
            f"[mock-glm] {self.address_string()} - {fmt % args}\n"
        )

    def do_POST(self):
        if self.path.rstrip("/") != "/api/paas/v4/chat/completions":
            self.send_error(404, f"unknown path: {self.path}")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except Exception as exc:
            self._send_json(400, {"error": f"bad json: {exc}"})
            return

        try:
            messages = payload.get("messages", [])
            pil_image, question = self._extract(messages)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
            return

        try:
            answer = _describe(pil_image, question)
        except Exception as exc:
            self._send_json(500, {"error": f"inference failed: {exc}"})
            return

        response = {
            "id": "mock-chatcmpl-0001",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "glm-4.5v"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        }
        self._send_json(200, response)

    def _extract(self, messages):
        """Pull first image (base64 data URL or raw b64) + first text from messages."""
        pil_image = None
        question = "What do you see in this image?"
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                question = content
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "text" and not question:
                    question = part.get("text", question)
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    pil_image = self._decode(url)
                elif part.get("type") == "image":
                    pil_image = self._decode(part.get("image", ""))
            if pil_image is not None:
                break
        if pil_image is None:
            raise ValueError("no image provided")
        return pil_image, question

    @staticmethod
    def _decode(url: str) -> Image.Image:
        if url.startswith("data:"):
            comma = url.find(",")
            b64 = url[comma + 1:]
        else:
            b64 = url
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw))

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mock GLM server (ResNet50)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11880)
    parser.add_argument("--preload", action="store_true",
                        help="Load ResNet50 at startup instead of on first request.")
    args = parser.parse_args()

    if args.preload:
        print("[mock-glm] loading ResNet50...", flush=True)
        _load_model()
        print("[mock-glm] model ready.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), GLMMockHandler)
    print(f"[mock-glm] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[mock-glm] endpoint: POST /api/paas/v4/chat/completions",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-glm] shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
