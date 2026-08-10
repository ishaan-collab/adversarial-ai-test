"""
Unified VLM Registry.

Single entry point for loading any VLM — white-box or black-box.

White-box models (HF, gradient access):
    - moondream2 (1B)
    - llava-1.5-7b (7B)
    - llava-1.5-13b (13B)
    - llava-next-7b (7B)
    - qwen2-vl-2b (2B)
    - qwen2-vl-7b (7B)
    - qwen2.5-vl-3b (3B)
    - qwen2.5-vl-7b (7B)
    - clip (vision-language, not generative)
    - siglip (vision-language, not generative)

Black-box models (API, no gradients):
    - Any OpenAI-compatible API (llama-server, vLLM, etc.)
    - Any Ollama-served VLM

Usage:
    from models.vlm_registry import get_vlm

    # White-box
    adapter = get_vlm("llava-1.5-7b", mode="whitebox")

    # Black-box (API)
    adapter = get_vlm("qwen-vyas", mode="blackbox",
                      host="http://127.0.0.1:11471",
                      model_name="vyas")

    # Black-box (Ollama)
    adapter = get_vlm("ollama-moondream", mode="blackbox",
                      host="http://127.0.0.1:11435",
                      model_name="moondream", api_type="ollama")
"""

import torch
from typing import Optional
from models.base import BaseVLMAdapter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# WHITE-BOX MODEL REGISTRY
# ============================================================

WHITEBOX_MODELS = {
    # Moondream (existing)
    "moondream2": {
        "module": "models.moondream_adapter",
        "class": "MoondreamAdapter",
        "params_b": 1.0,
        "image_size": 378,
    },

    # LLaVA family
    "llava-1.5-7b": {
        "module": "models.llava_adapter",
        "class": "LlavaAdapter",
        "kwargs": {"model_name": "llava-1.5-7b"},
        "params_b": 7.0,
        "image_size": 336,
    },
    "llava-1.5-13b": {
        "module": "models.llava_adapter",
        "class": "LlavaAdapter",
        "kwargs": {"model_name": "llava-1.5-13b"},
        "params_b": 13.0,
        "image_size": 336,
    },
    "llava-next-7b": {
        "module": "models.llava_adapter",
        "class": "LlavaAdapter",
        "kwargs": {"model_name": "llava-next-7b"},
        "params_b": 7.0,
        "image_size": 336,
    },

    # Qwen2-VL family
    "qwen2-vl-2b": {
        "module": "models.qwen_vl_adapter",
        "class": "QwenVLAdapter",
        "kwargs": {"model_name": "qwen2-vl-2b"},
        "params_b": 2.0,
        "image_size": 448,
    },
    "qwen2-vl-7b": {
        "module": "models.qwen_vl_adapter",
        "class": "QwenVLAdapter",
        "kwargs": {"model_name": "qwen2-vl-7b"},
        "params_b": 7.0,
        "image_size": 448,
    },
    "qwen2.5-vl-3b": {
        "module": "models.qwen_vl_adapter",
        "class": "QwenVLAdapter",
        "kwargs": {"model_name": "qwen2.5-vl-3b"},
        "params_b": 3.0,
        "image_size": 448,
    },
    "qwen2.5-vl-7b": {
        "module": "models.qwen_vl_adapter",
        "class": "QwenVLAdapter",
        "kwargs": {"model_name": "qwen2.5-vl-7b"},
        "params_b": 7.0,
        "image_size": 448,
    },

    # CLIP / SigLIP (existing)
    "clip": {
        "module": "models.vlm_registry",
        "class": "_load_clip",
        "params_b": 0.15,
        "image_size": 224,
    },
    "siglip": {
        "module": "models.vlm_registry",
        "class": "_load_siglip",
        "params_b": 0.11,
        "image_size": 224,
    },
}


# ============================================================
# BLACK-BOX MODEL CONFIGS (pre-defined endpoints)
# ============================================================

BLACKBOX_CONFIGS = {
    "qwen-vyas": {
        "host": "http://127.0.0.1:11471",
        "model_name": "vyas",
        "api_type": "openai",
        "image_size": 378,
    },
    "ollama-moondream": {
        "host": "http://127.0.0.1:11435",
        "model_name": "moondream",
        "api_type": "ollama",
        "image_size": 378,
    },
}


# ============================================================
# EXISTING CLIP/SIGLIP LOADERS (kept for backward compat)
# ============================================================

def _load_clip():
    from transformers import CLIPModel, CLIPProcessor
    from models.vision_language import VisionLanguageAdapter

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return VisionLanguageAdapter(
        model=model, processor=processor,
        name="clip", device=DEVICE, model_type="clip",
    )


def _load_siglip():
    from transformers import SiglipModel, SiglipProcessor
    from models.vision_language import VisionLanguageAdapter

    model_id = "google/siglip-base-patch16-224"
    model = SiglipModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    processor = SiglipProcessor.from_pretrained(model_id)
    return VisionLanguageAdapter(
        model=model, processor=processor,
        name="siglip", device=DEVICE, model_type="siglip",
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def get_vlm(name: str, mode: str = "whitebox", **kwargs) -> BaseVLMAdapter:
    """
    Load a VLM adapter by name.

    Args:
        name: Model name (see WHITEBOX_MODELS / BLACKBOX_CONFIGS)
        mode: "whitebox" or "blackbox"
        **kwargs: Override config (host, model_name, etc. for blackbox)

    Returns:
        BaseVLMAdapter (WhiteBoxVLMAdapter or BlackBoxVLMAdapter)
    """
    name = name.lower()

    if mode == "whitebox":
        return _load_whitebox(name, **kwargs)
    elif mode == "blackbox":
        return _load_blackbox(name, **kwargs)
    else:
        raise ValueError(f"mode must be 'whitebox' or 'blackbox', got '{mode}'")


def _load_whitebox(name: str, **kwargs) -> BaseVLMAdapter:
    """Load a white-box model from HuggingFace."""
    if name not in WHITEBOX_MODELS:
        available = ", ".join(WHITEBOX_MODELS.keys())
        raise ValueError(f"Unknown white-box model '{name}'. Available: {available}")

    config = WHITEBOX_MODELS[name]
    cls_name = config["class"]

    if cls_name == "_load_clip":
        return _load_clip()
    elif cls_name == "_load_siglip":
        return _load_siglip()

    import importlib
    module = importlib.import_module(config["module"])
    cls = getattr(module, cls_name)

    init_kwargs = config.get("kwargs", {})
    init_kwargs.update(kwargs)

    return cls(**init_kwargs)


def _load_blackbox(name: str, **kwargs) -> BaseVLMAdapter:
    """Load a black-box adapter for an API endpoint."""
    from models.api_adapter import APIVLMAdapter, OllamaVLMAdapter2

    config = BLACKBOX_CONFIGS.get(name, {}).copy()
    config.update(kwargs)

    host = config.get("host", "http://127.0.0.1:11471")
    model_name = config.get("model_name", name)
    api_type = config.get("api_type", "openai")
    image_size = config.get("image_size", 378)

    if api_type == "ollama":
        return OllamaVLMAdapter2(
            name=name, host=host, model_name=model_name,
            image_size=image_size,
        )
    else:
        return APIVLMAdapter(
            name=name, host=host, model_name=model_name,
            image_size=image_size, api_type=api_type,
        )


def list_models():
    """Print all available models."""
    print("\nWhite-box models (gradient access):")
    print("-" * 50)
    for name, config in WHITEBOX_MODELS.items():
        print(f"  {name:<20} {config['params_b']:.1f}B params")

    print("\nBlack-box models (API endpoints):")
    print("-" * 50)
    for name, config in BLACKBOX_CONFIGS.items():
        print(f"  {name:<20} {config['api_type']:<8} {config['host']}")

    print("\nUsage:")
    print("  from models.vlm_registry import get_vlm")
    print("  adapter = get_vlm('llava-1.5-7b', mode='whitebox')")
    print("  adapter = get_vlm('qwen-vyas', mode='blackbox')")


if __name__ == "__main__":
    list_models()
