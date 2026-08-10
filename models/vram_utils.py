"""
VRAM management utilities.

Handles model loading/unloading, precision selection,
and memory monitoring for large VLMs (up to 10B+ parameters).
"""

import gc
import torch
from typing import Optional, Literal


def get_vram_info() -> dict:
    """Return current VRAM usage info."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free = total - reserved

    return {
        "available": True,
        "device": props.name,
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": free / 1e9,
    }


def print_vram_status(label: str = ""):
    """Print current VRAM status."""
    info = get_vram_info()
    if not info["available"]:
        print(f"  [VRAM] CUDA not available")
        return
    tag = f" [{label}]" if label else ""
    print(f"  [VRAM]{tag} {info['allocated_gb']:.1f} / {info['total_gb']:.1f} GB "
          f"allocated, {info['free_gb']:.1f} GB free")


def auto_select_dtype(
    model_params_billions: float,
    overhead_factor: float = 2.5,
) -> torch.dtype:
    """
    Auto-select dtype based on model size and available VRAM.

    bf16: 2 bytes/param
    float16: 2 bytes/param (not recommended for some models)
    4-bit: ~0.5 bytes/param (requires bitsandbytes)

    Args:
        model_params_billions: model size in billions (e.g. 7.0 for 7B)
        overhead_factor: multiplier for activations, gradients, etc.
    Returns:
        torch.dtype to use
    """
    info = get_vram_info()
    if not info["available"]:
        return torch.float32

    free_gb = info["free_gb"]

    # bf16 needs 2 bytes/param × overhead
    bf16_gb = model_params_billions * 2 * overhead_factor
    if bf16_gb < free_gb:
        return torch.bfloat16

    # 4-bit would need ~0.5 bytes/param × overhead
    q4_gb = model_params_billions * 0.5 * overhead_factor
    if q4_gb < free_gb:
        print(f"  [VRAM] bf16 needs {bf16_gb:.1f} GB, only {free_gb:.1f} GB free")
        print(f"  [VRAM] Would use 4-bit ({q4_gb:.1f} GB) but bitsandbytes not available")
        print(f"  [VRAM] Falling back to bf16 — may OOM")
        return torch.bfloat16

    return torch.bfloat16


def free_vram():
    """Aggressively free VRAM: collect garbage, empty cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_hf_model(
    model_class,
    model_id: str,
    dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    **kwargs,
):
    """
    Load a HuggingFace model with VRAM management.

    Args:
        model_class: e.g. LlavaForConditionalGeneration
        model_id: HuggingFace model ID
        dtype: torch dtype (auto-selected if None)
        device_map: "auto" or specific device
        **kwargs: passed to from_pretrained
    Returns:
        Loaded model
    """
    if dtype is None:
        dtype = torch.bfloat16

    print(f"  [LOAD] {model_id}")
    print(f"  [LOAD] dtype={dtype}, device_map={device_map}")
    print_vram_status("before")

    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    model.eval()

    print_vram_status("after")
    return model


def unload_model(model):
    """Unload a model and free VRAM."""
    if model is not None:
        del model
    free_vram()
    print_vram_status("after unload")
