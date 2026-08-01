import torch

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,

    vit_b_16,
    ViT_B_16_Weights,
    vit_l_16,
    ViT_L_16_Weights,

    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,

    swin_b,
    Swin_B_Weights,
)

from models.adapter import ModelAdapter


# ============================================================
# DEVICE
# ============================================================

def _get_device():
    """
    Select CUDA when available, otherwise CPU.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# RESNET50
# ============================================================

def load_resnet50(device):

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="resnet50",
    )


# ============================================================
# RESNET101
# ============================================================

def load_resnet101(device):

    weights = ResNet101_Weights.DEFAULT

    model = resnet101(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="resnet101",
    )


# ============================================================
# VIT-B/16
# ============================================================

def load_vit_b_16(device):

    weights = ViT_B_16_Weights.DEFAULT

    model = vit_b_16(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="vit_b_16",
    )


# ============================================================
# VIT-L/16
# ============================================================

def load_vit_l_16(device):

    weights = ViT_L_16_Weights.DEFAULT

    model = vit_l_16(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="vit_l_16",
    )


# ============================================================
# CONVNEXT-TINY
# ============================================================

def load_convnext_tiny(device):

    weights = ConvNeXt_Tiny_Weights.DEFAULT

    model = convnext_tiny(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="convnext_tiny",
    )


# ============================================================
# CONVNEXT-BASE
# ============================================================

def load_convnext_base(device):

    weights = ConvNeXt_Base_Weights.DEFAULT

    model = convnext_base(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="convnext_base",
    )


# ============================================================
# SWIN-B
# ============================================================

def load_swin_b(device):

    weights = Swin_B_Weights.DEFAULT

    model = swin_b(
        weights=weights
    )

    model.eval()
    model = model.to(device)

    return ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=weights.transforms(),
        name="swin_b",
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_LOADERS = {

    # CNN
    "resnet50": load_resnet50,
    "resnet101": load_resnet101,

    # Vision Transformers
    "vit_b_16": load_vit_b_16,
    "vit_l_16": load_vit_l_16,

    # ConvNeXt
    "convnext_tiny": load_convnext_tiny,
    "convnext_base": load_convnext_base,

    # Swin Transformer
    "swin_b": load_swin_b,
}


# ============================================================
# PUBLIC API
# ============================================================

def get_model(name):
    """
    Load a model by registry name.
    """

    name = name.lower()

    if name not in MODEL_LOADERS:

        available = ", ".join(
            MODEL_LOADERS.keys()
        )

        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {available}"
        )

    device = _get_device()

    return MODEL_LOADERS[name](
        device
    )