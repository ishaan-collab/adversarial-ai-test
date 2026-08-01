import torch

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from models.adapter import ModelAdapter


def load_model():
    """
    Load the default ResNet-50 model
    and wrap it in a ModelAdapter.
    """

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights
    )

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    preprocess = weights.transforms()

    adapter = ModelAdapter(
        model=model,
        weights=weights,
        device=device,
        preprocess=preprocess,
        name="resnet50",
    )

    return adapter