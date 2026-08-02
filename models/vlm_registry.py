import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
)

from models.vision_language import (
    VisionLanguageAdapter,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_LOADERS = {

    "clip": (
        "openai/clip-vit-base-patch32",
        "clip",
    ),

    "siglip": (
        "google/siglip-base-patch16-224",
        "siglip",
    ),
}


def get_vlm(name):

    name = name.lower()

    if name not in MODEL_LOADERS:

        available = ", ".join(
            MODEL_LOADERS.keys()
        )

        raise ValueError(
            f"Unknown VLM '{name}'. "
            f"Available: {available}"
        )

    model_name, model_type = (
        MODEL_LOADERS[name]
    )

    print(
        f"Loading {name}: {model_name}"
    )

    if model_type == "clip":

        model = (
            CLIPModel
            .from_pretrained(model_name)
        )

        processor = (
            CLIPProcessor
            .from_pretrained(model_name)
        )

    else:

        model = (
            SiglipModel
            .from_pretrained(model_name)
        )

        processor = (
            SiglipProcessor
            .from_pretrained(model_name)
        )

    model = model.to(DEVICE)
    model.eval()

    return VisionLanguageAdapter(
        model=model,
        processor=processor,
        name=name,
        device=DEVICE,
        model_type=model_type,
    )
