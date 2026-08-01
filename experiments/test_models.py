import torch

from models.registry import get_model
from utils.image import load_image


IMAGE_PATH = "dog.jpg"

MODEL_NAMES = [
    "resnet50",
    "vit_b_16",
    "convnext_tiny",
]


print()
print("=" * 60)
print("MODEL BASELINE TEST")
print("=" * 60)


# ============================================================
# LOAD IMAGE
# ============================================================

_, image = load_image(
    IMAGE_PATH
)


# ============================================================
# TEST EACH MODEL
# ============================================================

for model_name in MODEL_NAMES:

    model_info = get_model(
        model_name
    )

    image_device = image.to(
        model_info.device
    )

    with torch.no_grad():

        output = model_info.predict(
            image_device
        )

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    confidence, class_id = (
        probabilities.max(dim=1)
    )

    class_id = class_id.item()
    confidence = confidence.item()

    category = model_info.weights.meta[
        "categories"
    ][class_id]

    print()
    print("=" * 60)
    print(
        f"MODEL: {model_name}"
    )
    print("=" * 60)

    print()
    print(
        "Prediction:",
        category,
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%",
    )


print()
print("=" * 60)
print("Baseline test complete.")
print("=" * 60)