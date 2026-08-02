from PIL import Image

from models.vlm_registry import get_vlm


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

TEXTS = [
    "a photo of a golden retriever",
    "a photo of a Norfolk terrier",
    "a photo of a cat",
    "a photo of a car",
]


# ============================================================
# IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# TEST MODELS
# ============================================================

for model_name in [
    "clip",
    "siglip",
]:

    print()
    print("=" * 70)
    print(
        f"{model_name.upper()} TEST"
    )
    print("=" * 70)

    model = get_vlm(
        model_name
    )

    result = model.score_image_text(
        image=image,
        texts=TEXTS,
    )

    print()

    print(
        "Image-text scores:"
    )

    for text, score, probability in zip(
        result["texts"],
        result["scores"],
        result["probabilities"],
    ):

        print(
            f"{score.item():>12.6f}  "
            f"{probability.item():>12.6f}  "
            f"{text}"
        )

    print()

    print(
        "Prediction:",
        result["prediction"],
    )