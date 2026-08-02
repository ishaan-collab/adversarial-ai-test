import numpy as np
import torch

from PIL import Image

from attacks.vlm_ensemble_pgd import vlm_ensemble_pgd
from models.vlm_registry import get_vlm
from utils.image import save_tensor_as_image


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

SOURCE_TEXT = (
    "a photo of a golden retriever"
)

TARGET_TEXT = (
    "a photo of a Norfolk terrier"
)

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 20

SEED = 42
RANDOM_START = False

MODEL_NAMES = [
    "clip",
    "siglip",
]

OUTPUT_PATH = (
    "outputs/vlm_ensemble_pgd.png"
)


# ============================================================
# LOAD IMAGE
# ============================================================

pil_image = (
    Image.open(IMAGE_PATH)
    .convert("RGB")
)

image = (
    torch.from_numpy(
        np.array(pil_image)
    )
    .permute(2, 0, 1)
    .float()
    / 255.0
)

image = image.unsqueeze(0)


# ============================================================
# LOAD MODELS
# ============================================================

models = [
    get_vlm(name)
    for name in MODEL_NAMES
]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("VLM ENSEMBLE TARGETED PGD")
print("=" * 70)

print()
print("Image:")
print(f"  {IMAGE_PATH}")

print()
print("Source:")
print(f"  {SOURCE_TEXT}")

print()
print("Target:")
print(f"  {TARGET_TEXT}")

print()
print("Models:")

for model in models:
    print(f"  - {model.name}")


# ============================================================
# CLEAN EVALUATION
# ============================================================

print()
print("=" * 70)
print("CLEAN EVALUATION")
print("=" * 70)

texts = [
    SOURCE_TEXT,
    TARGET_TEXT,
]

for model in models:

    result = model.predict(
        image=pil_image,
        texts=texts,
    )

    print()
    print(f"MODEL: {model.name}")

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    for text, score in zip(
        texts,
        result["scores"],
    ):

        print(
            f"  {score.item():.6f}  "
            f"{text}"
        )


# ============================================================
# GENERATE SHARED ADVERSARIAL IMAGE
# ============================================================

print()
print("=" * 70)
print("GENERATING SHARED ADVERSARIAL IMAGE")
print("=" * 70)

print()
print(
    f"Epsilon      : {EPSILON:.8f}"
)

print(
    f"Alpha        : {ALPHA:.8f}"
)

print(
    f"Iterations   : {ITERATIONS}"
)

print(
    f"Seed         : {SEED}"
)

print(
    f"Random start : {RANDOM_START}"
)

print()
print("Objective:")

print(
    f"  {SOURCE_TEXT}"
)

print("        ↓")

print(
    f"  {TARGET_TEXT}"
)

print()
print("ENSEMBLE MEMBERS")
print("-" * 60)

for model in models:
    print(f"  {model.name}")


adversarial = vlm_ensemble_pgd(
    models=models,
    image=image.to(
        models[0].device
    ),
    source_text=SOURCE_TEXT,
    target_text=TARGET_TEXT,
    epsilon=EPSILON,
    alpha=ALPHA,
    iterations=ITERATIONS,
    seed=SEED,
    random_start=RANDOM_START,
)


# ============================================================
# SAVE ADVERSARIAL IMAGE
# ============================================================

save_tensor_as_image(
    adversarial,
    OUTPUT_PATH,
)

print()
print("Saved adversarial image:")
print(f"  {OUTPUT_PATH}")


# ============================================================
# PERTURBATION METRICS
# ============================================================

original = image.to(
    adversarial.device
)

perturbation = (
    adversarial
    - original
)

linf = (
    perturbation
    .abs()
    .max()
    .item()
)

l2 = (
    torch.norm(
        perturbation.reshape(
            perturbation.shape[0],
            -1,
        ),
        p=2,
        dim=1,
    )
    .item()
)

mean_abs = (
    perturbation
    .abs()
    .mean()
    .item()
)


# ============================================================
# CONVERT ADVERSARIAL IMAGE ONCE
# ============================================================

adv_tensor = (
    adversarial[0]
    .detach()
    .cpu()
    .clamp(0.0, 1.0)
)

adv_array = (
    adv_tensor
    .permute(1, 2, 0)
    .numpy()
    * 255.0
)

adv_image = Image.fromarray(
    adv_array.astype(np.uint8)
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

successes = 0

for model in models:

    result = model.predict(
        image=adv_image,
        texts=texts,
    )

    success = (
        result["prediction"]
        == TARGET_TEXT
    )

    if success:
        successes += 1

    print()
    print("-" * 70)

    print(
        f"MODEL: {model.name}"
    )

    print()
    print("Adversarial prediction:")

    print(
        f"  {result['prediction']}"
    )

    print()
    print("Scores:")

    for text, score in zip(
        texts,
        result["scores"],
    ):

        print(
            f"  {score.item():.6f}  "
            f"{text}"
        )

    print()

    print(
        "Target achieved:",
        "YES" if success else "NO",
    )


# ============================================================
# SUCCESS RATE
# ============================================================

success_rate = (
    successes
    / len(models)
)


# ============================================================
# PERTURBATION REPORT
# ============================================================

print()
print("=" * 70)
print("PERTURBATION")
print("=" * 70)

print(
    f"L∞                  : "
    f"{linf:.8f}"
)

print(
    f"L2                  : "
    f"{l2:.8f}"
)

print(
    f"Mean |perturbation| : "
    f"{mean_abs:.8f}"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Models attacked      : "
    f"{len(models)}"
)

print(
    f"Models fooled        : "
    f"{successes}"
)

print(
    f"Target success rate  : "
    f"{success_rate * 100:.2f}%"
)

print(
    f"Configured epsilon   : "
    f"{EPSILON:.8f}"
)

print(
    f"Actual L∞            : "
    f"{linf:.8f}"
)

print(
    "Within budget        : "
    f"{'YES' if linf <= EPSILON + 1e-6 else 'NO'}"
)

print()
print(
    "Experiment complete."
)

print("=" * 70)