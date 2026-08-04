import numpy as np
import torch
from PIL import Image

from attacks.vlm_pgd import (
    targeted_clip_pgd,
)

from models.vlm_registry import (
    get_vlm,
)

from utils.image import (
    save_tensor_as_image,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

OUTPUT_PATH = (
    "outputs/vlm_clip_pgd.png"
)

MODEL_NAME = "clip"

SOURCE_TEXT = (
    "a photo of a golden retriever"
)

TARGET_TEXT = (
    "a photo of a Norfolk terrier"
)

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 20


# ============================================================
# HELPERS
# ============================================================

def pil_to_tensor(
    image,
    device,
):
    """
    Convert PIL RGB image to [1,3,H,W]
    tensor in [0,1].
    """

    array = np.array(
        image,
        dtype=np.float32,
    )

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        / 255.0
    )

    return tensor.unsqueeze(0).to(
        device
    )


def print_scores(
    result,
    texts,
):
    """
    Print image-text scores.
    """

    for text, score in zip(
        texts,
        result["scores"],
    ):

        print(
            f"  {score.item():.6f}  "
            f"{text}"
        )


# ============================================================
# LOAD IMAGE
# ============================================================

pil_image = (
    Image.open(
        IMAGE_PATH
    )
    .convert("RGB")
)


# ============================================================
# LOAD MODEL
# ============================================================

model = get_vlm(
    MODEL_NAME
)


# ============================================================
# TEXTS
# ============================================================

texts = [
    SOURCE_TEXT,
    TARGET_TEXT,
]


# ============================================================
# CLEAN EVALUATION
# ============================================================

clean = model.predict(
    image=pil_image,
    texts=texts,
)


print()
print("=" * 70)
print("CLIP TARGETED PGD")
print("=" * 70)

print()

print(
    "Image:",
    IMAGE_PATH,
)

print(
    "Model:",
    MODEL_NAME,
)

print()

print("Source:")
print(
    f"  {SOURCE_TEXT}"
)

print()

print("Target:")
print(
    f"  {TARGET_TEXT}"
)

print()

print("Clean prediction:")
print(
    f"  {clean['prediction']}"
)

print()

print("Clean scores:")

print_scores(
    clean,
    texts,
)


# ============================================================
# CONVERT IMAGE
# ============================================================

image = pil_to_tensor(
    pil_image,
    model.device,
)


# ============================================================
# GENERATE ADVERSARIAL IMAGE
# ============================================================

print()
print("=" * 70)
print("GENERATING ADVERSARIAL EXAMPLE")
print("=" * 70)

print()

print(
    f"Epsilon    : {EPSILON:.8f}"
)

print(
    f"Alpha      : {ALPHA:.8f}"
)

print(
    f"Iterations : {ITERATIONS}"
)

print()

print("Objective:")

print(
    f"  {SOURCE_TEXT}"
)

print(
    "        ↓"
)

print(
    f"  {TARGET_TEXT}"
)

print()

adversarial_image = (
    targeted_clip_pgd(
        model=model,
        image=image,
        source_text=SOURCE_TEXT,
        target_text=TARGET_TEXT,
        epsilon=EPSILON,
        alpha=ALPHA,
        iterations=ITERATIONS,
    )
)


# ============================================================
# SAVE ADVERSARIAL IMAGE
# ============================================================

save_tensor_as_image(
    adversarial_image,
    OUTPUT_PATH,
)

print()

print(
    "Saved adversarial image:"
)

print(
    f"  {OUTPUT_PATH}"
)


# ============================================================
# CONVERT ADVERSARIAL IMAGE TO PIL
# ============================================================

adv_tensor = (
    adversarial_image[0]
    .detach()
    .cpu()
    .clamp(0.0, 1.0)
)

adv_array = (
    adv_tensor
    .permute(1, 2, 0)
    .numpy()
)

adv_array = (
    adv_array * 255.0
).round().astype(
    np.uint8
)

adv_image = Image.fromarray(
    adv_array,
    mode="RGB",
)


# ============================================================
# ADVERSARIAL EVALUATION
# ============================================================

result = model.predict(
    image=adv_image,
    texts=texts,
)


# ============================================================
# PERTURBATION METRICS
# ============================================================

perturbation = (
    adversarial_image
    - image
)

linf = (
    perturbation
    .abs()
    .max()
    .item()
)

l2 = torch.norm(
    perturbation.reshape(
        perturbation.shape[0],
        -1,
    ),
    p=2,
    dim=1,
).item()

mean_abs = (
    perturbation
    .abs()
    .mean()
    .item()
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("ADVERSARIAL RESULTS")
print("=" * 70)

print()

print(
    "Adversarial prediction:"
)

print(
    f"  {result['prediction']}"
)

print()

print(
    "Adversarial scores:"
)

print_scores(
    result,
    texts,
)

print()

target_achieved = (
    result["prediction"]
    == TARGET_TEXT
)

print(
    "Target achieved:",
    "YES"
    if target_achieved
    else "NO",
)

print()

print(
    "Perturbation metrics:"
)

print(
    f"  L∞                  : "
    f"{linf:.8f}"
)

print(
    f"  L2                  : "
    f"{l2:.8f}"
)

print(
    f"  Mean |perturbation|  : "
    f"{mean_abs:.8f}"
)

print()

print(
    "Budget verification:"
)

print(
    f"  Configured epsilon  : "
    f"{EPSILON:.8f}"
)

print(
    f"  Actual L∞           : "
    f"{linf:.8f}"
)

print(
    "  Within budget       :",
    "YES"
    if linf <= EPSILON + 1e-6
    else "NO",
)

print()

print("=" * 70)
print("Experiment complete.")
print("=" * 70)