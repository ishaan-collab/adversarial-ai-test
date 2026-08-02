import numpy as np
import torch

from PIL import Image

from models.vlm_registry import get_vlm
from attacks.vlm_ensemble_pgd import vlm_ensemble_pgd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

SOURCE_TEXT = "a photo of a golden retriever"
TARGET_TEXT = "a photo of a Norfolk terrier"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 20
SEED = 42
RANDOM_START = False

MODEL_NAMES = [
    "clip",
    "siglip",
]


# ============================================================
# LOAD IMAGE
# ============================================================

pil_image = (
    Image.open(IMAGE_PATH)
    .convert("RGB")
)

image = torch.from_numpy(
    np.array(pil_image)
).permute(
    2, 0, 1
).float() / 255.0

image = image.unsqueeze(0)


# ============================================================
# LOAD MODELS
# ============================================================

models = {
    name: get_vlm(name)
    for name in MODEL_NAMES
}


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_attack(
    attack_name,
    attack_models,
):
    """
    Generate one adversarial image using attack_models,
    then evaluate it against every available model.
    """

    print()
    print("=" * 70)
    print(f"ATTACK: {attack_name}")
    print("=" * 70)

    print()
    print("Attack models:")

    for model in attack_models:
        print(f"  - {model.name}")

    # --------------------------------------------------------
    # Generate adversarial image
    # --------------------------------------------------------

    adversarial = vlm_ensemble_pgd(
        models=attack_models,
        image=image.to(
            attack_models[0].device
        ),
        source_text=SOURCE_TEXT,
        target_text=TARGET_TEXT,
        epsilon=EPSILON,
        alpha=ALPHA,
        seed=SEED,
        iterations=ITERATIONS,
        random_start=RANDOM_START,
    )

    # --------------------------------------------------------
    # Convert tensor -> PIL
    # --------------------------------------------------------

    adv_tensor = (
        adversarial[0]
        .detach()
        .cpu()
        .clamp(0, 1)
    )

    adv_array = (
        adv_tensor
        .permute(1, 2, 0)
        .numpy()
        * 255
    )

    adv_image = Image.fromarray(
        adv_array.astype(np.uint8)
    )

    # --------------------------------------------------------
    # Evaluate against ALL models
    # --------------------------------------------------------

    results = {}

    print()
    print("TRANSFER EVALUATION")
    print("-" * 70)

    for model_name, model in models.items():

        result = model.predict(
            image=adv_image,
            texts=[
                SOURCE_TEXT,
                TARGET_TEXT,
            ],
        )

        success = (
            result["prediction"]
            == TARGET_TEXT
        )

        results[model_name] = success

        print()
        print(
            f"MODEL: {model.name}"
        )

        print(
            f"Prediction: "
            f"{result['prediction']}"
        )

        for text, score in zip(
            [
                SOURCE_TEXT,
                TARGET_TEXT,
            ],
            result["scores"],
        ):

            print(
                f"  {score.item():.6f}  "
                f"{text}"
            )

        print(
            "Target achieved:",
            "YES" if success else "NO",
        )

    # --------------------------------------------------------
    # Perturbation
    # --------------------------------------------------------

    perturbation = (
        adversarial
        - image.to(adversarial.device)
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successes = sum(
        results.values()
    )

    success_rate = (
        successes
        / len(models)
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Models fooled : "
        f"{successes}/{len(models)}"
    )

    print(
        f"Success rate  : "
        f"{success_rate * 100:.2f}%"
    )

    print(
        f"L∞            : "
        f"{linf:.8f}"
    )

    print(
        f"L2            : "
        f"{l2:.8f}"
    )

    return {
        "attack": attack_name,
        "results": results,
        "success_rate": success_rate,
        "linf": linf,
        "l2": l2,
    }


# ============================================================
# RUN COMPARISON
# ============================================================

clip_result = evaluate_attack(
    attack_name="CLIP ONLY",
    attack_models=[
        models["clip"],
    ],
)

siglip_result = evaluate_attack(
    attack_name="SIGLIP ONLY",
    attack_models=[
        models["siglip"],
    ],
)

ensemble_result = evaluate_attack(
    attack_name="CLIP + SIGLIP ENSEMBLE",
    attack_models=[
        models["clip"],
        models["siglip"],
    ],
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Attack':<25}"
    f"{'CLIP':<12}"
    f"{'SigLIP':<12}"
    f"{'Overall':<12}"
)

print("-" * 70)

for result in [
    clip_result,
    siglip_result,
    ensemble_result,
]:

    clip_success = (
        "YES"
        if result["results"]["clip"]
        else "NO"
    )

    siglip_success = (
        "YES"
        if result["results"]["siglip"]
        else "NO"
    )

    print(
        f"{result['attack']:<25}"
        f"{clip_success:<12}"
        f"{siglip_success:<12}"
        f"{result['success_rate'] * 100:.2f}%"
    )

print()
print("=" * 70)
print("Comparison complete.")
print("=" * 70)
