import os
import numpy as np
import torch

from PIL import Image

from models.vlm_registry import get_vlm
from attacks.vlm_ensemble_pgd import vlm_ensemble_pgd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/vlm"

SOURCE_TEXT = "a photo of a dog"
TARGET_TEXT = "a photo of a cat"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 20

MODEL_NAMES = [
    "clip",
    "siglip",
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)


# ============================================================
# HELPERS
# ============================================================

def load_image(path):
    """
    Load an image as:

        PIL image
        [1, 3, H, W] float tensor in [0, 1]
    """

    pil_image = (
        Image.open(path)
        .convert("RGB")
    )

    image = torch.from_numpy(
        np.array(pil_image)
    ).permute(
        2, 0, 1
    ).float() / 255.0

    return (
        pil_image,
        image.unsqueeze(0),
    )


def tensor_to_pil(tensor):
    """
    Convert [1, 3, H, W] tensor to PIL image.
    """

    tensor = (
        tensor[0]
        .detach()
        .cpu()
        .clamp(0, 1)
    )

    array = (
        tensor
        .permute(1, 2, 0)
        .numpy()
        * 255
    )

    return Image.fromarray(
        array.astype(np.uint8)
    )


def get_image_paths():
    """
    Return all supported images from IMAGE_DIR.
    """

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{IMAGE_DIR}"
        )

    paths = []

    for filename in sorted(
        os.listdir(IMAGE_DIR)
    ):

        path = os.path.join(
            IMAGE_DIR,
            filename,
        )

        if (
            os.path.isfile(path)
            and filename.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ):
            paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    return paths


def calculate_metrics(
    original,
    adversarial,
):
    """
    Calculate perturbation metrics.
    """

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

    return {
        "linf": linf,
        "l2": l2,
        "mean_abs": mean_abs,
    }


# ============================================================
# EVALUATE IMAGE
# ============================================================

def evaluate_image(
    model,
    pil_image,
):
    """
    Evaluate one image against source and target prompts.
    """

    result = model.predict(
        image=pil_image,
        texts=[
            SOURCE_TEXT,
            TARGET_TEXT,
        ],
    )

    success = (
        result["prediction"]
        == TARGET_TEXT
    )

    return result, success


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VLM MULTI-IMAGE ENSEMBLE EVALUATION")
    print("=" * 70)

    print()
    print("Image directory:")
    print(f"  {IMAGE_DIR}")

    print()
    print("Source:")
    print(f"  {SOURCE_TEXT}")

    print()
    print("Target:")
    print(f"  {TARGET_TEXT}")

    print()
    print("Attack models:")

    for name in MODEL_NAMES:
        print(f"  - {name}")

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_paths = get_image_paths()

    print()
    print(
        f"Images found: {len(image_paths)}"
    )

    for path in image_paths:
        print(
            f"  - {os.path.basename(path)}"
        )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    models = []

    print()

    for name in MODEL_NAMES:

        models.append(
            get_vlm(name)
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        filename = os.path.basename(
            image_path
        )

        print()
        print("=" * 70)
        print(
            f"IMAGE {index}/{len(image_paths)}: "
            f"{filename}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        try:

            pil_image, image = load_image(
                image_path
            )

        except Exception as exc:

            print(
                f"ERROR loading {filename}: "
                f"{exc}"
            )

            continue

        # ----------------------------------------------------
        # CLEAN EVALUATION
        # ----------------------------------------------------

        print()
        print("CLEAN EVALUATION")
        print("-" * 70)

        clean_results = {}

        for model in models:

            result, success = (
                evaluate_image(
                    model,
                    pil_image,
                )
            )

            clean_results[
                model.name
            ] = {
                "prediction":
                    result["prediction"],

                "source_score":
                    result["scores"][0].item(),

                "target_score":
                    result["scores"][1].item(),

                "target":
                    success,
            }

            print()
            print(
                f"MODEL: {model.name}"
            )

            print(
                f"Prediction: "
                f"{result['prediction']}"
            )

            print(
                f"  source="
                f"{result['scores'][0].item():.6f}"
            )

            print(
                f"  target="
                f"{result['scores'][1].item():.6f}"
            )

        # ----------------------------------------------------
        # ATTACK
        # ----------------------------------------------------

        print()
        print("GENERATING ENSEMBLE ADVERSARIAL IMAGE")
        print("-" * 70)

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
            seed=42,
            random_start=False,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            image.to(adversarial.device),
            adversarial,
        )

        print()
        print("PERTURBATION")
        print(
            f"L∞        : "
            f"{metrics['linf']:.8f}"
        )

        print(
            f"L2        : "
            f"{metrics['l2']:.8f}"
        )

        print(
            f"Mean |δ|  : "
            f"{metrics['mean_abs']:.8f}"
        )

        # ----------------------------------------------------
        # SAVE ADVERSARIAL IMAGE
        # ----------------------------------------------------

        output_dir = (
            "outputs/vlm_dataset"
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        output_path = os.path.join(
            output_dir,
            f"adv_{filename}",
        )

        adversarial_image = (
            tensor_to_pil(adversarial)
        )

        adversarial_image.save(
            output_path
        )

        print()
        print(
            "Saved:"
        )

        print(
            f"  {output_path}"
        )

        # ----------------------------------------------------
        # ADVERSARIAL EVALUATION
        # ----------------------------------------------------

        print()
        print("ADVERSARIAL EVALUATION")
        print("-" * 70)

        adversarial_results = {}

        fooled = 0

        for model in models:

            result, success = (
                evaluate_image(
                    model,
                    adversarial_image,
                )
            )

            if success:
                fooled += 1

            adversarial_results[
                model.name
            ] = {
                "prediction":
                    result["prediction"],

                "source_score":
                    result["scores"][0].item(),

                "target_score":
                    result["scores"][1].item(),

                "target":
                    success,
            }

            print()
            print(
                f"MODEL: {model.name}"
            )

            print(
                f"Prediction: "
                f"{result['prediction']}"
            )

            print(
                f"  source="
                f"{result['scores'][0].item():.6f}"
            )

            print(
                f"  target="
                f"{result['scores'][1].item():.6f}"
            )

            print(
                "Target achieved: "
                + (
                    "YES"
                    if success
                    else "NO"
                )
            )

        # ----------------------------------------------------
        # IMAGE RESULT
        # ----------------------------------------------------

        image_success_rate = (
            fooled / len(models)
        )

        results.append(
            {
                "filename": filename,
                "clean": clean_results,
                "adversarial":
                    adversarial_results,
                "fooled": fooled,
                "success_rate":
                    image_success_rate,
                "linf":
                    metrics["linf"],
                "l2":
                    metrics["l2"],
                "mean_abs":
                    metrics["mean_abs"],
            }
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if not results:

        print()
        print(
            "No images were successfully "
            "processed."
        )

        return

    total_evaluations = (
        len(results)
        * len(models)
    )

    total_successes = sum(
        result["fooled"]
        for result in results
    )

    overall_success_rate = (
        total_successes
        / total_evaluations
    )

    average_linf = np.mean(
        [
            result["linf"]
            for result in results
        ]
    )

    average_l2 = np.mean(
        [
            result["l2"]
            for result in results
        ]
    )

    average_mean_abs = np.mean(
        [
            result["mean_abs"]
            for result in results
        ]
    )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL DATASET RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Image':<25}"
        f"{'CLIP':<10}"
        f"{'SigLIP':<10}"
        f"{'Success':<10}"
    )

    print("-" * 70)

    for result in results:

        clip_success = (
            result["adversarial"]
            ["clip"]["target"]
        )

        siglip_success = (
            result["adversarial"]
            ["siglip"]["target"]
        )

        print(
            f"{result['filename']:<25}"
            f"{'YES' if clip_success else 'NO':<10}"
            f"{'YES' if siglip_success else 'NO':<10}"
            f"{result['success_rate'] * 100:.0f}%"
        )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print()

    print(
        f"Images evaluated      : "
        f"{len(results)}"
    )

    print(
        f"Models per image      : "
        f"{len(models)}"
    )

    print(
        f"Total evaluations     : "
        f"{total_evaluations}"
    )

    print(
        f"Successful targets    : "
        f"{total_successes}"
    )

    print(
        f"Overall success rate  : "
        f"{overall_success_rate * 100:.2f}%"
    )

    print()
    print(
        f"Average L∞            : "
        f"{average_linf:.8f}"
    )

    print(
        f"Average L2            : "
        f"{average_l2:.8f}"
    )

    print(
        f"Average Mean |δ|      : "
        f"{average_mean_abs:.8f}"
    )

    print()
    print("=" * 70)
    print("Dataset evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
