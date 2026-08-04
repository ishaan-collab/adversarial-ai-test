"""
Transfer Attack Evaluation: White-Box Ensemble → Black-Box Ollama VLM

This experiment demonstrates adversarial transferability:
  1. Generate adversarial images using white-box surrogate models
     (CLIP + SigLIP ensemble via targeted PGD)
  2. Query the black-box Ollama moondream model (1B params, no gradient access)
  3. Measure whether the adversarial perturbation causes the black-box
     model to misclassify the image

Attack pipeline:
  - Surrogate models: CLIP + SigLIP (white-box, gradient access)
  - Target model: Ollama moondream (black-box, API-only)
  - Attack: Ensemble targeted PGD (source="a photo of a dog" → target="a photo of a cat")

Evaluation:
  - Clean image → moondream: should correctly identify as dog
  - Adversarial image → moondream: does it say cat? (transfer success)
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from attacks.vlm_ensemble_pgd import vlm_ensemble_pgd
from models.ollama_adapter import OllamaVLMAdapter
from models.vlm_registry import get_vlm


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

SOURCE_TEXT = "a photo of a dog"
TARGET_TEXT = "a photo of a cat"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 40
SEED = 42

SURROGATE_MODELS = ["clip", "siglip"]

OLLAMA_HOST = "http://127.0.0.1:11435"
OLLAMA_MODEL = "moondream"

OUTPUT_DIR = "outputs/ollama_transfer"

# Dataset for multi-image evaluation
DATASET_DIR = "data/vlm"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# ============================================================
# HELPERS
# ============================================================

def load_image_raw(path):
    """
    Load an image as PIL + [1, 3, H, W] float tensor in [0, 1]
    without torchvision transforms (raw resolution).
    """

    pil_image = Image.open(path).convert("RGB")
    tensor = (
        torch.from_numpy(np.array(pil_image))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    return pil_image, tensor.unsqueeze(0)


def tensor_to_pil(tensor):
    """Convert [1, 3, H, W] tensor to PIL."""

    tensor = tensor[0].detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def calculate_perturbation(original, adversarial):
    """Compute L∞, L2, and mean absolute perturbation."""

    perturbation = adversarial - original
    linf = perturbation.abs().max().item()
    l2 = (
        torch.norm(
            perturbation.reshape(perturbation.shape[0], -1),
            p=2,
            dim=1,
        ).item()
    )
    mean_abs = perturbation.abs().mean().item()
    return {"linf": linf, "l2": l2, "mean_abs": mean_abs}


def get_dataset_paths():
    """Return sorted image paths from DATASET_DIR."""

    if not os.path.isdir(DATASET_DIR):
        return []

    paths = []
    for filename in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, filename)
        if (
            os.path.isfile(path)
            and filename.lower().endswith(IMAGE_EXTENSIONS)
        ):
            paths.append(path)
    return paths


# ============================================================
# SINGLE IMAGE EVALUATION
# ============================================================

def evaluate_single_image(
    image_path,
    surrogate_models,
    ollama_adapter,
    output_dir,
):
    """
    Full transfer evaluation for one image:
      1. Clean → surrogates (verify correct classification)
      2. Clean → Ollama (baseline)
      3. Generate adversarial via ensemble PGD on surrogates
      4. Adversarial → surrogates (white-box success)
      5. Adversarial → Ollama (black-box transfer success)
    """

    filename = os.path.basename(image_path)

    print()
    print("=" * 70)
    print(f"IMAGE: {filename}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    pil_image, image_tensor = load_image_raw(image_path)

    print(f"  Image size: {pil_image.size}")

    # --------------------------------------------------------
    # STEP 1: CLEAN EVALUATION ON SURROGATES
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 1: CLEAN EVALUATION ON SURROGATE MODELS")
    print("-" * 70)

    texts = [SOURCE_TEXT, TARGET_TEXT]

    for model in surrogate_models:
        result = model.predict(image=pil_image, texts=texts)
        print(
            f"  {model.name:<10} → "
            f"{result['prediction']}  "
            f"(source={result['scores'][0].item():.4f}, "
            f"target={result['scores'][1].item():.4f})"
        )

    # --------------------------------------------------------
    # STEP 2: CLEAN EVALUATION ON OLLAMA (BLACK-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 2: CLEAN EVALUATION ON OLLAMA (BLACK-BOX)")
    print("-" * 70)

    clean_desc = ollama_adapter.describe_image(pil_image)
    print(f"  Description: {clean_desc}")

    clean_class = ollama_adapter.classify_image(
        pil_image, SOURCE_TEXT, TARGET_TEXT
    )
    print(
        f"  Classification: {clean_class['prediction']} "
        f"(label={clean_class['prediction_label']}, "
        f"method={clean_class.get('method', '?')})"
    )
    print(f"  Raw: {clean_class['raw_response']}")

    clean_keyword = "dog" in clean_desc.lower()
    clean_cat_keyword = "cat" in clean_desc.lower()

    # --------------------------------------------------------
    # STEP 3: GENERATE ADVERSARIAL IMAGE
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 3: GENERATING ADVERSARIAL IMAGE (ENSEMBLE PGD)")
    print("-" * 70)

    print(
        f"  Surrogates: {', '.join(m.name for m in surrogate_models)}"
    )
    print(f"  Epsilon: {EPSILON:.6f}")
    print(f"  Alpha: {ALPHA:.6f}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Source: {SOURCE_TEXT}")
    print(f"  Target: {TARGET_TEXT}")

    t0 = time.time()

    adversarial = vlm_ensemble_pgd(
        models=surrogate_models,
        image=image_tensor.to(surrogate_models[0].device),
        source_text=SOURCE_TEXT,
        target_text=TARGET_TEXT,
        epsilon=EPSILON,
        alpha=ALPHA,
        iterations=ITERATIONS,
        seed=SEED,
        random_start=False,
    )

    attack_time = time.time() - t0
    print(f"  Attack time: {attack_time:.2f}s")

    # --------------------------------------------------------
    # PERTURBATION METRICS
    # --------------------------------------------------------

    metrics = calculate_perturbation(
        image_tensor.to(adversarial.device),
        adversarial,
    )

    print(
        f"  L∞: {metrics['linf']:.8f}  "
        f"(budget: {EPSILON:.8f})  "
        f"within: {'YES' if metrics['linf'] <= EPSILON + 1e-6 else 'NO'}"
    )
    print(f"  L2: {metrics['l2']:.8f}")
    print(f"  Mean |δ|: {metrics['mean_abs']:.8f}")

    # --------------------------------------------------------
    # SAVE ADVERSARIAL IMAGE
    # --------------------------------------------------------

    adv_pil = tensor_to_pil(adversarial)

    os.makedirs(output_dir, exist_ok=True)
    adv_path = os.path.join(output_dir, f"adv_{filename}")
    adv_pil.save(adv_path)
    print(f"  Saved: {adv_path}")

    # --------------------------------------------------------
    # STEP 4: ADVERSARIAL EVALUATION ON SURROGATES
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 4: ADVERSARIAL EVALUATION ON SURROGATES (WHITE-BOX)")
    print("-" * 70)

    surrogate_success = 0

    for model in surrogate_models:
        result = model.predict(image=adv_pil, texts=texts)
        success = result["prediction"] == TARGET_TEXT
        if success:
            surrogate_success += 1

        print(
            f"  {model.name:<10} → "
            f"{result['prediction']}  "
            f"(source={result['scores'][0].item():.4f}, "
            f"target={result['scores'][1].item():.4f})  "
            f"{'✓ FOOLED' if success else '✗'}"
        )

    # --------------------------------------------------------
    # STEP 5: ADVERSARIAL EVALUATION ON OLLAMA (BLACK-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 5: ADVERSARIAL EVALUATION ON OLLAMA (BLACK-BOX)")
    print("-" * 70)

    adv_desc = ollama_adapter.describe_image(adv_pil)
    print(f"  Description: {adv_desc}")

    adv_class = ollama_adapter.classify_image(
        adv_pil, SOURCE_TEXT, TARGET_TEXT
    )
    print(
        f"  Classification: {adv_class['prediction']} "
        f"(label={adv_class['prediction_label']}, "
        f"method={adv_class.get('method', '?')})"
    )
    print(f"  Raw: {adv_class['raw_response']}")

    transfer_success = adv_class["target_selected"]

    # Also check description-based transfer
    adv_dog_keyword = "dog" in adv_desc.lower()
    adv_cat_keyword = "cat" in adv_desc.lower()
    desc_transfer = (
        adv_cat_keyword and not adv_dog_keyword
    )

    print()
    print(f"  A/B Transfer: {'YES' if transfer_success else 'NO'}")
    print(f"  Desc transfer (cat, no dog): {'YES' if desc_transfer else 'NO'}")
    print(f"  Clean mentions dog: {clean_keyword}, cat: {clean_cat_keyword}")
    print(f"  Adv mentions dog: {adv_dog_keyword}, cat: {adv_cat_keyword}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    return {
        "filename": filename,
        "clean_description": clean_desc,
        "clean_classification": clean_class["prediction"],
        "clean_label": clean_class["prediction_label"],
        "adversarial_description": adv_desc,
        "adversarial_classification": adv_class["prediction"],
        "adversarial_label": adv_class["prediction_label"],
        "surrogate_success": surrogate_success,
        "surrogate_total": len(surrogate_models),
        "transfer_success": transfer_success,
        "desc_transfer": desc_transfer,
        "clean_dog_kw": clean_keyword,
        "clean_cat_kw": clean_cat_keyword,
        "adv_dog_kw": adv_dog_keyword,
        "adv_cat_kw": adv_cat_keyword,
        "linf": metrics["linf"],
        "l2": metrics["l2"],
        "mean_abs": metrics["mean_abs"],
        "attack_time": attack_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TRANSFER ATTACK: WHITE-BOX ENSEMBLE → BLACK-BOX OLLAMA VLM")
    print("=" * 70)

    print()
    print("CONFIGURATION")
    print(f"  Surrogates     : {', '.join(SURROGATE_MODELS)}")
    print(f"  Target (black-box): Ollama {OLLAMA_MODEL} (1B params)")
    print(f"  Source text    : {SOURCE_TEXT}")
    print(f"  Target text    : {TARGET_TEXT}")
    print(f"  Epsilon        : {EPSILON:.6f}")
    print(f"  Alpha          : {ALPHA:.6f}")
    print(f"  Iterations     : {ITERATIONS}")
    print(f"  Ollama host    : {OLLAMA_HOST}")

    # ========================================================
    # LOAD SURROGATE MODELS
    # ========================================================

    print()
    print("-" * 70)
    print("LOADING SURROGATE MODELS")
    print("-" * 70)

    surrogates = []
    for name in SURROGATE_MODELS:
        print(f"  Loading {name}...")
        surrogates.append(get_vlm(name))

    # ========================================================
    # CONNECT TO OLLAMA
    # ========================================================

    print()
    print("-" * 70)
    print("CONNECTING TO OLLAMA")
    print("-" * 70)

    ollama = OllamaVLMAdapter(
        model_name=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        name="moondream",
        temperature=0.5,
        num_predict=200,
    )

    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Host: {OLLAMA_HOST}")

    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    image_paths = [IMAGE_PATH]

    dataset_paths = get_dataset_paths()
    if dataset_paths:
        image_paths.extend(dataset_paths)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)
    image_paths = unique_paths

    print()
    print(f"  Images to evaluate: {len(image_paths)}")

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for image_path in image_paths:
        try:
            result = evaluate_single_image(
                image_path=image_path,
                surrogate_models=surrogates,
                ollama_adapter=ollama,
                output_dir=OUTPUT_DIR,
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if not results:
        print()
        print("No images were successfully processed.")
        return

    print()
    print()
    print("=" * 70)
    print("FINAL TRANSFER RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'Image':<20}"
        f"{'Clean':<8}"
        f"{'Adv':<8}"
        f"{'Surr':<8}"
        f"{'A/B':<6}"
        f"{'Desc':<6}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['filename']:<20}"
            f"{r['clean_label']:<8}"
            f"{r['adversarial_label']:<8}"
            f"{r['surrogate_success']}/{r['surrogate_total']:<6}"
            f"{'YES' if r['transfer_success'] else 'NO':<6}"
            f"{'YES' if r['desc_transfer'] else 'NO':<6}"
        )

    # ========================================================
    # AGGREGATE
    # ========================================================

    total = len(results)
    transfer_successes = sum(1 for r in results if r["transfer_success"])
    desc_transfers = sum(1 for r in results if r["desc_transfer"])
    surrogate_total = sum(r["surrogate_success"] for r in results)
    surrogate_evals = sum(r["surrogate_total"] for r in results)

    # Count description changes (any change in output)
    desc_changed = sum(
        1
        for r in results
        if r["clean_description"].strip()
        and r["adversarial_description"].strip()
        and r["clean_description"].strip()
        != r["adversarial_description"].strip()
    )

    # Count cases where dog keyword disappeared
    dog_removed = sum(
        1
        for r in results
        if r["clean_dog_kw"] and not r["adv_dog_kw"]
    )

    avg_linf = np.mean([r["linf"] for r in results])
    avg_l2 = np.mean([r["l2"] for r in results])
    avg_time = np.mean([r["attack_time"] for r in results])

    print()
    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print()
    print(f"  Images evaluated          : {total}")
    print(f"  Surrogate success rate    : {surrogate_total}/{surrogate_evals} ({surrogate_total/surrogate_evals*100:.1f}%)")
    print(f"  A/B transfer rate         : {transfer_successes}/{total} ({transfer_successes/total*100:.1f}%)")
    print(f"  Desc transfer (cat,no dog): {desc_transfers}/{total} ({desc_transfers/total*100:.1f}%)")
    print(f"  Desc changed              : {desc_changed}/{total} ({desc_changed/total*100:.1f}%)")
    print(f"  Dog keyword removed       : {dog_removed}/{total} ({dog_removed/total*100:.1f}%)")
    print()
    print(f"  Average L∞                : {avg_linf:.8f}")
    print(f"  Average L2                : {avg_l2:.8f}")
    print(f"  Average attack time       : {avg_time:.2f}s")

    print()
    print("=" * 70)
    print("DESCRIPTIONS COMPARISON")
    print("=" * 70)

    for r in results:
        print()
        print(f"  {r['filename']}:")
        print(f"    Clean: {r['clean_description']}")
        print(f"    Adv:   {r['adversarial_description']}")

    print()
    print("=" * 70)
    print("Transfer evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
