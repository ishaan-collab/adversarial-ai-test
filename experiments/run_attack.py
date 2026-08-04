import torch

from models.loader import load_model

from utils.image import (
    load_image,
    save_tensor_as_image,
)

from attacks.fgsm import fgsm_attack
from attacks.bim import bim_attack
from attacks.pgd import pgd_attack

from evaluation.evaluator import (
    evaluate_attack,
    print_evaluation,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = "dog.jpg"

EPSILON = 8 / 255

BIM_ALPHA = 2 / 255
BIM_ITERATIONS = 10


# ============================================================
# Load model
# ============================================================

adapter = load_model()

print("Using device:", adapter.device)


# ============================================================
# Load image
# ============================================================

_, image = load_image(
    IMAGE_PATH
)

image = image.to(adapter.device)


# ============================================================
# Clean prediction
# ============================================================

clean_input = adapter.preprocess(
    image
)

with torch.no_grad():
    clean_output = adapter.model(
        clean_input
    )

clean_probabilities = torch.softmax(
    clean_output,
    dim=1,
)

clean_confidence, clean_class_id = (
    clean_probabilities.max(dim=1)
)

clean_class_id = (
    clean_class_id.item()
)

clean_confidence = (
    clean_confidence.item()
)

clean_category = adapter.weights.meta[
    "categories"
][clean_class_id]


print()
print("----- CLEAN IMAGE -----")
print(
    "Prediction :",
    clean_category,
)

print(
    "Confidence :",
    f"{clean_confidence * 100:.2f}%",
)


# ============================================================
# Attack label
# ============================================================

label = torch.tensor(
    [clean_class_id],
    device=adapter.device,
)


# ============================================================
# FGSM
# ============================================================

fgsm_image = fgsm_attack(
    model=adapter.model,
    image=image,
    label=label,
    epsilon=EPSILON,
    preprocess=adapter.preprocess,
)

save_tensor_as_image(
    fgsm_image,
    "outputs/dog_fgsm.png",
)

fgsm_results = evaluate_attack(
    model=adapter.model,
    weights=adapter.weights,
    original_image=image,
    adversarial_image=fgsm_image,
    preprocess=adapter.preprocess,
)

print_evaluation(
    fgsm_results,
    "FGSM",
)


# ============================================================
# BIM
# ============================================================

bim_image = bim_attack(
    model=adapter.model,
    image=image,
    label=label,
    epsilon=EPSILON,
    alpha=BIM_ALPHA,
    iterations=BIM_ITERATIONS,
    preprocess=adapter.preprocess,
)

save_tensor_as_image(
    bim_image,
    "outputs/dog_bim.png",
)

bim_results = evaluate_attack(
    model=adapter.model,
    weights=adapter.weights,
    original_image=image,
    adversarial_image=bim_image,
    preprocess=adapter.preprocess,
)

print_evaluation(
    bim_results,
    "BIM",
)

# ============================================================
# PGD
# ============================================================

PGD_ALPHA = 2 / 255
PGD_ITERATIONS = 10

pgd_image = pgd_attack(
    model=adapter.model,
    image=image,
    label=label,
    epsilon=EPSILON,
    alpha=PGD_ALPHA,
    iterations=PGD_ITERATIONS,
    preprocess=adapter.preprocess,
    random_start=True,
)

save_tensor_as_image(
    pgd_image,
    "outputs/dog_pgd.png",
)

pgd_results = evaluate_attack(
    model=adapter.model,
    weights=adapter.weights,
    original_image=image,
    adversarial_image=pgd_image,
    preprocess=adapter.preprocess,
)

print_evaluation(
    pgd_results,
    "PGD",
)

# ============================================================
# Experiment Summary
# ============================================================

print()
print("=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print()

print(
    f"{'Attack':<10}"
    f"{'Prediction Changed':<22}"
    f"{'Confidence':<15}"
    f"{'L∞':<15}"
)

print("-" * 60)

print(
    f"{'FGSM':<10}"
    f"{str(fgsm_results['prediction_changed']):<22}"
    f"{fgsm_results['adversarial']['confidence'] * 100:>6.2f}%"
    f"{fgsm_results['linf']:<15.8f}"
)

print(
    f"{'BIM':<10}"
    f"{str(bim_results['prediction_changed']):<22}"
    f"{bim_results['adversarial']['confidence'] * 100:>6.2f}%"
    f"{bim_results['linf']:<15.8f}"
)

print(
    f"{'PGD':<10}"
    f"{str(pgd_results['prediction_changed']):<22}"
    f"{pgd_results['adversarial']['confidence'] * 100:>6.2f}%"
    f"{pgd_results['linf']:<15.8f}"
)

print()
print("Experiment complete.")
