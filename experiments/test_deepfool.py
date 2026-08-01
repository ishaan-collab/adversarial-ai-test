import os
import torch

from models.registry import get_model
from attacks.deepfool import deepfool_attack
from evaluation.evaluator import evaluate_attack
from utils.image import (
    load_image,
    save_tensor_as_image,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = "dog.jpg"
MODEL_NAME = "resnet50"

MAX_ITERATIONS = 20
OVERSHOOT = 0.02

OUTPUT_PATH = "outputs/dog_deepfool.png"


# ============================================================
# Load model
# ============================================================

model_info = get_model(
    MODEL_NAME
)

print(
    "Using model:",
    model_info.name
)

print(
    "Device:",
    model_info.device
)


# ============================================================
# Load image
# ============================================================

_, image = load_image(
    IMAGE_PATH
)

image = image.to(
    model_info.device
)


# ============================================================
# Clean prediction
# ============================================================

with torch.no_grad():

    clean_output = model_info.predict(
        image
    )

clean_probabilities = torch.softmax(
    clean_output,
    dim=1
)

clean_confidence, clean_class = (
    clean_probabilities.max(
        dim=1
    )
)

clean_class = clean_class.item()
clean_confidence = clean_confidence.item()

clean_category = (
    model_info.weights.meta[
        "categories"
    ][clean_class]
)


# ============================================================
# Print experiment information
# ============================================================

print()
print("=" * 60)
print("DEEPFOOL ATTACK EXPERIMENT")
print("=" * 60)

print()

print(
    "Model:",
    model_info.name
)

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Max iterations:",
    MAX_ITERATIONS
)

print(
    "Overshoot:",
    OVERSHOOT
)

print()

print(
    "Clean prediction:",
    clean_category
)

print(
    "Clean confidence:",
    f"{clean_confidence * 100:.2f}%"
)


# ============================================================
# Prepare label
# ============================================================

label = torch.tensor(
    [clean_class],
    device=image.device
)


# ============================================================
# Generate DeepFool adversarial example
# ============================================================

print()
print(
    "Generating DeepFool adversarial example..."
)

adversarial_image = deepfool_attack(
    model=model_info.model,
    image=image,
    label=label,
    max_iterations=MAX_ITERATIONS,
    overshoot=OVERSHOOT,
    preprocess=model_info.preprocess,
)


# ============================================================
# Save adversarial image
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

save_tensor_as_image(
    adversarial_image,
    OUTPUT_PATH
)

print()
print(
    "Saved adversarial image:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# Evaluate attack
# ============================================================

results = evaluate_attack(
    model=model_info.model,
    weights=model_info.weights,
    original_image=image,
    adversarial_image=adversarial_image,
    preprocess=model_info.preprocess,
)


# ============================================================
# Extract results
# ============================================================

clean_result = results[
    "clean"
]

adversarial_result = results[
    "adversarial"
]


# ============================================================
# Print results
# ============================================================

print()
print("=" * 60)
print("DEEPFOOL RESULTS")
print("=" * 60)

print()

print(
    "Clean prediction:"
)

print(
    f"  {clean_result['category']}"
)

print(
    "Clean confidence:",
    f"{clean_result['confidence'] * 100:.2f}%"
)

print()

print(
    "Adversarial prediction:"
)

print(
    f"  {adversarial_result['category']}"
)

print(
    "Adversarial confidence:",
    f"{adversarial_result['confidence'] * 100:.2f}%"
)

print()

print(
    "Prediction changed:",
    "YES"
    if results["prediction_changed"]
    else "NO"
)

print(
    "Confidence change:",
    f"{results['confidence_change'] * 100:+.2f}"
    " percentage points"
)


# ============================================================
# Perturbation metrics
# ============================================================

print()

print(
    "Perturbation metrics:"
)

print(
    "  L∞:",
    f"{results['linf']:.8f}"
)

print(
    "  L2:",
    f"{results['l2']:.8f}"
)

print(
    "  Mean |perturbation|:",
    f"{results['mean_perturbation']:.8f}"
)


# ============================================================
# Final attack status
# ============================================================

print()

print(
    "Attack successful:",
    "YES"
    if results["prediction_changed"]
    else "NO"
)

print()

print("=" * 60)
print("Experiment complete.")
print("=" * 60)