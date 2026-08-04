from models.loader import load_model
from utils.image import load_image
from evaluation.evaluator import evaluate_attack
from engine.attack_runner import run_attack

import torch


EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 10


# --------------------------------------------------
# Load model
# --------------------------------------------------

adapter = load_model()

print("Using device:", adapter.device)


# --------------------------------------------------
# Load image
# --------------------------------------------------

_, image = load_image(
    "dog.jpg"
)

image = image.to(adapter.device)


# --------------------------------------------------
# Clean prediction
# --------------------------------------------------

clean = evaluate_attack(
    model=adapter.model,
    weights=adapter.weights,
    original_image=image,
    adversarial_image=image,
    preprocess=adapter.preprocess,
)

label = clean["clean"]["class_id"]


# --------------------------------------------------
# Generate PGD adversarial example
# --------------------------------------------------

adversarial_image = run_attack(
    attack_name="pgd",
    model=adapter.model,
    image=image,
    label=torch.tensor(
        [label],
        device=adapter.device,
    ),
    preprocess=adapter.preprocess,
    epsilon=EPSILON,
    alpha=ALPHA,
    iterations=ITERATIONS,
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

result = evaluate_attack(
    model=adapter.model,
    weights=adapter.weights,
    original_image=image,
    adversarial_image=adversarial_image,
    preprocess=adapter.preprocess,
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print()
print("=" * 60)
print("EVALUATION RESULT")
print("=" * 60)

print()
print("Clean:")
print(
    f"  {result['clean']['category']}"
)
print(
    f"  Confidence: "
    f"{result['clean']['confidence'] * 100:.2f}%"
)

print()
print("Adversarial:")
print(
    f"  {result['adversarial']['category']}"
)
print(
    f"  Confidence: "
    f"{result['adversarial']['confidence'] * 100:.2f}%"
)

print()
print(
    "Prediction changed:",
    "YES"
    if result["prediction_changed"]
    else "NO",
)

print(
    "Confidence change:",
    f"{result['confidence_change'] * 100:+.2f}"
    " percentage points"
)

print()
print("Perturbation:")
print(
    f"  L∞: "
    f"{result['linf']:.8f}"
)
print(
    f"  L2: "
    f"{result['l2']:.8f}"
)
print(
    f"  Mean |perturbation|: "
    f"{result['mean_perturbation']:.8f}"
)

print("=" * 60)
