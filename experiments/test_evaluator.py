from models.loader import load_model
from utils.image import load_image
from evaluation.evaluator import evaluate_attack
from engine.attack_runner import run_attack


EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 10


# --------------------------------------------------
# Load model
# --------------------------------------------------

model, weights, device = load_model()

print("Using device:", device)


# --------------------------------------------------
# Load image
# --------------------------------------------------

_, image = load_image(
    "dog.jpg"
)

image = image.to(device)


# --------------------------------------------------
# Clean prediction
# --------------------------------------------------

clean = evaluate_attack(
    model=model,
    weights=weights,
    original_image=image,
    adversarial_image=image,
    preprocess=lambda x: (
        (x - x.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1))
        /
        x.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)
    ),
)

label = clean["clean"]["class_id"]


# --------------------------------------------------
# Generate PGD adversarial example
# --------------------------------------------------

adversarial_image = run_attack(
    attack_name="pgd",
    model=model,
    image=image,
    label=__import__("torch").tensor(
        [label],
        device=device,
    ),
    preprocess=lambda x: (
        (x - x.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1))
        /
        x.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)
    ),
    epsilon=EPSILON,
    alpha=ALPHA,
    iterations=ITERATIONS,
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

result = evaluate_attack(
    model=model,
    weights=weights,
    original_image=image,
    adversarial_image=adversarial_image,
    preprocess=lambda x: (
        (x - x.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1))
        /
        x.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)
    ),
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
    f"{result['mean_abs']:.8f}"
)

print("=" * 60)
