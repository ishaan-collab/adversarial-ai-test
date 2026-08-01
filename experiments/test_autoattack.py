import torch

from models.registry import get_model
from attacks.autoattack import autoattack_attack
from evaluation.evaluator import evaluate_attack
from utils.image import load_image, save_tensor_as_image
from config.attacks import get_attack_config


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "swin_b"
IMAGE_PATH = "dog.jpg"


# ============================================================
# LOAD MODEL
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
# LOAD IMAGE
# ============================================================

_, image = load_image(
    IMAGE_PATH
)

image = image.to(
    model_info.device
)


# ============================================================
# CLEAN PREDICTION
# ============================================================

with torch.no_grad():

    output = model_info.predict(
        image
    )

probabilities = torch.softmax(
    output,
    dim=1
)

clean_confidence, clean_class = (
    probabilities.max(dim=1)
)

clean_class = clean_class.item()

label = torch.tensor(
    [clean_class],
    device=image.device,
)


# ============================================================
# CONFIGURATION
# ============================================================

config = get_attack_config(
    "autoattack"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("AUTOATTACK VALIDATION")
print("=" * 70)

print()
print("Model:", model_info.name)
print("Image:", IMAGE_PATH)

print()
print("Configuration:")

for key, value in config.items():

    print(
        f"  {key}: {value}"
    )

print()
print("Clean prediction:")

print(
    " ",
    model_info.weights.meta["categories"][
        clean_class
    ]
)

print(
    "Clean confidence:",
    f"{clean_confidence.item() * 100:.2f}%"
)


# ============================================================
# GENERATE ADVERSARIAL EXAMPLE
# ============================================================

print()
print(
    "Generating AutoAttack adversarial example..."
)

adversarial_image = autoattack_attack(
    model=model_info.model,
    image=image,
    label=label,
    preprocess=model_info.preprocess,
    **config,
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    "outputs/dog_autoattack.png"
)

save_tensor_as_image(
    adversarial_image,
    output_path,
)

print()
print(
    "Saved:",
    output_path
)


# ============================================================
# EVALUATE
# ============================================================

results = evaluate_attack(
    model=model_info.model,
    weights=model_info.weights,
    original_image=image,
    adversarial_image=adversarial_image,
    preprocess=model_info.preprocess,
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("AUTOATTACK RESULTS")
print("=" * 70)

print()

print(
    "Clean prediction:"
)

print(
    " ",
    results["clean"]["category"]
)

print(
    "Clean confidence:",
    f"{results['clean']['confidence'] * 100:.2f}%"
)

print()

print(
    "Adversarial prediction:"
)

print(
    " ",
    results["adversarial"]["category"]
)

print(
    "Adversarial confidence:",
    f"{results['adversarial']['confidence'] * 100:.2f}%"
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

print()

print(
    "Attack successful:",
    "YES"
    if results["prediction_changed"]
    else "NO"
)

print()
print("=" * 70)
print("Experiment complete.")
print("=" * 70)
