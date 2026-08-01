import torch

from models.registry import get_model
from utils.image import load_image, save_tensor_as_image

from engine.attack_runner import run_attack
from evaluation.predict import predict
from evaluation.evaluator import evaluate_attack
from config.attacks import get_attack_config


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "resnet50"
IMAGE_PATH = "dog.jpg"

ATTACKS = [
    "fgsm",
    "bim",
    "pgd",
    "deepfool",
    "autoattack",
]


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

clean = predict(
    model=model_info.model,
    image=image,
    weights=model_info.weights,
    preprocess=model_info.preprocess,
)


# ============================================================
# EXPERIMENT HEADER
# ============================================================

print()
print("=" * 70)
print("ADVERSARIAL ROBUSTNESS EXPERIMENT")
print("=" * 70)

print()
print("Model:", model_info.name)
print("Image:", IMAGE_PATH)


# ============================================================
# CLEAN BASELINE
# ============================================================

print()
print("-" * 70)
print("CLEAN BASELINE")
print("-" * 70)

print()
print(
    "Prediction:",
    clean["category"]
)

print(
    "Confidence:",
    f"{clean['confidence'] * 100:.2f}%"
)


# ============================================================
# ATTACK LABEL
# ============================================================

label = torch.tensor(
    [clean["class_id"]],
    device=model_info.device,
)


# ============================================================
# ATTACK LOOP
# ============================================================

results = []


for attack_name in ATTACKS:

    # --------------------------------------------------------
    # Load attack-specific configuration
    # --------------------------------------------------------

    config = get_attack_config(
        attack_name
    )

    print()
    print("=" * 70)
    print(
        f"{attack_name.upper()} ATTACK"
    )
    print("=" * 70)

    print()
    print("Configuration:")

    for key, value in config.items():

        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "Generating adversarial example..."
    )

    # --------------------------------------------------------
    # Run attack
    # --------------------------------------------------------

    adversarial_image = run_attack(
        attack_name=attack_name,
        model=model_info.model,
        image=image,
        label=label,
        preprocess=model_info.preprocess,
        **config,
    )

    # --------------------------------------------------------
    # Save adversarial image
    # --------------------------------------------------------

    output_path = (
        f"outputs/dog_{attack_name}.png"
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

    # --------------------------------------------------------
    # Evaluate attack
    # --------------------------------------------------------

    evaluation = evaluate_attack(
        model=model_info.model,
        weights=model_info.weights,
        original_image=image,
        adversarial_image=adversarial_image,
        preprocess=model_info.preprocess,
    )

    results.append(
        (
            attack_name,
            evaluation,
        )
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()

    print(
        "Adversarial prediction:",
        evaluation["adversarial"]["category"],
    )

    print(
        "Adversarial confidence:",
        f"{evaluation['adversarial']['confidence'] * 100:.2f}%",
    )

    print()

    print(
        "Prediction changed:",
        "YES"
        if evaluation["prediction_changed"]
        else "NO",
    )

    print(
        "Confidence change:",
        f"{evaluation['confidence_change'] * 100:+.2f}"
        " percentage points",
    )

    print()

    print("Perturbation metrics:")

    print(
        "  L∞:",
        f"{evaluation['linf']:.8f}",
    )

    print(
        "  L2:",
        f"{evaluation['l2']:.8f}",
    )

    print(
        "  Mean |perturbation|:",
        f"{evaluation['mean_perturbation']:.8f}",
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print()

print(
    f"{'Attack':<12}"
    f"{'Changed':<12}"
    f"{'Prediction':<25}"
    f"{'Confidence':<12}"
    f"{'L∞':<12}"
)

print("-" * 70)


for attack_name, result in results:

    adversarial = result[
        "adversarial"
    ]

    print(
        f"{attack_name:<12}"
        f"{str(result['prediction_changed']):<12}"
        f"{adversarial['category']:<25}"
        f"{adversarial['confidence'] * 100:>7.2f}%   "
        f"{result['linf']:<12.6f}"
    )


# ============================================================
# OVERALL SUCCESS RATE
# ============================================================

successful = sum(
    result["prediction_changed"]
    for _, result in results
)

total = len(results)

success_rate = (
    successful / total * 100
    if total > 0
    else 0
)

print()

print(
    f"Attack success rate: "
    f"{success_rate:.2f}%"
)

print()
print("=" * 70)
print("Experiment complete.")
print("=" * 70)