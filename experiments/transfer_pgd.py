import torch

from models.registry import get_model
from engine.attack_runner import run_attack
from evaluation.evaluator import evaluate_attack
from utils.image import load_image

from utils.image import (
    load_image,
    save_tensor_as_image,
)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_MODEL = "resnet50"

TARGET_MODELS = [
    "resnet50",
    "vit_b_16",
    "convnext_tiny",
]

ATTACK = "pgd"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 10

IMAGE_PATH = "dog.jpg"


# ============================================================
# LOAD SOURCE MODEL
# ============================================================

source = get_model(
    SOURCE_MODEL
)

print()
print("=" * 70)
print("TRANSFERABILITY EXPERIMENT")
print("=" * 70)

print()
print("Source model :", source.name)
print("Attack       :", ATTACK)
print("Epsilon      :", EPSILON)
print("Alpha        :", ALPHA)
print("Iterations   :", ITERATIONS)


# ============================================================
# LOAD IMAGE
# ============================================================

_, image = load_image(
    IMAGE_PATH
)

image = image.to(
    source.device
)


# ============================================================
# SOURCE CLEAN PREDICTION
# ============================================================

with torch.no_grad():

    output = source.predict(
        image
    )

probabilities = torch.softmax(
    output,
    dim=1
)

clean_confidence, clean_class_id = (
    probabilities.max(dim=1)
)

clean_class_id = (
    clean_class_id.item()
)

clean_confidence = (
    clean_confidence.item()
)

clean_category = (
    source.weights.meta["categories"]
    [clean_class_id]
)


print()
print("----- CLEAN SOURCE PREDICTION -----")
print("Prediction :", clean_category)
print(
    "Confidence :",
    f"{clean_confidence * 100:.2f}%"
)


# ============================================================
# GENERATE ADVERSARIAL EXAMPLE
# ============================================================

label = torch.tensor(
    [clean_class_id],
    device=source.device
)

print()
print("Generating PGD adversarial example...")

adversarial_image = run_attack(
    attack_name=ATTACK,
    model_adapter=source,
    image=image,
    label=label,
    epsilon=EPSILON,
    alpha=ALPHA,
    iterations=ITERATIONS,
)

save_tensor_as_image(
    adversarial_image,
    "outputs/transfer_pgd.png",
)

print()
print("Saved adversarial image:")
print("outputs/transfer_pgd.png")

# ============================================================
# PERTURBATION
# ============================================================

perturbation = (
    adversarial_image - image
)

linf = (
    perturbation.abs()
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


print()
print("----- PERTURBATION -----")
print("L∞ :", f"{linf:.8f}")
print("L2 :", f"{l2:.8f}")


# ============================================================
# EVALUATE AGAINST TARGET MODELS
# ============================================================

print()
print("=" * 70)
print("TRANSFER RESULTS")
print("=" * 70)


results = []


for target_name in TARGET_MODELS:

    print()
    print("-" * 70)
    print(
        f"TARGET MODEL: {target_name}"
    )
    print("-" * 70)

    target = get_model(
        target_name
    )

    # --------------------------------------------------------
    # Move image to target device
    # --------------------------------------------------------

    target_image = image.to(
        target.device
    )

    target_adversarial = (
        adversarial_image.to(
            target.device
        )
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_attack(
        model=target.model,
        weights=target.weights,
        original_image=target_image,
        adversarial_image=target_adversarial,
        preprocess=target.preprocess,
    )

    results.append(
        (
            target_name,
            evaluation,
        )
    )

    print(
        "Clean prediction :",
        evaluation["clean"]["category"],
    )

    print(
        "Clean confidence :",
        f"{evaluation['clean']['confidence'] * 100:.2f}%",
    )

    print()

    print(
        "Adversarial prediction :",
        evaluation["adversarial"]["category"],
    )

    print(
        "Adversarial confidence :",
        f"{evaluation['adversarial']['confidence'] * 100:.2f}%",
    )

    print()

    print(
        "Changed :",
        "YES"
        if evaluation["prediction_changed"]
        else "NO",
    )

    print(
        "L∞ :",
        f"{evaluation['linf']:.8f}",
    )

    print(
        "L2 :",
        f"{evaluation['l2']:.8f}",
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRANSFERABILITY SUMMARY")
print("=" * 70)

print()

print(
    f"{'Model':20s}"
    f"{'Changed':12s}"
    f"{'Clean':12s}"
    f"{'Adversarial':15s}"
)

print("-" * 70)


for target_name, evaluation in results:

    changed = (
        "YES"
        if evaluation["prediction_changed"]
        else "NO"
    )

    clean_conf = (
        evaluation["clean"]["confidence"]
        * 100
    )

    adv_conf = (
        evaluation["adversarial"]["confidence"]
        * 100
    )

    print(
        f"{target_name:20s}"
        f"{changed:12s}"
        f"{clean_conf:8.2f}%    "
        f"{adv_conf:8.2f}%"
    )


print()
print("=" * 70)
print("Experiment complete.")
print("=" * 70)