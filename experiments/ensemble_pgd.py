import torch

from models.registry import get_model
from attacks.ensemble_pgd import ensemble_pgd_attack
from evaluation.evaluator import evaluate_attack
from utils.image import (
    load_image,
    save_tensor_as_image,
)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_MODEL_NAMES = [
    "resnet50",
    "vit_b_16",
    "convnext_tiny",
]

TARGET_MODEL_NAMES = [
    "resnet101",
    "vit_l_16",
    "convnext_base",
    "swin_b",
]

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 10

IMAGE_PATH = "dog.jpg"

OUTPUT_PATH = (
    "outputs/ensemble_pgd_transfer.png"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("ENSEMBLE PGD → LARGE MODEL TRANSFER EXPERIMENT")
print("=" * 75)


# ============================================================
# LOAD SOURCE MODELS
# ============================================================

print()
print("SOURCE MODELS")
print("-" * 75)

source_models = []

for name in SOURCE_MODEL_NAMES:

    model = get_model(name)

    source_models.append(model)

    print(
        f"  {model.name}"
    )


# ============================================================
# LOAD TARGET MODELS
# ============================================================

print()
print("UNSEEN TARGET MODELS")
print("-" * 75)

target_models = []

for name in TARGET_MODEL_NAMES:

    model = get_model(name)

    target_models.append(model)

    print(
        f"  {model.name}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

_, image = load_image(
    IMAGE_PATH
)

image = image.to(
    source_models[0].device
)


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("CLEAN MODEL AGREEMENT")
print("=" * 75)

clean_results = []

for model in source_models + target_models:

    evaluation = evaluate_attack(
        model=model.model,
        weights=model.weights,
        original_image=image,
        adversarial_image=image,
        preprocess=model.preprocess,
    )

    clean_results.append(
        (
            model.name,
            evaluation["clean"],
        )
    )

    print()
    print(
        f"{model.name:<20}"
        f"{evaluation['clean']['category']:<30}"
        f"{evaluation['clean']['confidence'] * 100:>8.2f}%"
    )


# ============================================================
# DETERMINE COMMON LABEL
# ============================================================

source_labels = [
    result["class_id"]
    for _, result in clean_results
    if _ in SOURCE_MODEL_NAMES
]

if len(set(source_labels)) != 1:

    raise RuntimeError(
        "Source models do not agree on the clean label. "
        "Ensemble attack requires a common source label."
    )

clean_class_id = source_labels[0]

source_reference = source_models[0]

clean_category = (
    source_reference.weights.meta[
        "categories"
    ][clean_class_id]
)


print()
print(
    "Common source label:",
    clean_category
)


# ============================================================
# PREPARE ENSEMBLE
# ============================================================

model_objects = [
    model.model
    for model in source_models
]

preprocesses = [
    model.preprocess
    for model in source_models
]


label = torch.tensor(
    [clean_class_id],
    device=image.device,
)


# ============================================================
# GENERATE ENSEMBLE PERTURBATION
# ============================================================

print()
print("=" * 75)
print("GENERATING ENSEMBLE PGD")
print("=" * 75)

print()
print("Epsilon    :", EPSILON)
print("Alpha      :", ALPHA)
print("Iterations :", ITERATIONS)

print()
print("Optimizing against:")

for model in source_models:

    print(
        "  -",
        model.name
    )

print()
print(
    "Generating adversarial example..."
)


adversarial_image = ensemble_pgd_attack(
    models=model_objects,
    image=image,
    label=label,
    epsilon=EPSILON,
    alpha=ALPHA,
    iterations=ITERATIONS,
    preprocesses=preprocesses,
    random_start=True,
)


# ============================================================
# SAVE ADVERSARIAL IMAGE
# ============================================================

save_tensor_as_image(
    adversarial_image,
    OUTPUT_PATH,
)

print()
print(
    "Saved adversarial image:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# PERTURBATION METRICS
# ============================================================

perturbation = (
    adversarial_image
    - image
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

mean_abs = (
    perturbation.abs()
    .mean()
    .item()
)


print()
print("=" * 75)
print("PERTURBATION")
print("=" * 75)

print()
print(
    "L∞                  :",
    f"{linf:.8f}"
)

print(
    "L2                  :",
    f"{l2:.8f}"
)

print(
    "Mean |perturbation| :",
    f"{mean_abs:.8f}"
)


# ============================================================
# EVALUATE SOURCE MODELS
# ============================================================

print()
print("=" * 75)
print("SOURCE MODEL RESULTS")
print("=" * 75)

source_results = []

for model in source_models:

    evaluation = evaluate_attack(
        model=model.model,
        weights=model.weights,
        original_image=image,
        adversarial_image=adversarial_image,
        preprocess=model.preprocess,
    )

    source_results.append(
        (
            model.name,
            evaluation,
        )
    )

    print()
    print("-" * 75)

    print(
        "MODEL:",
        model.name
    )

    print(
        "Clean:",
        evaluation["clean"]["category"],
        f"({evaluation['clean']['confidence'] * 100:.2f}%)"
    )

    print(
        "Adversarial:",
        evaluation["adversarial"]["category"],
        f"({evaluation['adversarial']['confidence'] * 100:.2f}%)"
    )

    print(
        "Changed:",
        "YES"
        if evaluation["prediction_changed"]
        else "NO"
    )


# ============================================================
# EVALUATE UNSEEN TARGET MODELS
# ============================================================

print()
print("=" * 75)
print("UNSEEN LARGE MODEL TRANSFER")
print("=" * 75)

target_results = []

for model in target_models:

    evaluation = evaluate_attack(
        model=model.model,
        weights=model.weights,
        original_image=image,
        adversarial_image=adversarial_image,
        preprocess=model.preprocess,
    )

    target_results.append(
        (
            model.name,
            evaluation,
        )
    )

    print()
    print("-" * 75)

    print(
        "TARGET MODEL:",
        model.name
    )

    print(
        "Clean prediction:",
        evaluation["clean"]["category"]
    )

    print(
        "Clean confidence:",
        f"{evaluation['clean']['confidence'] * 100:.2f}%"
    )

    print()

    print(
        "Adversarial prediction:",
        evaluation["adversarial"]["category"]
    )

    print(
        "Adversarial confidence:",
        f"{evaluation['adversarial']['confidence'] * 100:.2f}%"
    )

    print()

    print(
        "Prediction changed:",
        "YES"
        if evaluation["prediction_changed"]
        else "NO"
    )

    print(
        "Confidence change:",
        f"{evaluation['confidence_change'] * 100:+.2f}"
        " percentage points"
    )


# ============================================================
# SOURCE SUCCESS
# ============================================================

source_successes = sum(
    evaluation["prediction_changed"]
    for _, evaluation in source_results
)

source_total = len(source_results)

source_success_rate = (
    source_successes / source_total
    if source_total > 0
    else 0.0
)


# ============================================================
# UNSEEN TARGET SUCCESS
# ============================================================

target_successes = sum(
    evaluation["prediction_changed"]
    for _, evaluation in target_results
)

target_total = len(target_results)

target_success_rate = (
    target_successes / target_total
    if target_total > 0
    else 0.0
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("TRANSFER SUMMARY")
print("=" * 75)

print()

print(
    f"{'Model':20s}"
    f"{'Role':12s}"
    f"{'Changed':12s}"
    f"{'Clean':12s}"
    f"{'Adversarial':15s}"
)

print("-" * 75)


for name, evaluation in source_results:

    changed = (
        "YES"
        if evaluation["prediction_changed"]
        else "NO"
    )

    print(
        f"{name:20s}"
        f"{'SOURCE':12s}"
        f"{changed:12s}"
        f"{evaluation['clean']['confidence'] * 100:8.2f}%    "
        f"{evaluation['adversarial']['confidence'] * 100:8.2f}%"
    )


for name, evaluation in target_results:

    changed = (
        "YES"
        if evaluation["prediction_changed"]
        else "NO"
    )

    print(
        f"{name:20s}"
        f"{'UNSEEN':12s}"
        f"{changed:12s}"
        f"{evaluation['clean']['confidence'] * 100:8.2f}%    "
        f"{evaluation['adversarial']['confidence'] * 100:8.2f}%"
    )


print()
print(
    "Source-model success rate:",
    f"{source_success_rate * 100:.2f}%"
)

print(
    "Unseen-model transfer rate:",
    f"{target_success_rate * 100:.2f}%"
)

print()

print(
    "Perturbation budget:",
    f"{EPSILON:.8f}"
)

print(
    "Actual L∞:",
    f"{linf:.8f}"
)

print()

print("=" * 75)
print("Experiment complete.")
print("=" * 75)