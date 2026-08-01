import numpy as np
import torch

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from models.registry import get_model
from utils.image import load_image, save_tensor_as_image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "resnet50"
IMAGE_PATH = "dog.jpg"

EPSILON = 8 / 255
EPS_STEP = 2 / 255
MAX_ITER = 10


# ============================================================
# LOAD MODEL
# ============================================================

model_info = get_model(MODEL_NAME)

model = model_info.model
weights = model_info.weights
device = model_info.device
preprocess = model_info.preprocess

print("Using model:", MODEL_NAME)
print("Device:", device)


# ============================================================
# LOAD IMAGE
# ============================================================

_, image = load_image(IMAGE_PATH)

image = image.to(device)

print()
print("=" * 60)
print("ART PGD VALIDATION")
print("=" * 60)

print()
print("Model:", MODEL_NAME)
print("Image:", IMAGE_PATH)
print("Epsilon:", EPSILON)
print("Step size:", EPS_STEP)
print("Iterations:", MAX_ITER)


# ============================================================
# CLEAN PREDICTION
# ============================================================

with torch.no_grad():

    clean_output = model(
        preprocess(image)
    )

clean_probabilities = torch.softmax(
    clean_output,
    dim=1,
)

clean_confidence, clean_class = (
    clean_probabilities.max(dim=1)
)

clean_class = clean_class.item()
clean_confidence = clean_confidence.item()

clean_category = weights.meta[
    "categories"
][clean_class]


print()
print("Clean prediction:", clean_category)
print(
    "Clean confidence:",
    f"{clean_confidence * 100:.2f}%"
)


# ============================================================
# CREATE ART CLASSIFIER
# ============================================================
#
# ART expects NumPy arrays.
#
# Our model itself expects ImageNet-normalized tensors.
#
# ART's preprocessing parameter performs:
#
#     (x - mean) / std
#
# before passing the image to the model.
#
# ============================================================

loss = torch.nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=model,
    loss=loss,
    input_shape=(3, 224, 224),
    nb_classes=1000,
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    ),
    preprocessing=(
        np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        ),
        np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        ),
    ),
    device_type="gpu" if device.type == "cuda" else "cpu",
)


# ============================================================
# CONVERT IMAGE TO NUMPY
# ============================================================

image_numpy = (
    image.detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


label = np.array(
    [clean_class],
    dtype=np.int64,
)


# ============================================================
# CREATE ART PGD ATTACK
# ============================================================

attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=EPSILON,
    eps_step=EPS_STEP,
    max_iter=MAX_ITER,
    targeted=False,
    num_random_init=1,
)


# ============================================================
# GENERATE ADVERSARIAL IMAGE
# ============================================================

print()
print("Generating ART PGD adversarial example...")

adversarial_numpy = attack.generate(
    x=image_numpy,
    y=label,
)


# ============================================================
# CONVERT BACK TO TORCH
# ============================================================

adversarial_image = torch.from_numpy(
    adversarial_numpy
).to(device)


# ============================================================
# SAVE ADVERSARIAL IMAGE
# ============================================================

output_path = "outputs/art_pgd.png"

save_tensor_as_image(
    adversarial_image,
    output_path,
)

print()
print("Saved adversarial image:")
print(output_path)


# ============================================================
# ADVERSARIAL PREDICTION
# ============================================================

with torch.no_grad():

    adversarial_output = model(
        preprocess(adversarial_image)
    )

adversarial_probabilities = torch.softmax(
    adversarial_output,
    dim=1,
)

adversarial_confidence, adversarial_class = (
    adversarial_probabilities.max(dim=1)
)

adversarial_class = (
    adversarial_class.item()
)

adversarial_confidence = (
    adversarial_confidence.item()
)

adversarial_category = weights.meta[
    "categories"
][adversarial_class]


# ============================================================
# PERTURBATION METRICS
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

mean_abs = (
    perturbation.abs()
    .mean()
    .item()
)


# ============================================================
# RESULTS
# ============================================================

prediction_changed = (
    clean_class != adversarial_class
)

confidence_change = (
    adversarial_confidence
    - clean_confidence
)


print()
print("=" * 60)
print("ART PGD RESULTS")
print("=" * 60)

print()

print("Clean prediction:")
print(f"  {clean_category}")

print(
    "Clean confidence:",
    f"{clean_confidence * 100:.2f}%"
)

print()

print("Adversarial prediction:")
print(f"  {adversarial_category}")

print(
    "Adversarial confidence:",
    f"{adversarial_confidence * 100:.2f}%"
)

print()

print(
    "Prediction changed:",
    "YES" if prediction_changed else "NO",
)

print(
    "Confidence change:",
    f"{confidence_change * 100:+.2f}"
    " percentage points",
)

print()

print("Perturbation metrics:")

print(
    "  L∞:",
    f"{linf:.8f}",
)

print(
    "  L2:",
    f"{l2:.8f}",
)

print(
    "  Mean |perturbation|:",
    f"{mean_abs:.8f}",
)

print()

print(
    "Attack successful:",
    "YES" if prediction_changed else "NO",
)

print()
print("=" * 60)
print("Experiment complete.")
print("=" * 60)