import numpy as np
import torch

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch

from models.loader import load_model
from utils.image import load_image


# ============================================================
# Configuration
# ============================================================

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 10


# ============================================================
# Load model
# ============================================================

adapter = load_model()

print("Using device:", adapter.device)


# ============================================================
# Load image
# ============================================================

_, image = load_image("dog.jpg")

image = image.to(adapter.device)


# ============================================================
# Get clean prediction
# ============================================================

clean_input = adapter.preprocess(image)

with torch.no_grad():

    clean_output = adapter.model(
        clean_input
    )

clean_probabilities = torch.softmax(
    clean_output,
    dim=1
)

clean_confidence, clean_class_id = (
    clean_probabilities.max(dim=1)
)

clean_confidence = clean_confidence.item()
clean_class_id = clean_class_id.item()

clean_category = adapter.weights.meta["categories"][
    clean_class_id
]

print()
print("----- CLEAN IMAGE -----")
print("Prediction :", clean_category)
print(
    "Confidence :",
    f"{clean_confidence * 100:.2f}%"
)


# ============================================================
# Create ART classifier
# ============================================================

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    adapter.model.parameters(),
    lr=0.01
)

classifier = PyTorchClassifier(
    model=adapter.model,
    loss=loss,
    optimizer=optimizer,
    input_shape=tuple(image.shape[1:]),
    nb_classes=1000,
    clip_values=(0.0, 1.0),
    preprocessing=(
        np.array([0.485, 0.456, 0.406]),
        np.array([0.229, 0.224, 0.225]),
    ),
    device_type="gpu" if adapter.device.type == "cuda" else "cpu",
)


# ============================================================
# Prepare input for ART
# ============================================================

image_numpy = (
    image.detach()
    .cpu()
    .numpy()
)


label_numpy = np.array(
    [clean_class_id]
)


# ============================================================
# ART PGD
# ============================================================

attack = ProjectedGradientDescentPyTorch(
    estimator=classifier,
    norm=np.inf,
    eps=EPSILON,
    eps_step=ALPHA,
    max_iter=ITERATIONS,
    targeted=False,
    num_random_init=1,
    batch_size=1,
    verbose=False,
)


print()
print("Running ART PGD...")


art_adversarial = attack.generate(
    x=image_numpy,
    y=label_numpy,
)


# ============================================================
# Convert ART result back to torch
# ============================================================

art_adversarial_tensor = torch.from_numpy(
    art_adversarial
).to(adapter.device)


# ============================================================
# Evaluate ART adversarial image
# ============================================================

art_input = adapter.preprocess(
    art_adversarial_tensor
)

with torch.no_grad():

    art_output = adapter.model(
        art_input
    )

art_probabilities = torch.softmax(
    art_output,
    dim=1
)

art_confidence, art_class_id = (
    art_probabilities.max(dim=1)
)

art_confidence = art_confidence.item()
art_class_id = art_class_id.item()

art_category = adapter.weights.meta["categories"][
    art_class_id
]


# ============================================================
# Perturbation
# ============================================================

perturbation = (
    art_adversarial_tensor
    - image
)

linf = (
    perturbation
    .abs()
    .max()
    .item()
)

l2 = (
    torch.linalg.vector_norm(
        perturbation.reshape(
            perturbation.shape[0],
            -1
        ),
        ord=2,
        dim=1,
    )
    .item()
)

mean_abs = (
    perturbation
    .abs()
    .mean()
    .item()
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("ART PGD RESULTS")
print("=" * 60)

print()
print("Clean prediction:")
print(" ", clean_category)

print(
    "Clean confidence:",
    f"{clean_confidence * 100:.2f}%"
)

print()
print("Adversarial prediction:")
print(" ", art_category)

print(
    "Adversarial confidence:",
    f"{art_confidence * 100:.2f}%"
)

print()
print(
    "Prediction changed:",
    "YES" if art_class_id != clean_class_id else "NO"
)

print()
print("Perturbation metrics:")
print(f"  L∞: {linf:.8f}")
print(f"  L2: {l2:.8f}")
print(f"  Mean |perturbation|: {mean_abs:.8f}")

print("=" * 60)
