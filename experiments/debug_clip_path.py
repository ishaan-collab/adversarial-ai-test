import numpy as np
import torch
from PIL import Image

from models.vlm_registry import get_vlm
from attacks.vlm_pgd import (
    _get_clip_image_features,
    _get_clip_image_preprocessed,
    _get_clip_text_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"

TEXTS = [
    "a photo of a golden retriever",
    "a photo of a Norfolk terrier",
]


# ============================================================
# LOAD MODEL AND IMAGE
# ============================================================

model = get_vlm("clip")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


print()
print("=" * 70)
print("CLIP FORWARD-PATH DIAGNOSTIC")
print("=" * 70)


# ============================================================
# OFFICIAL HUGGING FACE PATH
# ============================================================

official_inputs = model.processor(
    text=TEXTS,
    images=image,
    return_tensors="pt",
    padding=True,
)

official_inputs = {
    key: value.to(model.device)
    for key, value in official_inputs.items()
    if torch.is_tensor(value)
}


with torch.no_grad():

    official_outputs = model.model(
        **official_inputs
    )

official_logits = (
    official_outputs
    .logits_per_image[0]
    .detach()
)


print()
print("OFFICIAL HUGGING FACE LOGITS")

for text, score in zip(
    TEXTS,
    official_logits,
):

    print(
        f"  {score.item():.6f}  {text}"
    )


official_prediction = TEXTS[
    official_logits.argmax().item()
]

print()
print(
    "Official prediction:",
    official_prediction,
)


# ============================================================
# CONVERT ORIGINAL IMAGE TO [0,1]
# ============================================================

image_tensor = (
    torch.from_numpy(
        np.array(image)
    )
    .permute(2, 0, 1)
    .float()
    / 255.0
)

image_tensor = (
    image_tensor
    .unsqueeze(0)
    .to(model.device)
)


print()
print("ORIGINAL IMAGE TENSOR")

print(
    "Shape:",
    tuple(image_tensor.shape),
)

print(
    "Min:",
    image_tensor.min().item(),
)

print(
    "Max:",
    image_tensor.max().item(),
)


# ============================================================
# OUR DIFFERENTIABLE PREPROCESSING
# ============================================================

our_pixel_values = (
    _get_clip_image_preprocessed(
        model,
        image_tensor,
    )
)


print()
print("OUR PREPROCESSED IMAGE")

print(
    "Shape:",
    tuple(our_pixel_values.shape),
)

print(
    "Min:",
    our_pixel_values.min().item(),
)

print(
    "Max:",
    our_pixel_values.max().item(),
)


# ============================================================
# COMPARE PREPROCESSING DIRECTLY
# ============================================================

official_pixel_values = (
    official_inputs[
        "pixel_values"
    ]
)

preprocessing_difference = (
    our_pixel_values
    - official_pixel_values
)

print()
print("=" * 70)
print("PREPROCESSING COMPARISON")
print("=" * 70)

print()

print(
    "Official shape:",
    tuple(
        official_pixel_values.shape
    ),
)

print(
    "Our shape:",
    tuple(
        our_pixel_values.shape
    ),
)

print()

print(
    "Max absolute difference:",
    preprocessing_difference
    .abs()
    .max()
    .item(),
)

print(
    "Mean absolute difference:",
    preprocessing_difference
    .abs()
    .mean()
    .item(),
)

print(
    "L2 difference:",
    torch.norm(
        preprocessing_difference
    ).item(),
)


# ============================================================
# OUR IMAGE FEATURES
# ============================================================

with torch.no_grad():

    image_features = (
        _get_clip_image_features(
            model,
            our_pixel_values,
        )
    )


print()
print("OUR IMAGE FEATURES")

print(
    "Shape:",
    tuple(image_features.shape),
)

print(
    "Norm:",
    image_features
    .norm(dim=-1)
    .tolist(),
)


# ============================================================
# OUR TEXT FEATURES
# ============================================================

with torch.no_grad():

    text_features = (
        _get_clip_text_features(
            model,
            TEXTS,
        )
    )


print()
print("OUR TEXT FEATURES")

print(
    "Shape:",
    tuple(text_features.shape),
)

print(
    "Norm:",
    text_features
    .norm(dim=-1)
    .tolist(),
)


# ============================================================
# OUR COSINE SIMILARITIES
# ============================================================

manual_cosine = (
    image_features
    @ text_features.T
).squeeze(0)


print()
print("OUR COSINE SIMILARITIES")

for text, score in zip(
    TEXTS,
    manual_cosine,
):

    print(
        f"  {score.item():.6f}  {text}"
    )


# ============================================================
# CLIP LOGIT SCALE
# ============================================================

logit_scale = (
    model.model.logit_scale
    .exp()
    .detach()
    .item()
)


print()
print(
    "CLIP LOGIT SCALE:",
    logit_scale,
)


# ============================================================
# RECONSTRUCT LOGITS
# ============================================================

manual_logits = (
    manual_cosine
    * logit_scale
)


print()
print("OUR RECONSTRUCTED LOGITS")

for text, score in zip(
    TEXTS,
    manual_logits,
):

    print(
        f"  {score.item():.6f}  {text}"
    )


# ============================================================
# LOGIT COMPARISON
# ============================================================

logit_difference = (
    official_logits
    - manual_logits
)


print()
print("=" * 70)
print("FORWARD PATH COMPARISON")
print("=" * 70)

print()

for text, official, manual, diff in zip(
    TEXTS,
    official_logits,
    manual_logits,
    logit_difference,
):

    print(text)

    print(
        f"  Official : "
        f"{official.item():.6f}"
    )

    print(
        f"  Manual   : "
        f"{manual.item():.6f}"
    )

    print(
        f"  Difference: "
        f"{diff.item():.6f}"
    )

    print()


# ============================================================
# PREDICTION COMPARISON
# ============================================================

manual_prediction = TEXTS[
    manual_logits.argmax().item()
]


print(
    "Official prediction:",
    official_prediction,
)

print(
    "Manual prediction:",
    manual_prediction,
)


# ============================================================
# FINAL DIAGNOSTIC STATUS
# ============================================================

max_logit_difference = (
    logit_difference
    .abs()
    .max()
    .item()
)

max_pixel_difference = (
    preprocessing_difference
    .abs()
    .max()
    .item()
)


print()
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print()

print(
    "Max preprocessing difference:",
    f"{max_pixel_difference:.10f}",
)

print(
    "Max logit difference:",
    f"{max_logit_difference:.10f}",
)

print()

if max_logit_difference < 1e-3:

    print(
        "STATUS: PASS"
    )

    print(
        "Manual forward path closely "
        "matches Hugging Face."
    )

else:

    print(
        "STATUS: MISMATCH"
    )

    print(
        "Manual forward path does not "
        "yet reproduce Hugging Face."
    )


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
