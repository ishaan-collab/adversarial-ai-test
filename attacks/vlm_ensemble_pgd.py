import random

import numpy as np
import torch
import torch.nn.functional as F

from attacks.vlm_pgd import (
    _get_clip_image_preprocessed,
    _get_clip_image_features,
    _get_siglip_image_preprocessed,
    _get_siglip_image_features,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducible attack initialization.

    When random_start=False, the PGD attack itself is
    deterministic with respect to the model and input.
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# VLM TEXT FEATURES
# ============================================================

def _get_vlm_text_features(
    model,
    texts,
):
    """
    Extract normalized text embeddings for a supported VLM.

    Supported models:

        - CLIP
        - SigLIP

    Returns:

        Tensor of shape [num_texts, embedding_dim]
    """

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokenizer_output = model.processor(
        text=texts,
        return_tensors="pt",
        padding=True,
    )

    tokenizer_output = {
        key: value.to(model.device)
        for key, value in tokenizer_output.items()
        if hasattr(value, "to")
    }

    # --------------------------------------------------------
    # Model-specific text encoder
    # --------------------------------------------------------

    if model.model_type == "clip":

        features = model.model.get_text_features(
            **tokenizer_output
        )

    elif model.model_type == "siglip":

        features = model.model.get_text_features(
            **tokenizer_output
        )

    else:

        raise ValueError(
            "Unsupported VLM type: "
            f"{model.model_type}"
        )

    # --------------------------------------------------------
    # Handle Transformers model-output variants
    # --------------------------------------------------------

    if hasattr(
        features,
        "pooler_output",
    ):

        features = features.pooler_output

    elif hasattr(
        features,
        "last_hidden_state",
    ):

        features = features.last_hidden_state[:, 0]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not torch.is_tensor(features):

        raise TypeError(
            "Text feature extraction returned "
            f"{type(features).__name__} instead of "
            "a torch.Tensor."
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    return F.normalize(
        features,
        p=2,
        dim=-1,
    )


# ============================================================
# VLM IMAGE FEATURES
# ============================================================

def _get_vlm_image_features(
    model,
    image,
):
    """
    Extract normalized image embeddings using the correct
    differentiable preprocessing pipeline for each VLM.
    """

    if model.model_type == "clip":

        pixel_values = (
            _get_clip_image_preprocessed(
                model,
                image,
            )
        )

        return _get_clip_image_features(
            model,
            pixel_values,
        )

    if model.model_type == "siglip":

        pixel_values = (
            _get_siglip_image_preprocessed(
                model,
                image,
            )
        )

        return _get_siglip_image_features(
            model,
            pixel_values,
        )

    raise ValueError(
        "Unsupported VLM type: "
        f"{model.model_type}"
    )


# ============================================================
# VLM ENSEMBLE TARGETED PGD
# ============================================================

def vlm_ensemble_pgd(
    models,
    image,
    source_text,
    target_text,
    epsilon=8 / 255,
    alpha=2 / 255,
    seed=42,
    iterations=20,
    random_start=False,
    return_details=False,
):
    """
    Generate one shared targeted adversarial image against
    an ensemble of vision-language models.

    The attack minimizes:

        source_similarity - target_similarity

    Therefore:

        source similarity -> decreases
        target similarity -> increases

    The same adversarial image is optimized jointly against
    every model in the ensemble.

    Args:
        models:
            List of VisionLanguageAdapter instances.

        image:
            Tensor [1, 3, H, W] in [0, 1].

        source_text:
            Original semantic description.

        target_text:
            Desired target semantic description.

        epsilon:
            L-infinity perturbation budget.

        alpha:
            PGD step size.

        seed:
            Random seed.

        iterations:
            Number of PGD iterations.

        random_start:
            Whether to initialize randomly inside the
            epsilon-ball.

        return_details:
            Whether to return attack metrics and history.

    Returns:
        adversarial

        or:

        adversarial, details
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not models:

        raise ValueError(
            "At least one model is required."
        )

    if image.ndim != 4:

        raise ValueError(
            "Expected image with shape [B, 3, H, W], "
            f"got {tuple(image.shape)}"
        )

    if image.shape[0] != 1:

        raise ValueError(
            "This implementation currently expects "
            "a single image."
        )

    if image.shape[1] != 3:

        raise ValueError(
            "Expected RGB image with 3 channels."
        )

    if epsilon <= 0:

        raise ValueError(
            "epsilon must be positive."
        )

    if alpha <= 0:

        raise ValueError(
            "alpha must be positive."
        )

    if not isinstance(
        iterations,
        int,
    ):

        raise ValueError(
            "iterations must be an integer."
        )

    if iterations <= 0:

        raise ValueError(
            "iterations must be positive."
        )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(seed)

    # ========================================================
    # DEVICE
    # ========================================================

    device = models[0].device

    for model in models:

        if model.device != device:

            raise ValueError(
                "All ensemble models must be on "
                "the same device."
            )

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original = (
        image.detach()
        .clone()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    original = torch.clamp(
        original,
        0.0,
        1.0,
    )

    # ========================================================
    # L-INFINITY CONSTRAINT
    # ========================================================

    epsilon_tensor = torch.tensor(
        float(epsilon),
        device=device,
        dtype=original.dtype,
    )

    lower_bound = torch.clamp(
        original - epsilon_tensor,
        0.0,
        1.0,
    )

    upper_bound = torch.clamp(
        original + epsilon_tensor,
        0.0,
        1.0,
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    if random_start:

        noise = torch.empty_like(
            original
        ).uniform_(
            -float(epsilon),
            float(epsilon),
        )

        adversarial = (
            original + noise
        )

        adversarial = torch.maximum(
            torch.minimum(
                adversarial,
                upper_bound,
            ),
            lower_bound,
        )

    else:

        adversarial = (
            original.clone()
        )

    adversarial = torch.clamp(
        adversarial,
        0.0,
        1.0,
    )

    # ========================================================
    # PRECOMPUTE TEXT FEATURES
    # ========================================================

    text_features = []

    for model in models:

        features = _get_vlm_text_features(
            model,
            [
                source_text,
                target_text,
            ],
        )

        text_features.append(
            features.detach()
        )

    # ========================================================
    # ATTACK HISTORY
    # ========================================================

    history = []

    # ========================================================
    # TARGETED PGD LOOP
    # ========================================================

    for iteration in range(
        iterations
    ):

        adversarial.requires_grad_(True)

        total_loss = torch.zeros(
            (),
            device=device,
            dtype=adversarial.dtype,
        )

        model_statistics = []

        # ====================================================
        # ENSEMBLE FORWARD PASS
        # ====================================================

        for (
            model,
            model_text_features,
        ) in zip(
            models,
            text_features,
        ):

            # ------------------------------------------------
            # Differentiable model-specific image encoding
            # ------------------------------------------------

            image_features = (
                _get_vlm_image_features(
                    model,
                    adversarial,
                )
            )

            # ------------------------------------------------
            # Cosine similarities
            # ------------------------------------------------

            similarities = (
                image_features
                @ model_text_features.T
            )

            source_similarity = (
                similarities[0, 0]
            )

            target_similarity = (
                similarities[0, 1]
            )

            # ------------------------------------------------
            # Targeted objective
            # ------------------------------------------------

            model_loss = (
                source_similarity
                - target_similarity
            )

            total_loss = (
                total_loss
                + model_loss
            )

            model_statistics.append(
                {
                    "name": model.name,
                    "source": source_similarity.detach(),
                    "target": target_similarity.detach(),
                }
            )

        # ====================================================
        # AVERAGE ENSEMBLE LOSS
        # ====================================================

        ensemble_loss = (
            total_loss
            / len(models)
        )

        # ====================================================
        # GRADIENT
        # ====================================================

        gradient = torch.autograd.grad(
            ensemble_loss,
            adversarial,
            retain_graph=False,
            create_graph=False,
        )[0]

        # ====================================================
        # TARGETED PGD UPDATE
        # ====================================================

        with torch.no_grad():

            adversarial = (
                adversarial
                - float(alpha)
                * gradient.sign()
            )

            # ------------------------------------------------
            # Project into epsilon-ball
            # ------------------------------------------------

            adversarial = torch.maximum(
                torch.minimum(
                    adversarial,
                    upper_bound,
                ),
                lower_bound,
            )

            # ------------------------------------------------
            # Valid pixel range
            # ------------------------------------------------

            adversarial = torch.clamp(
                adversarial,
                0.0,
                1.0,
            )

        # ====================================================
        # LOGGING
        # ====================================================

        should_print = (
            iteration == 0
            or (iteration + 1) % 5 == 0
            or iteration == iterations - 1
        )

        if should_print:

            history_models = {}

            for stats in model_statistics:

                history_models[
                    stats["name"]
                ] = {
                    "source_similarity":
                        stats["source"].item(),

                    "target_similarity":
                        stats["target"].item(),
                }

            history.append(
                {
                    "iteration":
                        iteration + 1,

                    "ensemble_loss":
                        ensemble_loss.item(),

                    "models":
                        history_models,
                }
            )

            print(
                f"Iteration "
                f"{iteration + 1:02d}/"
                f"{iterations} | "
                f"ensemble loss: "
                f"{ensemble_loss.item():.6f}"
            )

            for stats in model_statistics:

                print(
                    f"  {stats['name']:<16}"
                    f" source="
                    f"{stats['source'].item():.6f}"
                    f" target="
                    f"{stats['target'].item():.6f}"
                )

    # ========================================================
    # FINAL STRICT PROJECTION
    # ========================================================

    with torch.no_grad():

        adversarial = torch.maximum(
            torch.minimum(
                adversarial,
                upper_bound,
            ),
            lower_bound,
        )

        adversarial = torch.clamp(
            adversarial,
            0.0,
            1.0,
        )

    adversarial = adversarial.detach()

    # ========================================================
    # SIMPLE RETURN
    # ========================================================

    if not return_details:

        return adversarial

    # ========================================================
    # PERTURBATION METRICS
    # ========================================================

    with torch.no_grad():

        perturbation = (
            adversarial
            - original
        )

        linf = (
            perturbation
            .abs()
            .max()
            .item()
        )

        l2 = (
            torch.norm(
                perturbation.reshape(
                    perturbation.shape[0],
                    -1,
                ),
                p=2,
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

    # ========================================================
    # DETAILS
    # ========================================================

    details = {
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "iterations": int(iterations),
        "seed": seed,
        "random_start": bool(random_start),
        "source_text": source_text,
        "target_text": target_text,
        "linf": linf,
        "l2": l2,
        "mean_abs": mean_abs,
        "history": history,
    }

    return adversarial, details