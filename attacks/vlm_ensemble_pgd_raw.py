import random

import numpy as np
import torch

from attacks.vlm_pgd import (
    _get_clip_image_preprocessed,
    _get_clip_image_features,
    _get_clip_text_features,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible attack initialization.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL SCORE FUNCTION
# ============================================================

def _get_model_scores(
    model,
    image,
    texts,
    text_features=None,
):
    """
    Return differentiable image-text scores for a VLM.

    CLIP:
        Uses the existing differentiable CLIP embedding path.

    SigLIP:
        Uses the adapter's differentiable raw-logit interface.

    Returns:
        Tensor of shape [2]:

            [source_score, target_score]
    """

    model_type = getattr(
        model,
        "model_type",
        None,
    )

    # ========================================================
    # CLIP
    # ========================================================

    if model_type == "clip":

        if text_features is None:
            raise ValueError(
                "CLIP text features must be precomputed."
            )

        pixel_values = (
            _get_clip_image_preprocessed(
                model,
                image,
            )
        )

        image_features = (
            _get_clip_image_features(
                model,
                pixel_values,
            )
        )

        similarities = (
            image_features
            @ text_features.T
        )

        return similarities[0]

    # ========================================================
    # SIGLIP
    # ========================================================

    if model_type == "siglip":

        return model.get_image_text_scores(
            image,
            texts,
        )

    # ========================================================
    # UNSUPPORTED MODEL
    # ========================================================

    raise ValueError(
        f"Unsupported VLM model type: {model_type!r}. "
        f"Supported types are: 'clip' and 'siglip'."
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
    Generate one shared targeted adversarial image against an
    ensemble of vision-language models.

    The attack minimizes the average source-vs-target margin:

        loss = source_score - target_score

    Therefore:

        source score -> decreases
        target score -> increases

    Targeted PGD update:

        x_adv <- x_adv - alpha * sign(gradient)

    Args:
        models:
            List of VLM adapters.

        image:
            Tensor [1, 3, H, W] in [0, 1].

        source_text:
            Original/source semantic description.

        target_text:
            Desired target semantic description.

        epsilon:
            L-infinity perturbation budget.

        alpha:
            PGD step size.

        seed:
            Random seed used when random_start=True.

        iterations:
            Number of PGD iterations.

        random_start:
            If True, initialize randomly inside the epsilon ball.

        return_details:
            If True, return:

                adversarial, details

            Otherwise return only:

                adversarial

    Returns:
        Tensor [1, 3, H, W] in [0, 1].

        Optionally returns attack details.
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

    if not isinstance(iterations, int):
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
    # VALIDATE MODEL DEVICES
    # ========================================================

    for model in models:

        if model.device != device:

            raise ValueError(
                "All ensemble models must be on "
                "the same device."
            )

    # ========================================================
    # L-INFINITY BOUNDS
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
    # INITIALIZE ADVERSARIAL IMAGE
    # ========================================================

    if random_start:

        random_noise = torch.empty_like(
            original
        ).uniform_(
            -float(epsilon),
            float(epsilon),
        )

        adversarial = (
            original
            + random_noise
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
    # PRECOMPUTE CLIP TEXT FEATURES
    # ========================================================
    #
    # Text features do not depend on the adversarial image.
    #
    # SigLIP does not need this because its wrapper handles
    # the text inputs directly through get_image_text_scores().
    # ========================================================

    text_features = []

    texts = [
        source_text,
        target_text,
    ]

    for model in models:

        if getattr(
            model,
            "model_type",
            None,
        ) == "clip":

            features = _get_clip_text_features(
                model,
                texts,
            )

            features = features.detach()

        else:

            features = None

        text_features.append(
            features
        )

    # ========================================================
    # ATTACK HISTORY
    # ========================================================

    history = []

    # ========================================================
    # TARGETED PGD LOOP
    # ========================================================

    for iteration in range(iterations):

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

        for model, model_text_features in zip(
            models,
            text_features,
        ):

            scores = _get_model_scores(
                model=model,
                image=adversarial,
                texts=texts,
                text_features=model_text_features,
            )

            source_score = scores[0]

            target_score = scores[1]

            # ------------------------------------------------
            # TARGETED OBJECTIVE
            #
            # Minimize:
            #
            #     source - target
            #
            # Therefore:
            #
            #     source ↓
            #     target ↑
            # ------------------------------------------------

            model_loss = (
                source_score
                - target_score
            )

            total_loss = (
                total_loss
                + model_loss
            )

            model_statistics.append(
                (
                    model.name,
                    source_score.detach(),
                    target_score.detach(),
                )
            )

        # ====================================================
        # AVERAGE ENSEMBLE LOSS
        # ====================================================

        ensemble_loss = (
            total_loss
            / len(models)
        )

        # ====================================================
        # COMPUTE IMAGE GRADIENT
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
            # PROJECT INTO L-INFINITY BALL
            # ------------------------------------------------

            adversarial = torch.maximum(
                torch.minimum(
                    adversarial,
                    upper_bound,
                ),
                lower_bound,
            )

            # ------------------------------------------------
            # VALID PIXEL RANGE
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

            iteration_models = {}

            for (
                model_name,
                source_score,
                target_score,
            ) in model_statistics:

                iteration_models[
                    model_name
                ] = {
                    "source_score":
                        source_score.item(),

                    "target_score":
                        target_score.item(),
                }

            history.append(
                {
                    "iteration":
                        iteration + 1,

                    "ensemble_loss":
                        ensemble_loss.item(),

                    "models":
                        iteration_models,
                }
            )

            print(
                f"Iteration "
                f"{iteration + 1:02d}/"
                f"{iterations} | "
                f"ensemble loss: "
                f"{ensemble_loss.item():.6f}"
            )

            for (
                model_name,
                source_score,
                target_score,
            ) in model_statistics:

                print(
                    f"  {model_name:<16}"
                    f" source="
                    f"{source_score.item():.6f}"
                    f" target="
                    f"{target_score.item():.6f}"
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
        "seed": int(seed),
        "iterations": int(iterations),
        "random_start": bool(random_start),
        "source_text": source_text,
        "target_text": target_text,
        "linf": linf,
        "l2": l2,
        "mean_abs": mean_abs,
        "history": history,
    }

    return adversarial, details