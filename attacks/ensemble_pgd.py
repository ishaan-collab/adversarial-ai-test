import torch
import torch.nn.functional as F


def ensemble_pgd_attack(
    models,
    image,
    label,
    epsilon,
    alpha,
    iterations,
    preprocesses=None,
    weights=None,
    random_start=True,
):
    """
    Ensemble PGD attack.

    Optimizes the average weighted loss across
    multiple models.

    Args:
        models:
            List of PyTorch models.

        image:
            Raw image tensor in [0, 1].

        label:
            Correct class label.

        epsilon:
            L-infinity perturbation budget.

        alpha:
            PGD step size.

        iterations:
            Number of iterations.

        preprocesses:
            Model-specific preprocessing functions.

        weights:
            Optional loss weights.

        random_start:
            Start randomly inside epsilon ball.
    """

    original_image = (
        image.detach().clone()
    )

    number_of_models = len(models)

    if preprocesses is None:
        preprocesses = [
            None
            for _ in models
        ]

    if weights is None:
        weights = [
            1.0 / number_of_models
            for _ in models
        ]

    # --------------------------------------------------
    # Random initialization
    # --------------------------------------------------

    if random_start:

        noise = torch.empty_like(
            original_image
        ).uniform_(
            -epsilon,
            epsilon,
        )

        adversarial_image = (
            original_image + noise
        )

        adversarial_image = torch.clamp(
            adversarial_image,
            0.0,
            1.0,
        )

    else:

        adversarial_image = (
            original_image.clone()
        )

    # --------------------------------------------------
    # PGD iterations
    # --------------------------------------------------

    for _ in range(iterations):

        adversarial_image.requires_grad_(True)

        total_loss = 0.0

        # ----------------------------------------------
        # Calculate ensemble loss
        # ----------------------------------------------

        for model, preprocess, weight in zip(
            models,
            preprocesses,
            weights,
        ):

            if preprocess is not None:

                model_input = preprocess(
                    adversarial_image
                )

            else:

                model_input = adversarial_image

            output = model(
                model_input
            )

            loss = F.cross_entropy(
                output,
                label,
            )

            total_loss = (
                total_loss
                + weight * loss
            )

        # ----------------------------------------------
        # Gradient
        # ----------------------------------------------

        for model in models:
            model.zero_grad()

        if adversarial_image.grad is not None:
            adversarial_image.grad.zero_()

        total_loss.backward()

        gradient = (
            adversarial_image.grad
        )

        # ----------------------------------------------
        # Gradient ascent
        # ----------------------------------------------

        adversarial_image = (
            adversarial_image
            + alpha * gradient.sign()
        )

        # ----------------------------------------------
        # Project into epsilon ball
        # ----------------------------------------------

        perturbation = (
            adversarial_image
            - original_image
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon,
        )

        adversarial_image = (
            original_image
            + perturbation
        )

        # ----------------------------------------------
        # Valid image range
        # ----------------------------------------------

        adversarial_image = torch.clamp(
            adversarial_image,
            0.0,
            1.0,
        ).detach()

    return adversarial_image
