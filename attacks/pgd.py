import torch
import torch.nn.functional as F


def pgd_attack(
    model,
    image,
    label,
    epsilon,
    alpha,
    iterations,
    preprocess=None,
    random_start=True,
):
    """
    Projected Gradient Descent (PGD)
    for an untargeted L-infinity attack.

    image:
        Raw image tensor in [0, 1].

    epsilon:
        Maximum L-infinity perturbation.

    alpha:
        Step size.

    iterations:
        Number of gradient steps.

    preprocess:
        Optional model preprocessing function.

    random_start:
        If True, initialize randomly inside
        the epsilon L-infinity ball.
    """

    original_image = image.detach().clone()

    # ========================================================
    # Initialization
    # ========================================================

    if random_start:

        random_noise = torch.empty_like(
            original_image
        ).uniform_(
            -epsilon,
            epsilon,
        )

        adversarial_image = (
            original_image
            + random_noise
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

    # ========================================================
    # Iterative attack
    # ========================================================

    for step in range(iterations):

        adversarial_image.requires_grad_(True)

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        if preprocess is not None:

            model_input = preprocess(
                adversarial_image
            )

        else:

            model_input = adversarial_image

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            model_input
        )

        loss = F.cross_entropy(
            output,
            label,
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        model.zero_grad()

        if adversarial_image.grad is not None:
            adversarial_image.grad.zero_()

        loss.backward()

        gradient = (
            adversarial_image.grad
        )

        # ----------------------------------------------------
        # Gradient ascent
        # ----------------------------------------------------

        adversarial_image = (
            adversarial_image
            + alpha * gradient.sign()
        )

        # ----------------------------------------------------
        # Project into epsilon ball
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Valid image range
        # ----------------------------------------------------

        adversarial_image = torch.clamp(
            adversarial_image,
            0.0,
            1.0,
        ).detach()

    return adversarial_image
