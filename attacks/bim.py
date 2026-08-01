import torch


def bim_attack(
    model,
    image,
    label,
    epsilon,
    alpha,
    iterations,
    preprocess=None,
):
    """
    Basic Iterative Method (BIM).

    Args:
        model:
            PyTorch classification model.

        image:
            Original image tensor in [0, 1].

        label:
            Class label used for the untargeted attack.

        epsilon:
            Maximum L-infinity perturbation.

        alpha:
            Step size per iteration.

        iterations:
            Number of attack iterations.

        preprocess:
            Optional preprocessing function.

    Returns:
        Adversarial image tensor in [0, 1].
    """

    original_image = image.clone().detach()

    adversarial_image = original_image.clone().detach()

    for step in range(iterations):

        adversarial_image.requires_grad = True

        # -----------------------------------------
        # Model input
        # -----------------------------------------

        if preprocess is not None:
            model_input = preprocess(
                adversarial_image
            )
        else:
            model_input = adversarial_image

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------

        output = model(model_input)

        loss = torch.nn.functional.cross_entropy(
            output,
            label,
        )

        # -----------------------------------------
        # Compute gradient
        # -----------------------------------------

        model.zero_grad()

        loss.backward()

        gradient = adversarial_image.grad

        # -----------------------------------------
        # Take gradient step
        # -----------------------------------------

        adversarial_image = (
            adversarial_image
            + alpha * gradient.sign()
        )

        # -----------------------------------------
        # Project back into epsilon ball
        # -----------------------------------------

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

        # -----------------------------------------
        # Keep pixels in valid range
        # -----------------------------------------

        adversarial_image = torch.clamp(
            adversarial_image,
            0.0,
            1.0,
        ).detach()

    return adversarial_image
