import torch


def fgsm_attack(
    model,
    image,
    label,
    epsilon,
    preprocess=None
):
    """
    Generate an adversarial example using FGSM.

    Args:
        model: PyTorch classification model
        image: Raw image tensor in [0, 1]
        label: Target/correct class label
        epsilon: Maximum L-infinity perturbation
        preprocess: Optional preprocessing function

    Returns:
        adversarial_image: Tensor in [0, 1]
    """

    # Work on a separate tensor
    image = image.clone().detach()

    # Enable gradients with respect to the image
    image.requires_grad = True

    # Apply preprocessing before feeding image to model
    if preprocess is not None:
        model_input = preprocess(image)
    else:
        model_input = image

    # Forward pass
    output = model(model_input)

    # Calculate loss
    loss = torch.nn.functional.cross_entropy(
        output,
        label
    )

    # Clear model gradients
    model.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient of loss with respect to image
    gradient = image.grad

    # FGSM perturbation
    perturbation = epsilon * gradient.sign()

    # Create adversarial image
    adversarial_image = image + perturbation

    # Keep pixels valid
    adversarial_image = torch.clamp(
        adversarial_image,
        0.0,
        1.0
    )

    return adversarial_image.detach()