import torch
from evaluation.metrics import compute_perturbation_metrics

def deepfool_attack(
    model,
    image,
    label=None,
    max_iterations=20,
    overshoot=0.02,
    preprocess=None,
    num_classes=10,
):
    """
    DeepFool attack for an image classification model.

    DeepFool iteratively estimates the closest decision
    boundary and moves the image toward it.

    Args:
        model:
            PyTorch classification model.

        image:
            Raw image tensor in [0, 1].

        label:
            Original class label. If None, the model's
            current prediction is used.

        max_iterations:
            Maximum number of attack iterations.

        overshoot:
            Small factor used to move slightly beyond
            the estimated decision boundary.

        preprocess:
            Optional preprocessing function.

        num_classes:
            Number of competing classes to examine.
            We use the top classes rather than all 1000
            ImageNet classes for efficiency.

    Returns:
        Adversarial image tensor in [0, 1].
    """

    original_image = (
        image.detach().clone()
    )

    adversarial_image = (
        original_image.clone()
    )

    # ============================================================
    # Determine original class
    # ============================================================

    with torch.no_grad():

        if preprocess is not None:
            model_input = preprocess(
                adversarial_image
            )
        else:
            model_input = adversarial_image

        output = model(
            model_input
        )

    predicted_class = (
        output.argmax(dim=1)
        .item()
    )

    if label is not None:
        original_class = label.item()
    else:
        original_class = predicted_class

    # ============================================================
    # DeepFool iterations
    # ============================================================

    for iteration in range(
        max_iterations
    ):

        # --------------------------------------------------------
        # IMPORTANT:
        # Make a fresh leaf tensor every iteration.
        # --------------------------------------------------------

        adversarial_image = (
            adversarial_image
            .detach()
            .requires_grad_(True)
        )

        # --------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------

        if preprocess is not None:
            model_input = preprocess(
                adversarial_image
            )
        else:
            model_input = adversarial_image

        output = model(
            model_input
        )

        current_class = (
            output.argmax(dim=1)
            .item()
        )

        # --------------------------------------------------------
        # Attack succeeded
        # --------------------------------------------------------

        if current_class != original_class:
            break

        # --------------------------------------------------------
        # Select strongest competing classes
        # --------------------------------------------------------

        k = min(
            num_classes,
            output.shape[1],
        )

        top_indices = (
            torch.topk(
                output[0],
                k=k,
            ).indices
        )

        # --------------------------------------------------------
        # Gradient of original class
        # --------------------------------------------------------

        original_score = (
            output[0, original_class]
        )

        original_gradient = torch.autograd.grad(
            original_score,
            adversarial_image,
            retain_graph=True,
        )[0]

        # --------------------------------------------------------
        # Find closest decision boundary
        # --------------------------------------------------------

        best_distance = None
        best_gradient = None

        for class_id in top_indices:

            class_id = (
                class_id.item()
            )

            if class_id == original_class:
                continue

            class_score = (
                output[0, class_id]
            )

            class_gradient = torch.autograd.grad(
                class_score,
                adversarial_image,
                retain_graph=True,
            )[0]

            gradient_difference = (
                class_gradient
                - original_gradient
            )

            score_difference = (
                class_score
                - original_score
            )

            gradient_norm = (
                torch.norm(
                    gradient_difference
                )
                + 1e-12
            )

            distance = (
                torch.abs(
                    score_difference
                )
                / gradient_norm
            )

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_gradient = (
                    gradient_difference
                )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if best_gradient is None:
            break

        # --------------------------------------------------------
        # Move toward closest boundary
        # --------------------------------------------------------

        gradient_norm = (
            torch.norm(
                best_gradient
            )
            + 1e-12
        )

        perturbation = (
            (
                best_distance
                / gradient_norm
            )
            * best_gradient
        )

        perturbation = (
            perturbation
            * (1.0 + overshoot)
        )

        # --------------------------------------------------------
        # Update image
        # --------------------------------------------------------

        adversarial_image = (
            adversarial_image.detach()
            + perturbation.detach()
        )

        adversarial_image = (
            torch.clamp(
                adversarial_image,
                0.0,
                1.0,
            )
        )

    return adversarial_image.detach()