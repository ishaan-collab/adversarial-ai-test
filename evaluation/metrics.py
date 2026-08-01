import torch


def compute_perturbation_metrics(
    original_image,
    adversarial_image,
):
    """
    Compute perturbation statistics in raw [0, 1]
    pixel space.

    Returns:
        dict containing:
            linf
            l2
            mean_perturbation
    """

    perturbation = (
        adversarial_image
        - original_image
    )

    # L-infinity norm
    linf = (
        perturbation.abs()
        .max()
        .item()
    )

    # L2 norm
    l2 = torch.norm(
        perturbation.reshape(
            perturbation.shape[0],
            -1,
        ),
        p=2,
        dim=1,
    ).item()

    # Mean absolute perturbation
    mean_perturbation = (
        perturbation.abs()
        .mean()
        .item()
    )

    return {
        "linf": linf,
        "l2": l2,
        "mean_perturbation": mean_perturbation,
    }