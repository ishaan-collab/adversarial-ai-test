import torch

from autoattack import AutoAttack


def autoattack_attack(
    model,
    image,
    label,
    epsilon,
    preprocess=None,
    version="standard",
):
    """
    Generate an adversarial example using AutoAttack.

    AutoAttack operates on raw images in [0, 1].
    The optional preprocessing function is applied
    internally before the model receives the image.
    """

    # --------------------------------------------------
    # Model wrapper
    # --------------------------------------------------

    class PreprocessedModel(torch.nn.Module):

        def __init__(self, model, preprocess):
            super().__init__()

            self.model = model
            self.preprocess = preprocess

        def forward(self, x):

            if self.preprocess is not None:
                x = self.preprocess(x)

            return self.model(x)

    wrapped_model = PreprocessedModel(
        model=model,
        preprocess=preprocess,
    )

    wrapped_model.eval()

    # --------------------------------------------------
    # AutoAttack
    # --------------------------------------------------

    adversary = AutoAttack(
        wrapped_model,
        norm="Linf",
        eps=epsilon,
        version=version,
        verbose=True,
    )

    # --------------------------------------------------
    # Generate adversarial examples
    # --------------------------------------------------

    image = image.detach()
    label = label.detach()

    adversarial_image = adversary.run_standard_evaluation(
        image,
        label,
    )

    return adversarial_image.detach()
