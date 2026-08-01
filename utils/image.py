from PIL import Image

import torch
from torchvision.transforms import functional as TF


def load_image(path):
    """
    Load an image and convert it into the common
    224x224 [0,1] representation used by the
    adversarial attacks.

    Spatial preprocessing:
        resize shortest side -> 232
        center crop -> 224

    Normalization is intentionally NOT performed here.
    """

    image = Image.open(path).convert("RGB")

    tensor = TF.to_tensor(image)

    # Match torchvision ImageNet weight transforms.
    tensor = TF.resize(
        tensor,
        232,
    )

    tensor = TF.center_crop(
        tensor,
        [224, 224],
    )

    return image, tensor.unsqueeze(0)


def save_tensor_as_image(tensor, path):
    """
    Save a [1,3,H,W] or [3,H,W] tensor in [0,1].
    """

    tensor = tensor.detach().cpu()

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    tensor = (
        tensor * 255
    ).byte()

    tensor = tensor.permute(
        1,
        2,
        0,
    )

    image = Image.fromarray(
        tensor.numpy()
    )

    image.save(path)