import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


# ============================================================
# CLIP IMAGE PREPROCESSING
# ============================================================

def _get_clip_image_preprocessed(
    model,
    image,
):
    """
    Differentiable approximation of the Hugging Face CLIP
    image preprocessing pipeline.

    Input:
        image: [B, 3, H, W], float32 in [0, 1]

    Output:
        [B, 3, 224, 224]
    """

    processor = model.processor.image_processor

    size = processor.size
    crop_size = processor.crop_size

    shortest_edge = getattr(
        size,
        "shortest_edge",
        None,
    )

    if shortest_edge is None:

        if isinstance(size, dict):

            shortest_edge = size.get(
                "shortest_edge"
            )

    if shortest_edge is None:

        raise ValueError(
            "Unable to determine CLIP shortest edge."
        )

    shortest_edge = int(
        shortest_edge
    )

    crop_height = getattr(
        crop_size,
        "height",
        shortest_edge,
    )

    crop_width = getattr(
        crop_size,
        "width",
        shortest_edge,
    )

    if isinstance(crop_size, dict):

        crop_height = crop_size.get(
            "height",
            crop_height,
        )

        crop_width = crop_size.get(
            "width",
            crop_width,
        )

    crop_height = int(crop_height)
    crop_width = int(crop_width)

    if image.ndim != 4:

        raise ValueError(
            "Expected [B, 3, H, W], got "
            f"{tuple(image.shape)}"
        )

    if image.shape[1] != 3:

        raise ValueError(
            "Expected RGB image."
        )

    _, _, h, w = image.shape

    scale = (
        shortest_edge
        / min(h, w)
    )

    new_h = int(
        round(h * scale)
    )

    new_w = int(
        round(w * scale)
    )

    if h <= w:

        new_h = shortest_edge

    else:

        new_w = shortest_edge

    image = resize(
        image,
        [new_h, new_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    _, _, h, w = image.shape

    top = (
        h - crop_height
    ) // 2

    left = (
        w - crop_width
    ) // 2

    image = image[
        :,
        :,
        top:top + crop_height,
        left:left + crop_width,
    ]

    mean = torch.as_tensor(
        processor.image_mean,
        device=image.device,
        dtype=image.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.as_tensor(
        processor.image_std,
        device=image.device,
        dtype=image.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    return (
        image - mean
    ) / std


# ============================================================
# CLIP IMAGE FEATURES
# ============================================================

def _get_clip_image_features(
    model,
    pixel_values,
):
    """
    Extract normalized CLIP image embeddings.
    """

    vision_outputs = (
        model.model.vision_model(
            pixel_values=pixel_values
        )
    )

    pooled_output = (
        vision_outputs.pooler_output
    )

    image_features = (
        model.model.visual_projection(
            pooled_output
        )
    )

    return F.normalize(
        image_features,
        dim=-1,
    )

# ============================================================
# SIGLIP IMAGE PREPROCESSING
# ============================================================

def _get_siglip_image_preprocessed(
    model,
    image,
):
    """
    Differentiable SigLIP image preprocessing.

    Input:
        image: [B, 3, H, W], float32 in [0, 1]

    Output:
        [B, 3, H_target, W_target]

    This intentionally avoids calling the Hugging Face
    processor on the image because the PGD attack requires
    gradients to flow from the model back to the input image.
    """

    processor = model.processor.image_processor

    size = processor.size

    # --------------------------------------------------------
    # Resolve target image dimensions
    # --------------------------------------------------------

    if isinstance(size, dict):

        target_height = size.get(
            "height",
            224,
        )

        target_width = size.get(
            "width",
            224,
        )

    else:

        target_height = getattr(
            size,
            "height",
            None,
        )

        target_width = getattr(
            size,
            "width",
            None,
        )

        if target_height is None:
            target_height = 224

        if target_width is None:
            target_width = target_height

    target_height = int(target_height)
    target_width = int(target_width)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if image.ndim != 4:

        raise ValueError(
            "Expected image with shape "
            f"[B, 3, H, W], got {tuple(image.shape)}"
        )

    if image.shape[1] != 3:

        raise ValueError(
            "Expected RGB image with 3 channels."
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------
    #
    # SigLIP's standard image processor uses a fixed spatial
    # resolution rather than CLIP's shortest-edge + center-crop
    # pipeline.
    #

    image = resize(
        image,
        [target_height, target_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # --------------------------------------------------------
    # Rescale
    # --------------------------------------------------------
    #
    # Our attack image is already represented in [0, 1].
    # Therefore no additional 1/255 scaling is required.
    #

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = torch.as_tensor(
        processor.image_mean,
        device=image.device,
        dtype=image.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.as_tensor(
        processor.image_std,
        device=image.device,
        dtype=image.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    return (
        image - mean
    ) / std


# ============================================================
# SIGLIP IMAGE FEATURES
# ============================================================

def _get_siglip_image_features(
    model,
    pixel_values,
):
    """
    Extract normalized SigLIP image embeddings.

    Uses Hugging Face's native get_image_features()
    interface instead of manually accessing internal
    projection layers.
    """

    features = model.model.get_image_features(
        pixel_values=pixel_values,
    )

    # Some Transformers versions may expose a model-output
    # object rather than the final tensor.

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

    if not torch.is_tensor(features):

        raise TypeError(
            "SigLIP image feature extraction returned "
            f"{type(features).__name__} instead of "
            "a torch.Tensor."
        )

    return F.normalize(
        features,
        p=2,
        dim=-1,
    )

# ============================================================
# CLIP TEXT FEATURES
# ============================================================

def _get_clip_text_features(
    model,
    texts,
):
    """
    Extract normalized CLIP text embeddings.
    """

    tokenizer_output = model.processor(
        text=texts,
        return_tensors="pt",
        padding=True,
    )

    input_ids = (
        tokenizer_output["input_ids"]
        .to(model.device)
    )

    attention_mask = (
        tokenizer_output["attention_mask"]
        .to(model.device)
    )

    text_outputs = (
        model.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    )

    pooled_output = (
        text_outputs.pooler_output
    )

    text_features = (
        model.model.text_projection(
            pooled_output
        )
    )

    return F.normalize(
        text_features,
        dim=-1,
    )


# ============================================================
# TARGETED CLIP PGD
# ============================================================

def targeted_clip_pgd(
    model,
    image,
    source_text,
    target_text,
    epsilon=8 / 255,
    alpha=2 / 255,
    iterations=20,
):
    """
    Targeted PGD attack against CLIP.

    Args:
        model:
            VLM wrapper returned by get_vlm().

        image:
            [1, 3, H, W] tensor in [0, 1].

        source_text:
            Original semantic class.

        target_text:
            Desired target class.

        epsilon:
            L-infinity perturbation budget.

        alpha:
            PGD step size.

        iterations:
            Number of PGD iterations.

    Returns:
        Adversarial image in [0, 1].
    """

    if image.ndim != 4:

        raise ValueError(
            "image must have shape [B, 3, H, W]"
        )

    original = (
        image.detach()
        .clone()
    )

    lower = torch.clamp(
        original - epsilon,
        0.0,
        1.0,
    )

    upper = torch.clamp(
        original + epsilon,
        0.0,
        1.0,
    )

    texts = [
        source_text,
        target_text,
    ]

    text_features = (
        _get_clip_text_features(
            model,
            texts,
        )
        .detach()
    )

    x_adv = (
        original.clone()
    )

    for iteration in range(
        iterations
    ):

        x_adv.requires_grad_(True)

        pixel_values = (
            _get_clip_image_preprocessed(
                model,
                x_adv,
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

        source_similarity = (
            similarities[:, 0]
        )

        target_similarity = (
            similarities[:, 1]
        )

        # ----------------------------------------------------
        # Targeted objective
        #
        # We want:
        #
        # target_similarity ↑
        # source_similarity ↓
        #
        # Therefore minimize:
        #
        # source - target
        # ----------------------------------------------------

        loss = (
            source_similarity
            - target_similarity
        ).mean()

        gradient = torch.autograd.grad(
            loss,
            x_adv,
            retain_graph=False,
            create_graph=False,
        )[0]

        # ----------------------------------------------------
        # TARGETED PGD
        #
        # Minimize the loss.
        # ----------------------------------------------------

        with torch.no_grad():

            x_adv = (
                x_adv
                - alpha
                * gradient.sign()
            )

            x_adv = torch.max(
                torch.min(
                    x_adv,
                    upper,
                ),
                lower,
            )

            x_adv = torch.clamp(
                x_adv,
                0.0,
                1.0,
            )

        if (
            iteration == 0
            or (iteration + 1) % 5 == 0
            or iteration == iterations - 1
        ):

            print(
                f"Iteration "
                f"{iteration + 1:02d}/"
                f"{iterations} | "
                f"loss={loss.item():.6f} | "
                f"source="
                f"{source_similarity.mean().item():.6f} | "
                f"target="
                f"{target_similarity.mean().item():.6f}"
            )

    return x_adv.detach()