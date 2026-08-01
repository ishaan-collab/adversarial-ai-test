import torch


def predict(
    model,
    image,
    weights,
    preprocess=None,
):
    """
    Run model inference and return
    prediction information.
    """

    if preprocess is not None:
        model_input = preprocess(image)
    else:
        model_input = image

    with torch.no_grad():
        output = model(
            model_input
        )

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    confidence, class_id = (
        probabilities.max(dim=1)
    )

    class_id = class_id.item()
    confidence = confidence.item()

    category = weights.meta[
        "categories"
    ][class_id]

    return {
        "class_id": class_id,
        "category": category,
        "confidence": confidence,
        "probabilities": probabilities,
    }
