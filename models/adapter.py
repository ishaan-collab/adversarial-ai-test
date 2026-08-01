class ModelAdapter:
    """
    Standard interface around a vision model.

    The adapter keeps model-specific preprocessing
    separate from attacks and evaluation.
    """

    def __init__(
        self,
        model,
        weights,
        device,
        preprocess,
        name,
    ):
        self.model = model
        self.weights = weights
        self.device = device
        self.preprocess = preprocess
        self.name = name

    def predict(self, image):
        """
        Run the model on a raw [0, 1] image tensor.
        """

        model_input = self.preprocess(
            image
        )

        return self.model(
            model_input
        )
