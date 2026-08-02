import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
)


class VisionLanguageAdapter:

    def __init__(
        self,
        model,
        processor,
        name,
        device,
        model_type,
    ):
        self.model = model
        self.processor = processor
        self.name = name
        self.device = device
        self.model_type = model_type

        self.model.eval()

    # ============================================================
    # IMAGE + TEXT SCORING
    # ============================================================

    def score_image_text(
        self,
        image,
        texts,
    ):
        """
        Evaluate image-text compatibility.

        This method is intended for evaluation/inference.

        Returns:
            scores:
                Raw model image-text logits.

            probabilities:
                Human-readable normalized scores.

            prediction:
                Highest-scoring text.
        """

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        # --------------------------------------------------------
        # Raw image-text compatibility scores
        # --------------------------------------------------------

        scores = outputs.logits_per_image[0]

        # --------------------------------------------------------
        # Human-readable normalization
        # --------------------------------------------------------

        if self.model_type == "clip":

            probabilities = torch.softmax(
                scores,
                dim=0,
            )

        elif self.model_type == "siglip":

            # SigLIP uses independent pairwise logits.
            #
            # These sigmoid values are NOT a probability
            # distribution over the candidate texts.
            probabilities = torch.sigmoid(
                scores
            )

        else:

            raise ValueError(
                f"Unknown model type: {self.model_type}"
            )

        best_index = scores.argmax().item()

        return {
            "texts": texts,
            "scores": scores.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "prediction": texts[best_index],
            "prediction_index": best_index,
        }

    # ============================================================
    # RAW SCORES
    # ============================================================

    def get_image_text_scores(
        self,
        image,
        texts,
    ):
        """
        Return raw differentiable image-text logits.

        This method is intended for future adversarial attacks.

        IMPORTANT:
        No torch.no_grad() is used here.
        """

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        outputs = self.model(
            **inputs
        )

        return outputs.logits_per_image[0]

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(
        self,
        image,
        texts,
    ):
        """
        Return the highest-scoring text description.
        """

        result = self.score_image_text(
            image=image,
            texts=texts,
        )

        return {
            "prediction": result["prediction"],
            "prediction_index": result["prediction_index"],
            "scores": result["scores"],
            "probabilities": result["probabilities"],
        }