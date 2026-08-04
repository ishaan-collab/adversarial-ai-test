import base64
import io
import time

import requests
import torch
from PIL import Image


class OllamaVLMAdapter:
    """
    Black-box adapter for querying vision-language models
    served by Ollama via its REST API.

    No gradients are available — this adapter is intended
    for transfer attack evaluation only.
    """

    def __init__(
        self,
        model_name,
        host="http://127.0.0.1:11435",
        name=None,
        temperature=0.5,
        num_predict=200,
        timeout=120,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.name = name or model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.device = torch.device("cpu")

    # ============================================================
    # IMAGE UTILITIES
    # ============================================================

    @staticmethod
    def _tensor_to_base64(tensor):
        """
        Convert a [1, 3, H, W] or [3, H, W] tensor
        in [0, 1] to a base64-encoded JPEG string.
        """

        tensor = tensor.detach().cpu()

        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = (tensor * 255).byte()
        tensor = tensor.permute(1, 2, 0)

        image = Image.fromarray(tensor.numpy())

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def _pil_to_base64(pil_image):
        """
        Convert a PIL image to base64-encoded JPEG.
        """

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    # ============================================================
    # CORE API CALL
    # ============================================================

    def _chat(self, prompt, image_b64=None, retries=3):
        """
        Send a chat request to the Ollama API and
        return the raw text response.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if image_b64 is not None:
            messages[0]["images"] = [image_b64]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": 0.9,
                "top_k": 40,
            },
        }

        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

        return ""

    # ============================================================
    # PUBLIC QUERY METHODS
    # ============================================================

    def ask_image(self, image, question):
        """
        Ask a free-form question about an image.

        Args:
            image: PIL Image or [1,3,H,W] tensor in [0,1].
            question: str

        Returns:
            str: model's text response.
        """

        if isinstance(image, Image.Image):
            image_b64 = self._pil_to_base64(image)
        elif isinstance(image, torch.Tensor):
            image_b64 = self._tensor_to_base64(image)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        return self._chat(question, image_b64=image_b64)

    def classify_image(self, image, source_text, target_text):
        """
        Classify the image by asking for a description and
        checking for source/target keywords.

        Moondream doesn't respond well to direct A/B or yes/no
        questions, so we use an open-ended describe prompt and
        do keyword matching on the response.

        Returns:
            dict with prediction, raw_response, and
            whether the target was selected.
        """

        source_keyword = source_text.replace(
            "a photo of ", ""
        ).strip()
        target_keyword = target_text.replace(
            "a photo of ", ""
        ).strip()

        prompt = "What do you see in this image?"

        raw_response = self.ask_image(image, prompt)
        response_lower = raw_response.lower()

        has_source = source_keyword.lower() in response_lower
        has_target = target_keyword.lower() in response_lower

        if has_target and not has_source:
            prediction = target_text
            prediction_label = "B"
            target_selected = True
        elif has_source and not has_target:
            prediction = source_text
            prediction_label = "A"
            target_selected = False
        elif has_source and has_target:
            prediction = "both"
            prediction_label = "?"
            target_selected = False
        else:
            prediction = "neither"
            prediction_label = "?"
            target_selected = False

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "target_selected": target_selected,
            "raw_response": raw_response,
            "has_source": has_source,
            "has_target": has_target,
            "method": "keyword",
        }

    def describe_image(self, image):
        """
        Ask the model to describe the image.
        Uses an open-ended prompt that moondream responds to reliably.
        """

        return self.ask_image(
            image,
            "What do you see in this image?",
        )
