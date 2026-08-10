"""
White-box adapter for LLaVA-1.5 VLMs.

Supports LLaVA-1.5-7B, LLaVA-1.5-13B, and LLaVA-Next variants.
Uses HuggingFace transformers' LlavaForConditionalGeneration.

Architecture:
    Vision encoder: CLIP-ViT-L/14 (336×336, 1024 dim, 24 layers)
    Connector: 2-layer MLP projectors (1024 → LLM dim)
    LLM: LlamaForCausalLM (7B or 13B)

The adapter provides differentiable forward passes for:
    - Vision features (pre-connector)
    - Connector tokens (post-projector)
    - LLM logits (full pipeline)
    - Teacher-forced CE loss (for targeted attacks)
"""

import io
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    AutoProcessor,
)
from models.base import WhiteBoxVLMAdapter
from models.vram_utils import load_hf_model, print_vram_status


LLAVA_MODELS = {
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "image_size": 336,
        "params_b": 7.0,
    },
    "llava-1.5-13b": {
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "image_size": 336,
        "params_b": 13.0,
    },
    "llava-next-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "image_size": 336,
        "params_b": 7.0,
    },
}


class LlavaAdapter(WhiteBoxVLMAdapter):
    """White-box adapter for LLaVA-1.5 VLMs."""

    def __init__(self, model_name: str = "llava-1.5-7b",
                 device: str = "cuda",
                 dtype: Optional[torch.dtype] = None):
        config = LLAVA_MODELS[model_name]
        model_id = config["model_id"]
        image_size = config["image_size"]

        if dtype is None:
            dtype = torch.bfloat16

        print(f"\n{'='*60}")
        print(f"Loading LLaVA: {model_name} ({config['params_b']:.1f}B params)")
        print(f"{'='*60}")

        model = load_hf_model(
            LlavaForConditionalGeneration, model_id, dtype=dtype,
        )
        processor = AutoProcessor.from_pretrained(model_id)

        super().__init__(
            name=model_name,
            model=model,
            processor=processor,
            device=device,
            image_size=image_size,
            dtype=dtype,
        )

        self.image_mean = torch.tensor(
            processor.image_processor.image_mean,
            device=device, dtype=dtype,
        )
        self.image_std = torch.tensor(
            processor.image_processor.image_std,
            device=device, dtype=dtype,
        )

        self.vision_tower = model.model.vision_tower
        self.multi_modal_projector = model.model.multi_modal_projector
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = model.config
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Differentiable preprocessing: PIL → [1,3,336,336] normalized.
        Preserves gradient flow for adversarial optimization.
        """
        img = pil_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(self.device, dtype=self.dtype)
        t = (t - self.image_mean.view(1, 3, 1, 1)) / self.image_std.view(1, 3, 1, 1)
        return t

    def _preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply differentiable preprocessing to a [1,3,H,W] tensor in [0,1]."""
        if image_tensor.shape[-1] != self.image_size or image_tensor.shape[-2] != self.image_size:
            image_tensor = F.interpolate(
                image_tensor, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        t = image_tensor.to(self.dtype)
        t = (t - self.image_mean.view(1, 3, 1, 1)) / self.image_std.view(1, 3, 1, 1)
        return t

    def get_vision_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Vision encoder output (before connector). Differentiable."""
        pixel_values = self._preprocess_tensor(image_tensor)
        with torch.set_grad_enabled(True):
            features = self.vision_tower(pixel_values, output_hidden_states=True)
            if hasattr(features, 'last_hidden_state'):
                return features.last_hidden_state
            return features[0]

    def get_connector_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Connector output (after projection). Differentiable."""
        vision_features = self.get_vision_features(image_tensor)
        selected = vision_features[:, 1:]
        return self.multi_modal_projector(selected)

    def get_llm_logits(self, image_tensor: torch.Tensor,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """Full pipeline LLM logits. Differentiable."""
        pixel_values = self._preprocess_tensor(image_tensor)

        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs.logits

    def _build_prompt_ids(self, question: str = "What do you see in this image?") -> torch.Tensor:
        """Build prompt token IDs with <image> placeholder expanded to 576 tokens."""
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        pil_dummy = Image.new("RGB", (self.image_size, self.image_size))
        inputs = self.processor(text=prompt, images=pil_dummy, return_tensors="pt")
        return inputs.input_ids.to(self.device)

    def compute_loss(self, image_tensor: torch.Tensor,
                     target_ids: torch.Tensor,
                     prompt_ids: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced CE loss.

        Constructs input as: [prompt_ids (with <image>), target_ids]
        Computes loss only on target token positions.
        """
        device = self.device
        full_ids = torch.cat([prompt_ids, target_ids], dim=1).to(device)
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        attention_mask = torch.ones_like(full_ids)
        pixel_values = self._preprocess_tensor(image_tensor)

        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        return outputs.loss

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, return [1, L] token IDs."""
        ids = self.processor.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def tokenize_prompt(self, question: str = "What do you see in this image?") -> torch.Tensor:
        """Tokenize a prompt with <image> placeholder for the model."""
        return self._build_prompt_ids(question)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.processor.tokenizer.decode(
            token_ids.squeeze(0), skip_special_tokens=True
        )

    def describe(self, pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 100) -> str:
        """Generate a description (non-differentiable)."""
        img = pil_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(
            text=prompt, images=img, return_tensors="pt"
        ).to(self.device, self.dtype)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, temperature=1.0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        return self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

    def classify(self, pil_image: Image.Image,
                 source_text: str, target_text: str) -> Dict:
        """A/B classification based on keywords."""
        desc = self.describe(pil_image)
        desc_lower = desc.lower()

        has_source = source_text.lower() in desc_lower
        has_target = target_text.lower() in desc_lower

        return {
            "description": desc,
            "source_match": has_source,
            "target_match": has_target,
            "classification": "target" if has_target and not has_source
                            else ("source" if has_source else "neither"),
        }
