"""
White-box adapter for Qwen2-VL VLMs.

Supports Qwen2-VL-2B, Qwen2-VL-7B, Qwen2.5-VL-3B, Qwen2.5-VL-7B.
Uses HuggingFace transformers' Qwen2VLForConditionalGeneration.

Architecture:
    Vision encoder: ViT (dynamic resolution, 1280 dim, 32 blocks)
    Connector: PatchMerger (4 patches -> 1 token, 1280*4 -> LLM dim)
    LLM: Qwen2Model (2B-7B)

Differentiable pipeline:
    [1,3,H,W] in [0,1] -> resize -> normalize -> patch extraction -> [1,1024,1176]
    -> vision encoder -> [1024,1280] (pre-merge) / [256,3584] (post-merge)
    -> LLM -> logits -> CE loss
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from models.base import WhiteBoxVLMAdapter
from models.vram_utils import load_hf_model


QWEN_VL_MODELS = {
    "qwen2-vl-2b": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "image_size": 448,
        "params_b": 2.0,
    },
    "qwen2-vl-7b": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "image_size": 448,
        "params_b": 7.0,
    },
    "qwen2.5-vl-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "image_size": 448,
        "params_b": 3.0,
    },
    "qwen2.5-vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "image_size": 448,
        "params_b": 7.0,
    },
}


class QwenVLAdapter(WhiteBoxVLMAdapter):
    """White-box adapter for Qwen2-VL VLMs."""

    def __init__(self, model_name: str = "qwen2-vl-2b",
                 device: str = "cuda",
                 dtype: Optional[torch.dtype] = None):
        config = QWEN_VL_MODELS[model_name]
        model_id = config["model_id"]
        image_size = config["image_size"]

        if dtype is None:
            dtype = torch.bfloat16

        print(f"\n{'='*60}")
        print(f"Loading Qwen2-VL: {model_name} ({config['params_b']:.1f}B params)")
        print(f"{'='*60}")

        model = load_hf_model(
            Qwen2VLForConditionalGeneration, model_id, dtype=dtype,
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
            [0.48145466, 0.4578275, 0.40821073],
            device=device, dtype=dtype,
        )
        self.image_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device, dtype=dtype,
        )

        self.patch_size = 14
        self.temporal_patch_size = 2
        self.merge_size = 2
        self.grid_h = image_size // self.patch_size
        self.grid_w = image_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.num_image_tokens = self.num_patches // (self.merge_size ** 2)

        self._grid_thw = torch.tensor(
            [[1, self.grid_h, self.grid_w]],
            device=device, dtype=torch.long,
        )

        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def _preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Differentiable preprocessing: [1,3,H,W] in [0,1] -> [1, 1024, 1176] patches.

        Matches Qwen2VLImageProcessor._preprocess():
        1. Resize to image_size x image_size
        2. Normalize with CLIP mean/std
        3. Reshape -> permute -> expand(temporal) -> flatten to patch format
        """
        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size
        gh = self.grid_h
        gw = self.grid_w

        if image_tensor.shape[-1] != self.image_size or image_tensor.shape[-2] != self.image_size:
            image_tensor = F.interpolate(
                image_tensor, size=(self.image_size, self.image_size),
                mode="bicubic", align_corners=False, antialias=True,
            )

        t = image_tensor.to(self.dtype)
        t = (t - self.image_mean.view(1, 3, 1, 1)) / self.image_std.view(1, 3, 1, 1)

        B, C, H, W = t.shape

        patches = t.reshape(
            B, C,
            gh // ms, ms, ps,
            gw // ms, ms, ps,
        )

        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()

        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, tps, -1, -1)
            .reshape(B, gh * gw, C * tps * ps * ps)
        )

        return flatten_patches

    def get_vision_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Vision encoder output (before merger). Differentiable.

        Returns: [num_patches, vision_dim] = [1024, 1280]
        """
        pixel_values = self._preprocess_tensor(image_tensor)
        with torch.set_grad_enabled(True):
            outputs = self.model.get_image_features(
                pixel_values, image_grid_thw=self._grid_thw,
            )
        return outputs.last_hidden_state

    def get_connector_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Connector output (after merger). Differentiable.

        Returns: [num_image_tokens, llm_dim] = [256, 3584]
        """
        pixel_values = self._preprocess_tensor(image_tensor)
        with torch.set_grad_enabled(True):
            outputs = self.model.get_image_features(
                pixel_values, image_grid_thw=self._grid_thw,
            )
        pooler = outputs.pooler_output
        if isinstance(pooler, (tuple, list)):
            return pooler[0]
        return pooler

    def _make_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate mm_token_type_ids: 0=text, 1=image, 2=video."""
        mm = torch.zeros_like(input_ids, dtype=torch.long)
        mm[input_ids == self.image_token_id] = 1
        return mm

    def get_llm_logits(self, image_tensor: torch.Tensor,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """Full pipeline LLM logits. Differentiable."""
        pixel_values = self._preprocess_tensor(image_tensor)
        mm_token_type_ids = self._make_mm_token_type_ids(input_ids)

        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=self._grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
            )
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs[1]

    def compute_loss(self, image_tensor: torch.Tensor,
                     target_ids: torch.Tensor,
                     prompt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced CE loss.

        prompt_ids already contains expanded <|image_pad|> tokens (from tokenize_prompt).
        """
        device = self.device

        full_ids = torch.cat([prompt_ids, target_ids], dim=1).to(device)
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        attention_mask = torch.ones_like(full_ids)
        mm_token_type_ids = self._make_mm_token_type_ids(full_ids)

        pixel_values = self._preprocess_tensor(image_tensor)

        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=pixel_values,
                image_grid_thw=self._grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                labels=labels,
                use_cache=False,
            )

        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, return [1, L] token IDs."""
        ids = self.processor.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def tokenize_prompt(self, question: str = "What do you see in this image?") -> torch.Tensor:
        """Build prompt token IDs with expanded <|image_pad|> tokens.

        Uses the processor with a dummy image to get the correct number
        of image pad tokens (256 for 448x448).
        """
        img = Image.new("RGB", (self.image_size, self.image_size))
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt",
        )
        return inputs.input_ids.to(self.device)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.processor.tokenizer.decode(
            token_ids.squeeze(0), skip_special_tokens=True,
        )

    def describe(self, pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 100) -> str:
        """Generate a description (non-differentiable)."""
        img = pil_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS,
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt",
        ).to(self.device, self.dtype)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, temperature=1.0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        return self.processor.tokenizer.decode(
            generated, skip_special_tokens=True,
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

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL -> [1, 3, H, W] in [0, 1] (differentiable-ready)."""
        import numpy as np
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)
