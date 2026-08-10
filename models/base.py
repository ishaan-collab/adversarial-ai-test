"""
Abstract base classes for VLM adapters.

Provides a unified interface so attacks can work with any VLM,
regardless of architecture or access mode (white-box / black-box).

Hierarchy:
    BaseVLMAdapter
    ├── WhiteBoxVLMAdapter  (gradient access, for PGD/BIM/FGSM)
    └── BlackBoxVLMAdapter  (API-only, for SPSA/Square/Genetic)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import torch


class BaseVLMAdapter(ABC):
    """Common interface for all VLM adapters."""

    def __init__(self, name: str, device: str = "cuda",
                 image_size: int = 336):
        self.name = name
        self.device = device
        self.image_size = image_size

    @abstractmethod
    def describe(self, pil_image: Image.Image,
                 question: str = "What do you see in this image?",
                 max_tokens: int = 100) -> str:
        """Generate a description of the image (non-differentiable)."""
        ...

    @abstractmethod
    def classify(self, pil_image: Image.Image,
                 source_text: str, target_text: str) -> Dict[str, Any]:
        """A/B classification: does the output match source or target?"""
        ...

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert [1,3,H,W] or [3,H,W] tensor in [0,1] to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        arr = (tensor.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
        return Image.fromarray(arr)

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to [1,3,H,W] tensor in [0,1]."""
        import numpy as np
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)


class WhiteBoxVLMAdapter(BaseVLMAdapter):
    """
    Adapter with full gradient access through the model.

    Used for white-box attacks: PGD, BIM, FGSM, DeepFool.
    Subclasses must implement differentiable forward passes.
    """

    def __init__(self, name: str, model, processor,
                 device: str = "cuda", image_size: int = 336,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(name, device, image_size)
        self.model = model
        self.processor = processor
        self.dtype = dtype

    @abstractmethod
    def get_vision_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Vision encoder output (before connector). Differentiable."""
        ...

    @abstractmethod
    def get_connector_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Connector output (after projection). Differentiable."""
        ...

    @abstractmethod
    def get_llm_logits(self, image_tensor: torch.Tensor,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """Full pipeline LLM logits. Differentiable."""
        ...

    @abstractmethod
    def compute_loss(self, image_tensor: torch.Tensor,
                     target_ids: torch.Tensor,
                     prompt_ids: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced cross-entropy loss.
        Differentiable w.r.t. image_tensor.

        Args:
            image_tensor: [1,3,H,W] in [0,1], requires_grad=True
            target_ids: [1, L_target] token IDs of desired output
            prompt_ids: [1, L_prompt] token IDs of prompt text
        Returns:
            scalar loss tensor
        """
        ...

    @abstractmethod
    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Differentiable preprocessing: PIL → [1,3,H,W] normalized tensor.
        Must preserve gradient flow for adversarial optimization.
        """
        ...

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, return [1, L] token IDs on device."""
        ...

    @abstractmethod
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        ...

    def get_all_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience: return vision, connector, and logits in one pass."""
        return {
            "vision": self.get_vision_features(image_tensor),
            "connector": self.get_connector_tokens(image_tensor),
        }


class BlackBoxVLMAdapter(BaseVLMAdapter):
    """
    Adapter for API-only access (no gradients).

    Used for black-box attacks: SPSA, Square Attack, Genetic.
    Subclasses implement API-specific query methods.
    """

    def __init__(self, name: str, host: str, model_name: str,
                 device: str = "cpu", image_size: int = 378,
                 api_type: str = "openai"):
        super().__init__(name, device, image_size)
        self.host = host
        self.model_name = model_name
        self.api_type = api_type
        self.query_count = 0

    @abstractmethod
    def query(self, pil_image: Image.Image,
              question: str = "What do you see in this image?",
              max_tokens: int = 50,
              temperature: float = 0.1) -> str:
        """Send image + question to the API, return text response."""
        ...

    @abstractmethod
    def score(self, pil_image: Image.Image,
              keywords: List[str],
              target_keyword: Optional[str] = None,
              max_tokens: int = 50) -> float:
        """
        Score an image: higher = better for attacker.
        Default: character-position of first keyword (lower = worse).
        Override for custom scoring.
        """
        ...

    def reset_query_count(self):
        self.query_count = 0
