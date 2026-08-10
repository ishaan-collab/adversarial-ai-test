"""
Universal white-box PGD attack for VLMs.

Works with any WhiteBoxVLMAdapter (LLaVA, Qwen2-VL, moondream, etc.).
Supports:
    - Targeted / untargeted attacks
    - Multi-level loss (vision + connector + language)
    - MI-FGSM momentum
    - Random restart
    - Teacher-forced CE loss

Usage:
    PYTHONPATH=. python attacks/vlm_pgd_universal.py \
        --model llava-1.5-7b \
        --image data/vlm/dog03.jpg \
        --target "A cat sitting on a couch" \
        --epsilon 8/255 --iterations 300
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List
import time
import sys
import os

from models.base import WhiteBoxVLMAdapter


class UniversalPGDAttack:
    """
    Model-agnostic PGD attack for VLMs.

    Uses the WhiteBoxVLMAdapter interface to compute
    differentiable loss and gradients.
    """

    def __init__(self, adapter: WhiteBoxVLMAdapter,
                 epsilon: float = 8 / 255,
                 alpha: float = 2 / 255,
                 iterations: int = 300,
                 momentum: float = 0.0,
                 random_start: bool = True,
                 seed: int = 42):
        self.adapter = adapter
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.momentum = momentum
        self.random_start = random_start
        self.seed = seed
        self.device = adapter.device

    def attack(self, image_tensor: torch.Tensor,
               target_text: str,
               prompt: str = "What do you see in this image?",
               verbose: bool = True) -> torch.Tensor:
        """
        Run targeted PGD attack.

        Args:
            image_tensor: [1, 3, H, W] clean image in [0, 1]
            target_text: desired output text
            prompt: question/prompt for the model
        Returns:
            adversarial image tensor [1, 3, H, W] in [0, 1]
        """
        torch.manual_seed(self.seed)

        image_tensor = image_tensor.to(self.device)
        clean = image_tensor.clone()
        lower = torch.clamp(clean - self.epsilon, 0, 1)
        upper = torch.clamp(clean + self.epsilon, 0, 1)

        target_ids = self.adapter.tokenize(target_text)
        if hasattr(self.adapter, 'tokenize_prompt'):
            prompt_ids = self.adapter.tokenize_prompt(prompt)
        else:
            prompt_ids = self.adapter.tokenize(prompt)

        if self.random_start:
            noise = torch.empty_like(image_tensor).uniform_(-self.epsilon, self.epsilon)
            adv = torch.clamp(image_tensor + noise, 0, 1)
        else:
            adv = image_tensor.clone()

        adv.requires_grad_(True)
        velocity = torch.zeros_like(adv)

        if verbose:
            print(f"  [PGD] target: \"{target_text}\"")
            print(f"  [PGD] eps={self.epsilon:.6f} alpha={self.alpha:.6f} "
                  f"iters={self.iterations} momentum={self.momentum}")
            print(f"  [PGD] target tokens: {target_ids.shape[1]}")
            print(f"  [PGD] prompt tokens: {prompt_ids.shape[1]}")

        for i in range(self.iterations):
            try:
                loss = self.adapter.compute_loss(
                    adv, target_ids, prompt_ids,
                )
            except Exception as e:
                if verbose:
                    print(f"  [PGD] iter {i}: loss computation failed: {e}")
                break

            if loss is None or torch.isnan(loss):
                if verbose:
                    print(f"  [PGD] iter {i}: loss is NaN, stopping")
                break

            grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]

            if self.momentum > 0:
                velocity = self.momentum * velocity + grad / (grad.norm(p=1) + 1e-8)
                update = torch.sign(velocity)
            else:
                update = torch.sign(grad)

            adv = adv.detach() - self.alpha * update
            adv = torch.clamp(adv, lower, upper)
            adv = adv.detach().requires_grad_(True)

            if verbose and (i + 1) % 50 == 0:
                linf = (adv.detach() - clean).abs().max().item()
                print(f"  [PGD] iter {i+1}/{self.iterations}: "
                      f"loss={loss.item():.4f} linf={linf:.6f}")

        adv = adv.detach()
        linf = (adv - clean).abs().max().item()

        if verbose:
            print(f"  [PGD] done. L-inf={linf:.6f} (budget={self.epsilon:.6f})")

        return adv

    def attack_pil(self, pil_image: Image.Image,
                   target_text: str,
                   prompt: str = "What do you see in this image?",
                   verbose: bool = True) -> Image.Image:
        """Convenience: attack from PIL, return PIL."""
        image_tensor = self.adapter.pil_to_tensor(pil_image)
        adv_tensor = self.attack(image_tensor, target_text, prompt, verbose)
        return self.adapter.tensor_to_pil(adv_tensor)


class MultiLevelPGDAttack(UniversalPGDAttack):
    """
    Multi-level PGD attack with vision + connector + language losses.

    Combines three complementary loss surfaces:
    - Level 1: L_vision  — MSE between adversarial and clean vision features
    - Level 2: L_align   — MSE between adversarial and clean connector tokens
    - Level 3: L_lang    — teacher-forced cross-entropy on target tokens

    Total loss (minimised by PGD):
        L = lambda_language * L_lang
            - lambda_vision    * L_vision
            - lambda_alignment * L_alignment

    Uses MI-FGSM momentum (default 0.9) for better transferability.
    """

    def __init__(self, adapter: WhiteBoxVLMAdapter,
                 lambda_vision: float = 0.0,
                 lambda_alignment: float = 0.0,
                 lambda_language: float = 1.0,
                 **kwargs):
        super().__init__(adapter=adapter, **kwargs)
        self.lambda_vision = lambda_vision
        self.lambda_alignment = lambda_alignment
        self.lambda_language = lambda_language

    def attack(self, image_tensor: torch.Tensor,
               target_text: str,
               prompt: str = "What do you see in this image?",
               verbose: bool = True,
               return_details: bool = False):
        """
        Run multi-level PGD attack.

        Args:
            image_tensor: [1, 3, H, W] clean image in [0, 1]
            target_text: desired output text
            prompt: question/prompt for the model
            verbose: print progress
            return_details: if True, return (adv_tensor, details_dict)

        Returns:
            adversarial tensor [1, 3, H, W] in [0, 1],
            optionally with a details dict.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        image_tensor = image_tensor.to(self.device)
        clean = image_tensor.clone()
        lower = torch.clamp(clean - self.epsilon, 0, 1)
        upper = torch.clamp(clean + self.epsilon, 0, 1)

        target_ids = self.adapter.tokenize(target_text)
        if hasattr(self.adapter, 'tokenize_prompt'):
            prompt_ids = self.adapter.tokenize_prompt(prompt)
        else:
            prompt_ids = self.adapter.tokenize(prompt)

        # --- Clean reference features (no gradient) ---
        clean_vision = None
        clean_connector = None

        with torch.no_grad():
            if self.lambda_vision > 0:
                clean_vision = self.adapter.get_vision_features(
                    clean.to(self.adapter.dtype)
                ).detach().float()
            if self.lambda_alignment > 0:
                clean_connector = self.adapter.get_connector_tokens(
                    clean.to(self.adapter.dtype)
                ).detach().float()

        if self.random_start:
            noise = torch.empty_like(image_tensor).uniform_(
                -self.epsilon, self.epsilon
            )
            adv = torch.clamp(image_tensor + noise, 0, 1)
        else:
            adv = image_tensor.clone()

        velocity = torch.zeros_like(adv)
        history = []

        if verbose:
            print(f"  [ML-PGD] levels: vision={self.lambda_vision} "
                  f"align={self.lambda_alignment} lang={self.lambda_language}")
            print(f"  [ML-PGD] eps={self.epsilon:.6f} alpha={self.alpha:.6f} "
                  f"iters={self.iterations} momentum={self.momentum}")
            print(f"  [ML-PGD] target tokens: {target_ids.shape[1]}")
            print(f"  [ML-PGD] prompt tokens: {prompt_ids.shape[1]}")

        for i in range(self.iterations):
            adv = adv.detach().requires_grad_(True)

            total_loss = torch.tensor(
                0.0, device=self.device, dtype=torch.float32
            )
            l_vision_val = 0.0
            l_align_val = 0.0
            l_lang_val = 0.0

            # --- Level 3: Language CE loss ---
            if self.lambda_language > 0:
                lang_loss = self.adapter.compute_loss(
                    adv, target_ids, prompt_ids
                )
                if lang_loss is not None and not torch.isnan(lang_loss):
                    l_lang_val = lang_loss.item()
                    total_loss = total_loss + self.lambda_language * lang_loss.float()

            # --- Level 1: Vision feature disruption ---
            if self.lambda_vision > 0 and clean_vision is not None:
                adv_vision = self.adapter.get_vision_features(adv)
                l_vision = F.mse_loss(
                    adv_vision.float(), clean_vision
                )
                l_vision_val = l_vision.item()
                total_loss = total_loss - self.lambda_vision * l_vision

            # --- Level 2: Connector token disruption ---
            if self.lambda_alignment > 0 and clean_connector is not None:
                adv_connector = self.adapter.get_connector_tokens(adv)
                l_alignment = F.mse_loss(
                    adv_connector.float(), clean_connector
                )
                l_align_val = l_alignment.item()
                total_loss = total_loss - self.lambda_alignment * l_alignment

            if torch.isnan(total_loss):
                if verbose:
                    print(f"  [ML-PGD] iter {i}: loss is NaN, stopping")
                break

            grad = torch.autograd.grad(
                total_loss, adv, retain_graph=False
            )[0]

            # MI-FGSM momentum
            if self.momentum > 0:
                grad_l1 = grad.abs().mean()
                if grad_l1 > 0:
                    grad = grad / grad_l1
                velocity = self.momentum * velocity + grad
                update = velocity.sign()
                if update.abs().sum() == 0:
                    update = grad.sign()
            else:
                update = torch.sign(grad)

            with torch.no_grad():
                adv = adv - self.alpha * update
                adv = torch.max(torch.min(adv, upper), lower)
                adv = adv.clamp(0, 1)

            should_log = (
                verbose and (
                    i == 0
                    or (i + 1) % 25 == 0
                    or i == self.iterations - 1
                )
            )

            if should_log:
                linf = (adv.detach() - clean).abs().max().item()
                print(
                    f"  [ML-PGD] iter {i+1:3d}/{self.iterations} | "
                    f"loss={total_loss.item():.4f} "
                    f"L_vis={l_vision_val:.4f} "
                    f"L_align={l_align_val:.4f} "
                    f"L_lang={l_lang_val:.4f} | "
                    f"linf={linf:.6f}"
                )

            if (i + 1) % 10 == 0 or i == self.iterations - 1:
                history.append({
                    "iteration": i + 1,
                    "loss": total_loss.item(),
                    "l_vision": l_vision_val,
                    "l_alignment": l_align_val,
                    "l_language": l_lang_val,
                })

        adv = adv.detach()
        with torch.no_grad():
            adv = torch.max(torch.min(adv, upper), lower)
            adv = adv.clamp(0, 1)

        perturbation = adv - clean
        linf = perturbation.abs().max().item()
        l2 = torch.norm(
            perturbation.reshape(perturbation.shape[0], -1), p=2, dim=1
        ).item()
        mean_abs = perturbation.abs().mean().item()

        if verbose:
            print(f"  [ML-PGD] done. L-inf={linf:.6f} "
                  f"(budget={self.epsilon:.6f}) "
                  f"L2={l2:.4f} mean={mean_abs:.6f}")

        if not return_details:
            return adv

        details = {
            "epsilon": float(self.epsilon),
            "alpha": float(self.alpha),
            "iterations": int(self.iterations),
            "seed": int(self.seed),
            "random_start": bool(self.random_start),
            "momentum": float(self.momentum),
            "target_text": target_text,
            "lambda_vision": float(self.lambda_vision),
            "lambda_alignment": float(self.lambda_alignment),
            "lambda_language": float(self.lambda_language),
            "linf": linf,
            "l2": l2,
            "mean_abs": mean_abs,
            "history": history,
        }
        return adv, details


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal VLM PGD Attack"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (llava-1.5-7b, qwen2-vl-7b, etc.)")
    parser.add_argument("--image", default="data/vlm/dog03.jpg")
    parser.add_argument("--target", default="A cat sitting on a couch")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/adv_universal.png")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    from models.vlm_registry import get_vlm

    print(f"Loading model: {args.model}")
    adapter = get_vlm(args.model, mode="whitebox")
    print(f"Model loaded: {adapter.name}")

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((adapter.image_size, adapter.image_size), Image.LANCZOS)
    print(f"Image: {args.image} ({adapter.image_size}x{adapter.image_size})")

    print(f"\nClean description:")
    clean_desc = adapter.describe(pil)
    print(f"  {clean_desc[:200]}")

    attack = UniversalPGDAttack(
        adapter=adapter,
        epsilon=args.epsilon,
        alpha=args.alpha,
        iterations=args.iterations,
        momentum=args.momentum,
        seed=args.seed,
    )

    print(f"\nRunning PGD attack...")
    t0 = time.time()
    adv_pil = attack.attack_pil(pil, args.target)
    elapsed = time.time() - t0
    print(f"Attack completed in {elapsed:.1f}s")

    print(f"\nAdversarial description:")
    adv_desc = adapter.describe(adv_pil)
    print(f"  {adv_desc[:200]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")
