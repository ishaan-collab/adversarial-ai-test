"""
Universal black-box attack for VLMs.

Works with any BlackBoxVLMAdapter (API-based, Ollama, llama-server, etc.).
Uses SPSA + random search + square refinement.

Usage:
    PYTHONPATH=. python attacks/blackbox_universal.py \
        --host http://127.0.0.1:11471 --model-name vyas \
        --image data/vlm/dog03.jpg \
        --epsilon 8/255 --queries 300
"""

import io
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, List
import sys
import os

from models.base import BlackBoxVLMAdapter
from models.api_adapter import DEFAULT_KEYWORDS, SUCCESS_SCORE


def upscale_perturbation(low_pert, target_h, target_w):
    """Bilinearly upscale [D, D, 3] perturbation to [H, W, 3]."""
    t = torch.from_numpy(low_pert).permute(2, 0, 1).unsqueeze(0)
    t_up = F.interpolate(
        t, size=(target_h, target_w),
        mode="bilinear", align_corners=False, antialias=True,
    )
    return t_up.squeeze(0).permute(1, 2, 0).numpy()


def apply_perturbation(clean_arr, low_pert, epsilon):
    """Apply upscaled perturbation, clipped to epsilon ball."""
    H, W, C = clean_arr.shape
    pert_up = upscale_perturbation(low_pert, H, W)
    pert_up = np.clip(pert_up, -epsilon, epsilon)
    result = clean_arr + pert_up
    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)
    return np.clip(result, lower, upper)


def pil_from_array(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


class UniversalBlackBoxAttack:
    """
    Multi-phase black-box attack on any VLM.

    Phase 1: Random search (exploration)
    Phase 2: SPSA gradient estimation with accumulation
    Phase 3: Square refinement (fine-tuning)
    """

    def __init__(self, adapter: BlackBoxVLMAdapter,
                 epsilon: float = 8 / 255,
                 queries: int = 1000,
                 low_dim: int = 32,
                 seed: int = 42,
                 max_tokens: int = 50,
                 keywords: Optional[List[str]] = None,
                 target_keyword: Optional[str] = "cat"):
        self.adapter = adapter
        self.epsilon = epsilon
        self.queries = queries
        self.low_dim = low_dim
        self.seed = seed
        self.max_tokens = max_tokens
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.target_keyword = target_keyword
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def attack(self, pil_image: Image.Image,
               verbose: bool = True,
               early_stop: bool = True) -> Image.Image:
        """
        Run black-box attack on a PIL image.
        Returns adversarial PIL image.
        """
        self.adapter.reset_query_count()
        self.adapter.query_count = 0

        clean_arr = np.array(pil_image, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape
        D = self.low_dim

        # Baseline
        score = self.adapter.score(
            pil_from_array(clean_arr),
            keywords=self.keywords,
            target_keyword=self.target_keyword,
            max_tokens=self.max_tokens,
        )
        best_score = score
        best_text = self.adapter.describe(
            pil_from_array(clean_arr), max_tokens=self.max_tokens
        )
        best_low_pert = np.zeros((D, D, C), dtype=np.float32)

        if verbose:
            print(f"  [init] score={best_score:.1f} | {best_text[:100]}")

        if early_stop and best_score >= SUCCESS_SCORE:
            return self._finish(clean_arr, best_low_pert, best_score, best_text, verbose)

        # === Phase 1: Random search ===
        phase1_end = min(80, max(10, self.queries // 8))
        if verbose:
            print(f"\n  Phase 1: Random search ({phase1_end} queries)")

        for q in range(phase1_end):
            if self.adapter.query_count >= self.queries:
                break

            low_pert = self.np_rng.randn(D, D, C).astype(np.float32) * self.epsilon * 0.5
            candidate = apply_perturbation(clean_arr, low_pert, self.epsilon)
            pil = pil_from_array(candidate)

            score = self.adapter.score(
                pil, keywords=self.keywords,
                target_keyword=self.target_keyword,
                max_tokens=self.max_tokens,
            )

            if score > best_score:
                best_score = score
                best_text = self.adapter.describe(pil, max_tokens=self.max_tokens)
                best_low_pert = low_pert.copy()
                if verbose:
                    print(f"  [{self.adapter.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} | {best_text[:80]} *** IMPROVED ***")

            if early_stop and best_score >= SUCCESS_SCORE:
                return self._finish(clean_arr, best_low_pert, best_score, best_text, verbose)

        # === Phase 2: SPSA ===
        phase2_end = max(phase1_end, min(int(self.queries * 0.85), self.queries - 50))
        if verbose:
            print(f"\n  Phase 2: SPSA ({phase2_end - phase1_end} iters)")

        alpha = self.epsilon * 0.5
        c_spsa = self.epsilon * 0.1
        eval_interval = 15

        current_pert = best_low_pert.copy()
        spsa_iters = 0

        for q in range(phase1_end, phase2_end):
            if self.adapter.query_count + 2 > self.queries:
                break

            delta = self.np_rng.choice([-1, 1], size=(D, D, C)).astype(np.float32)

            plus_pert = current_pert + c_spsa * delta
            plus_img = apply_perturbation(clean_arr, plus_pert, self.epsilon)
            pil_plus = pil_from_array(plus_img)
            score_plus = self.adapter.score(
                pil_plus, keywords=self.keywords,
                target_keyword=self.target_keyword,
                max_tokens=self.max_tokens,
            )

            minus_pert = current_pert - c_spsa * delta
            minus_img = apply_perturbation(clean_arr, minus_pert, self.epsilon)
            pil_minus = pil_from_array(minus_img)
            score_minus = self.adapter.score(
                pil_minus, keywords=self.keywords,
                target_keyword=self.target_keyword,
                max_tokens=self.max_tokens,
            )

            spsa_iters += 1
            grad_est = (score_plus - score_minus) / (2 * c_spsa) * delta
            current_pert = current_pert + alpha * np.sign(grad_est)

            if spsa_iters % eval_interval == 0 or q == phase2_end - 1:
                candidate = apply_perturbation(clean_arr, current_pert, self.epsilon)
                pil = pil_from_array(candidate)
                score = self.adapter.score(
                    pil, keywords=self.keywords,
                    target_keyword=self.target_keyword,
                    max_tokens=self.max_tokens,
                )

                if score > best_score:
                    best_score = score
                    best_text = self.adapter.describe(pil, max_tokens=self.max_tokens)
                    best_low_pert = current_pert.copy()
                    if verbose:
                        print(f"  [{self.adapter.query_count:5d}] score={score:.1f} "
                              f"best={best_score:.1f} | {best_text[:80]} *** IMPROVED ***")
                else:
                    current_pert = best_low_pert.copy()
                    alpha *= 0.95

                if early_stop and best_score >= SUCCESS_SCORE:
                    return self._finish(clean_arr, best_low_pert, best_score, best_text, verbose)

            if (q + 1) % 100 == 0 and verbose:
                print(f"  [{self.adapter.query_count:5d}] best={best_score:.1f} "
                      f"alpha={alpha:.6f} spsa_iters={spsa_iters}")

        # === Phase 3: Square refinement ===
        remaining = self.queries - self.adapter.query_count
        if verbose:
            print(f"\n  Phase 3: Square refinement ({remaining} queries)")

        no_improve = 0

        while self.adapter.query_count < self.queries:
            p = self.rng.randint(2, max(3, D // 2))
            y0 = self.rng.randint(0, D - p)
            x0 = self.rng.randint(0, D - p)

            new_pert = best_low_pert.copy()
            noise = self.np_rng.randn(p, p, C).astype(np.float32) * self.epsilon * 0.3
            new_pert[y0:y0 + p, x0:x0 + p] += noise

            candidate = apply_perturbation(clean_arr, new_pert, self.epsilon)
            pil = pil_from_array(candidate)
            score = self.adapter.score(
                pil, keywords=self.keywords,
                target_keyword=self.target_keyword,
                max_tokens=self.max_tokens,
            )

            if score > best_score:
                best_score = score
                best_text = self.adapter.describe(pil, max_tokens=self.max_tokens)
                best_low_pert = new_pert
                no_improve = 0
                if verbose:
                    print(f"  [{self.adapter.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} | {best_text[:80]} *** IMPROVED ***")
            else:
                no_improve += 1

            if early_stop and best_score >= SUCCESS_SCORE:
                if verbose:
                    print(f"  SUCCESS at query {self.adapter.query_count}!")
                break

            if no_improve > 0 and no_improve % 100 == 0 and verbose:
                print(f"  [{self.adapter.query_count:5d}] best={best_score:.1f} "
                      f"no_improve={no_improve}")

        return self._finish(clean_arr, best_low_pert, best_score, best_text, verbose)

    def _finish(self, clean_arr, low_pert, best_score, best_text, verbose):
        adv_arr = apply_perturbation(clean_arr, low_pert, self.epsilon)
        adv_pil = pil_from_array(adv_arr)
        linf = np.abs(adv_arr - clean_arr).max()

        if verbose:
            print(f"\n  Final: score={best_score:.1f} "
                  f"queries={self.adapter.query_count}")
            print(f"  L-inf: {linf:.8f} (budget: {self.epsilon:.8f})")
            print(f"  Text: {best_text[:200]}")

        return adv_pil


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Black-Box VLM Attack"
    )
    parser.add_argument("--host", default="http://127.0.0.1:11471")
    parser.add_argument("--model-name", default="vyas")
    parser.add_argument("--api-type", default="openai",
                        choices=["openai", "ollama"])
    parser.add_argument("--image", default="data/vlm/dog03.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--low-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--target-keyword", default="cat")
    parser.add_argument("--output", default="outputs/adv_blackbox_universal.png")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    from models.api_adapter import APIVLMAdapter, OllamaVLMAdapter2

    if args.api_type == "ollama":
        adapter = OllamaVLMAdapter2(
            host=args.host, model_name=args.model_name,
        )
    else:
        adapter = APIVLMAdapter(
            name=args.model_name,
            host=args.host,
            model_name=args.model_name,
        )

    print(f"Model: {adapter.name} ({args.host})")
    print(f"Image: {args.image}")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print()

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((adapter.image_size, adapter.image_size), Image.LANCZOS)

    print(f"Clean description:")
    desc = adapter.describe(pil, max_tokens=100)
    print(f"  {desc[:200]}")
    print()

    attack = UniversalBlackBoxAttack(
        adapter=adapter,
        epsilon=args.epsilon,
        queries=args.queries,
        low_dim=args.low_dim,
        seed=args.seed,
        max_tokens=args.max_tokens,
        target_keyword=args.target_keyword,
    )

    print(f"Running black-box attack...")
    t0 = time.time()
    adv_pil = attack.attack(pil, verbose=True)
    elapsed = time.time() - t0
    print(f"\nAttack completed in {elapsed:.1f}s")

    print(f"\n=== VERIFICATION ===")
    desc = adapter.describe(adv_pil, max_tokens=200)
    has_dog = any(kw in desc.lower() for kw in DEFAULT_KEYWORDS)
    has_target = args.target_keyword in desc.lower()
    print(f"Dog keyword present: {has_dog}")
    print(f"Target keyword present: {has_target}")
    print(f"Description: {desc[:300]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")
