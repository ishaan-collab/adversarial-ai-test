"""
Black-Box Attack on Qwen 35B VLM via OpenAI-compatible API.

Adapts the SPSA + random search approach from blackbox_attack.py
to query Qwen3.6-35B-A3B (Q8_0) via llama-server on port 11471.

Key differences from Ollama attack:
  - OpenAI-compatible chat completions API (not Ollama /api/generate)
  - Image sent as base64 data URL in messages content
  - Q8 quantization (8-bit, 4x weaker noise than Q4)
  - 35B parameter model (much larger than 1B moondream)
  - ~2.5s per query (vs 0.3s for Ollama)

Usage:
    PYTHONPATH=. python attacks/qwen_blackbox.py --image data/vlm/dog03.jpg
    PYTHONPATH=. python attacks/qwen_blackbox.py --epsilon 8/255 --queries 2000
"""

import io
import json
import time
import random
import base64
import requests
import numpy as np
import sys
from PIL import Image
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

QWEN_HOST = "http://127.0.0.1:11471"
QWEN_MODEL = "vyas"

DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]

TARGET_KEYWORD = "cat"
SUCCESS_SCORE = 200.0


# ============================================================
# QWEN QUERY + SCORING
# ============================================================

def qwen_query(pil_img, host=QWEN_HOST, model=QWEN_MODEL,
               question="What do you see in this image?",
               temperature=0.1, max_tokens=50):
    """Query Qwen 35B via OpenAI-compatible API with an image."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        f"{host}/v1/chat/completions", json=payload, timeout=120,
    )
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

    return resp.json()["choices"][0]["message"]["content"].strip()


def compute_score(text, keywords=DOG_KEYWORDS, target=TARGET_KEYWORD):
    """
    Character-position scoring. Higher = better for attacker.
    - Dog keyword early: low score (bad for attacker)
    - No dog keyword: score = 200+ (success)
    - Target "cat" present: +50 bonus
    """
    text_lower = text.lower().strip()

    if not text_lower:
        return SUCCESS_SCORE

    min_pos = len(text_lower)
    for kw in keywords:
        pos = text_lower.find(kw)
        if 0 <= pos < min_pos:
            min_pos = pos

    dog_count = sum(text_lower.count(kw) for kw in keywords)
    target_bonus = 50.0 if target and target in text_lower else 0.0

    if min_pos >= len(text_lower):
        score = SUCCESS_SCORE + len(text_lower) * 0.5 + target_bonus
    else:
        score = float(min_pos) - dog_count * 5.0 + target_bonus

    return score


def qwen_score(pil_img, host=QWEN_HOST, max_tokens=50, retries=2):
    """Query Qwen and return (score, text)."""
    for attempt in range(retries + 1):
        try:
            text = qwen_query(pil_img, host=host, max_tokens=max_tokens)
            break
        except Exception as e:
            if attempt == retries:
                return -999.0, f"<error: {e}>"
            time.sleep(2)

    score = compute_score(text)
    return score, text


# ============================================================
# LOW-DIM PERTURBATION UTILITIES
# ============================================================

def upscale_perturbation(low_pert, target_h, target_w):
    """Bilinearly upscale [D, D, 3] perturbation to [H, W, 3]."""
    t = torch.from_numpy(low_pert).permute(2, 0, 1).unsqueeze(0)
    t_up = F.interpolate(
        t, size=(target_h, target_w),
        mode="bilinear", align_corners=False, antialias=True,
    )
    return t_up.squeeze(0).permute(1, 2, 0).numpy()


def apply_perturbation(clean_arr, low_pert, epsilon):
    """Apply upscaled perturbation to clean image, clipped to epsilon ball."""
    H, W, C = clean_arr.shape
    pert_up = upscale_perturbation(low_pert, H, W)
    pert_up = np.clip(pert_up, -epsilon, epsilon)

    result = clean_arr + pert_up
    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)
    return np.clip(result, lower, upper)


def pil_from_array(arr):
    return Image.fromarray(
        (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    )


# ============================================================
# BLACK-BOX ATTACK
# ============================================================

class QwenBlackBoxAttack:
    """
    Multi-phase black-box attack on Qwen 35B VLM.

    Phase 1: Random search in low-dim space (exploration)
    Phase 2: SPSA gradient estimation with accumulation
    Phase 3: Square refinement (fine-tuning)
    """

    def __init__(self, host=QWEN_HOST, model=QWEN_MODEL,
                 low_dim=32, seed=42, max_tokens=50):
        self.host = host
        self.model = model
        self.low_dim = low_dim
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.query_count = 0
        self.history = []

    def attack(self, clean_pil, epsilon=8 / 255, queries=1000,
               verbose=True, early_stop=True):
        self.query_count = 0
        self.history = []

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape
        D = self.low_dim

        # Initial score
        score, text = qwen_score(
            pil_from_array(clean_arr), host=self.host,
            max_tokens=self.max_tokens,
        )
        self.query_count += 1
        best_score = score
        best_text = text
        best_low_pert = np.zeros((D, D, C), dtype=np.float32)

        if verbose:
            print(f"  [init] score={best_score:.1f} | {best_text[:100]}")

        if early_stop and best_score >= SUCCESS_SCORE:
            return self._finish(clean_arr, best_low_pert, epsilon,
                                best_score, best_text, verbose)

        # ============================================
        # Phase 1: Random search (~10% of queries)
        # ============================================
        phase1_end = min(80, queries // 8)
        if verbose:
            print(f"\n  Phase 1: Random search ({phase1_end} queries)")

        for q in range(phase1_end):
            if self.query_count >= queries:
                break

            low_pert = self.np_rng.randn(D, D, C).astype(np.float32) * epsilon * 0.5
            candidate = apply_perturbation(clean_arr, low_pert, epsilon)
            pil = pil_from_array(candidate)
            score, text = qwen_score(
                pil, host=self.host, max_tokens=self.max_tokens,
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_text = text
                best_low_pert = low_pert.copy()
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} "
                          f"| {text[:80]} *** IMPROVED ***")

            if early_stop and best_score >= SUCCESS_SCORE:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

        # ============================================
        # Phase 2: SPSA with accumulation (~75% of queries)
        # ============================================
        phase2_end = max(phase1_end, min(int(queries * 0.85), queries - 50))
        if verbose:
            print(f"\n  Phase 2: SPSA ({phase2_end - phase1_end} iters, "
                  f"{(phase2_end - phase1_end) * 2 + (phase2_end - phase1_end) // 20} queries)")

        alpha = epsilon * 0.5
        c_spsa = epsilon * 0.1
        eval_interval = 15

        current_pert = best_low_pert.copy()
        spsa_iters = 0

        for q in range(phase1_end, phase2_end):
            if self.query_count + 2 > queries:
                break

            delta = self.np_rng.choice(
                [-1, 1], size=(D, D, C)
            ).astype(np.float32)

            plus_pert = current_pert + c_spsa * delta
            plus_img = apply_perturbation(clean_arr, plus_pert, epsilon)
            pil_plus = pil_from_array(plus_img)
            score_plus, _ = qwen_score(
                pil_plus, host=self.host, max_tokens=self.max_tokens,
            )

            minus_pert = current_pert - c_spsa * delta
            minus_img = apply_perturbation(clean_arr, minus_pert, epsilon)
            pil_minus = pil_from_array(minus_img)
            score_minus, _ = qwen_score(
                pil_minus, host=self.host, max_tokens=self.max_tokens,
            )

            self.query_count += 2
            spsa_iters += 1

            grad_est = (
                (score_plus - score_minus) / (2 * c_spsa) * delta
            )

            current_pert = current_pert + alpha * np.sign(grad_est)

            if spsa_iters % eval_interval == 0 or q == phase2_end - 1:
                candidate = apply_perturbation(
                    clean_arr, current_pert, epsilon
                )
                pil = pil_from_array(candidate)
                score, text = qwen_score(
                    pil, host=self.host, max_tokens=self.max_tokens,
                )
                self.query_count += 1

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_low_pert = current_pert.copy()
                    if verbose:
                        print(f"  [{self.query_count:5d}] score={score:.1f} "
                              f"best={best_score:.1f} "
                              f"| {text[:80]} *** IMPROVED ***")
                else:
                    current_pert = best_low_pert.copy()
                    alpha *= 0.95

                if early_stop and best_score >= SUCCESS_SCORE:
                    return self._finish(clean_arr, best_low_pert, epsilon,
                                        best_score, best_text, verbose)

            if (q + 1) % 100 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"alpha={alpha:.6f} spsa_iters={spsa_iters}")

        # ============================================
        # Phase 3: Square refinement
        # ============================================
        remaining = queries - self.query_count
        if verbose:
            print(f"\n  Phase 3: Square refinement ({remaining} queries)")

        no_improve = 0

        while self.query_count < queries:
            p = self.rng.randint(2, max(3, D // 2))
            y0 = self.rng.randint(0, D - p)
            x0 = self.rng.randint(0, D - p)

            new_pert = best_low_pert.copy()
            noise = self.np_rng.randn(p, p, C).astype(np.float32) * epsilon * 0.3
            new_pert[y0:y0 + p, x0:x0 + p] += noise

            candidate = apply_perturbation(clean_arr, new_pert, epsilon)
            pil = pil_from_array(candidate)
            score, text = qwen_score(
                pil, host=self.host, max_tokens=self.max_tokens,
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_text = text
                best_low_pert = new_pert
                no_improve = 0
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} "
                          f"| {text[:80]} *** IMPROVED ***")
            else:
                no_improve += 1

            if early_stop and best_score >= SUCCESS_SCORE:
                if verbose:
                    print(f"  SUCCESS at query {self.query_count}!")
                    print(f"  Text: {best_text[:200]}")
                break

            if no_improve > 0 and no_improve % 100 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"no_improve={no_improve}")

        return self._finish(clean_arr, best_low_pert, epsilon,
                            best_score, best_text, verbose)

    def _finish(self, clean_arr, low_pert, epsilon,
                best_score, best_text, verbose):
        adv_arr = apply_perturbation(clean_arr, low_pert, epsilon)
        adv_pil = pil_from_array(adv_arr)
        linf = np.abs(adv_arr - clean_arr).max()

        if verbose:
            print(f"\n  Final: score={best_score:.1f} "
                  f"queries={self.query_count}")
            print(f"  L-inf: {linf:.8f} (budget: {epsilon:.8f})")
            print(f"  Text: {best_text[:200]}")

        return adv_pil, {
            "best_score": best_score,
            "queries": self.query_count,
            "epsilon": epsilon,
            "best_text": best_text,
            "linf": float(linf),
            "success": best_score >= SUCCESS_SCORE,
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Black-box attack on Qwen 35B VLM"
    )
    parser.add_argument("--image", default="data/vlm/dog03.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--output", default="outputs/adv_qwen_blackbox.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-dim", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((378, 378), Image.LANCZOS)

    print(f"Image: {args.image} (378x378)")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print(f"Low-dim: {args.low_dim}x{args.low_dim}")
    print(f"Model: Qwen3.6-35B-A3B (Q8_0) at {QWEN_HOST}")
    print()

    attack = QwenBlackBoxAttack(
        seed=args.seed, low_dim=args.low_dim,
        max_tokens=args.max_tokens,
    )
    adv_pil, info = attack.attack(
        pil, epsilon=args.epsilon, queries=args.queries, verbose=True
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")

    print("\n=== VERIFICATION (full 200-token description) ===")
    score, text = qwen_score(
        adv_pil, host=QWEN_HOST, max_tokens=200,
    )
    has_dog = any(kw in text.lower() for kw in DOG_KEYWORDS)
    has_cat = "cat" in text.lower()
    print(f"Dog keyword present: {has_dog}")
    print(f"Cat keyword present: {has_cat}")
    print(f"Score: {score:.1f}")
    print(f"Description: {text[:300]}")
