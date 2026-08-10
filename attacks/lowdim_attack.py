"""
Low-Dimensional Black-Box Attack on Ollama moondream.

Key insight: 378×378×3 = 428K dimensions is too many for query-based
gradient estimation. Instead, optimize in a low-dimensional space (e.g.,
32×32×3 = 3072 dims) and bilinearly upscale to 378×378.

This has two benefits:
  1. Fewer dimensions → SPSA gradient estimation is much more effective
  2. Smooth perturbations → more likely to survive Q4 quantization
     (which acts as a low-pass filter on high-frequency noise)

Strategy:
  Phase 1: Random search in low-dim space (find promising direction)
  Phase 2: SPSA gradient estimation in low-dim space
  Phase 3: Binary search along best direction (refine magnitude)
"""

import io
import json
import time
import random
import base64
import requests
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ============================================================
# DOG KEYWORDS
# ============================================================

DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]


# ============================================================
# OLLAMA QUERY + SCORING
# ============================================================

def ollama_query(pil_img, host="http://127.0.0.1:11435",
                 model="moondream",
                 prompt=" Question: What do you see in this image?\n\n Answer:",
                 temperature=0.1, num_predict=20):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "images": [img_b64],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    resp = requests.post(
        f"{host}/api/generate", json=payload, timeout=120, stream=True
    )

    tokens = []
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk.get("response"):
                tokens.append(chunk["response"])
            if chunk.get("done"):
                break
    return tokens


def compute_score(tokens, num_predict=20):
    """Continuous score. Higher = better for attacker."""
    text = "".join(tokens)
    text_lower = text.lower()

    dog_pos = num_predict
    char_pos = len(text)
    accumulated = ""
    for i, tok in enumerate(tokens):
        accumulated += tok
        lower = accumulated.lower()
        for kw in DOG_KEYWORDS:
            if kw in lower:
                recent = "".join(
                    tokens[max(0, i - 3):i + 1]
                ).lower()
                if kw in recent:
                    dog_pos = i
                    idx = lower.find(kw)
                    if idx >= 0:
                        char_pos = idx
                    break
        if dog_pos < num_predict:
            break

    dog_count = sum(text_lower.count(kw) for kw in DOG_KEYWORDS)
    text_before = char_pos if dog_pos < num_predict else len(text)

    score = (
        dog_pos * 10.0
        - dog_count * 5.0
        + min(text_before, 50) * 0.1
    )

    return score, dog_pos, dog_count, text


def ollama_score(pil_img, host="http://127.0.0.1:11435",
                 num_predict=20, retries=2):
    for attempt in range(retries + 1):
        try:
            tokens = ollama_query(
                pil_img, host=host, num_predict=num_predict
            )
            break
        except Exception:
            if attempt == retries:
                return -999, "<error>", []
            time.sleep(1)

    score, _, _, text = compute_score(tokens, num_predict)
    return score, text, tokens


# ============================================================
# LOW-DIM ATTACK
# ============================================================

class LowDimAttack:
    """
    Black-box attack in a low-dimensional perturbation space.

    Perturbation is optimized at low resolution (e.g. 32x32) and
    bilinearly upscaled to the full image size. This makes SPSA
    gradient estimation tractable and produces smooth perturbations
    that are more robust to Q4 quantization.
    """

    def __init__(self, host="http://127.0.0.1:11435",
                 model="moondream", num_predict=20,
                 low_dim=32, seed=42):
        self.host = host
        self.model = model
        self.num_predict = num_predict
        self.low_dim = low_dim
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.query_count = 0
        self.history = []

    def _pil_from_array(self, arr):
        return Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        )

    def _upscale_perturbation(self, low_pert, target_h, target_w):
        """Bilinearly upscale [low_h, low_w, 3] perturbation to target size."""
        t = torch.from_numpy(low_pert).permute(2, 0, 1).unsqueeze(0)
        t_up = F.interpolate(
            t, size=(target_h, target_w),
            mode="bilinear", align_corners=False, antialias=True,
        )
        return t_up.squeeze(0).permute(1, 2, 0).numpy()

    def _apply_perturbation(self, clean_arr, low_pert, epsilon):
        """Apply upscaled perturbation to clean image, clipped to epsilon ball."""
        H, W, C = clean_arr.shape
        pert_up = self._upscale_perturbation(low_pert, H, W)

        # Scale perturbation to epsilon budget
        pert_max = np.abs(pert_up).max()
        if pert_max > 0:
            pert_up = pert_up / pert_max * epsilon

        result = clean_arr + pert_up
        return np.clip(result, 0, 1)

    def attack(self, clean_pil, epsilon=8/255, queries=5000,
               verbose=True, early_stop=True):
        """
        Run low-dimensional black-box attack.

        Phase 1: Random search (find promising direction)
        Phase 2: SPSA gradient estimation
        Phase 3: Line search along best direction
        """
        self.query_count = 0
        self.history = []

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape
        D = self.low_dim

        # Initial score
        pil = self._pil_from_array(clean_arr)
        score, text, _ = ollama_score(
            pil, host=self.host, num_predict=self.num_predict
        )
        self.query_count += 1
        best_score = score
        best_text = text
        best_low_pert = np.zeros((D, D, C), dtype=np.float32)

        if verbose:
            print(f"  [init] score={best_score:.1f} | {best_text[:100]}")

        # Success threshold: score >= num_predict * 10 (no dog in 20 tokens)
        success_threshold = self.num_predict * 10.0

        # ============================================
        # PHASE 1: Random search (20% of queries)
        # ============================================
        phase1_end = min(200, queries // 5)
        if verbose:
            print(f"\n  Phase 1: Random search ({phase1_end} queries)")

        for q in range(phase1_end):
            if self.query_count >= queries:
                break

            # Random low-dim perturbation
            low_pert = self.np_rng.randn(D, D, C).astype(np.float32) * 0.5
            candidate = self._apply_perturbation(clean_arr, low_pert, epsilon)

            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_text = text
                best_low_pert = low_pert
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} "
                          f"| {text[:80]} *** IMPROVED ***")

            if early_stop and best_score >= success_threshold:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

        # ============================================
        # PHASE 2: SPSA gradient estimation (60% of queries)
        # ============================================
        phase2_end = min(queries - 200, int(queries * 0.8))
        if verbose:
            print(f"\n  Phase 2: SPSA ({phase2_end - phase1_end} queries)")

        alpha = 0.3  # step size in low-dim space
        c_spsa = 0.2  # perturbation magnitude for gradient estimation

        spsa_interval = 50  # evaluate full quality every N SPSA steps

        for q in range(phase1_end, phase2_end):
            if self.query_count >= queries:
                break

            # SPSA: random direction in low-dim space
            delta = self.np_rng.choice(
                [-1, 1], size=(D, D, C)
            ).astype(np.float32)

            # Plus
            plus_pert = best_low_pert + c_spsa * delta
            plus_img = self._apply_perturbation(clean_arr, plus_pert, epsilon)
            pil_plus = self._pil_from_array(plus_img)
            score_plus, _, _ = ollama_score(
                pil_plus, host=self.host, num_predict=self.num_predict
            )

            # Minus
            minus_pert = best_low_pert - c_spsa * delta
            minus_img = self._apply_perturbation(
                clean_arr, minus_pert, epsilon
            )
            pil_minus = self._pil_from_array(minus_img)
            score_minus, _, _ = ollama_score(
                pil_minus, host=self.host, num_predict=self.num_predict
            )

            self.query_count += 2

            # Gradient estimate
            grad_est = (
                (score_plus - score_minus) / (2 * c_spsa) * delta
            )

            # Gradient step
            new_pert = best_low_pert + alpha * np.sign(grad_est)

            # Evaluate periodically
            if (q - phase1_end) % spsa_interval == 0 or q == phase2_end - 1:
                candidate = self._apply_perturbation(
                    clean_arr, new_pert, epsilon
                )
                pil = self._pil_from_array(candidate)
                score, text, _ = ollama_score(
                    pil, host=self.host, num_predict=self.num_predict
                )
                self.query_count += 1

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_low_pert = new_pert
                    if verbose:
                        print(f"  [{self.query_count:5d}] score={score:.1f} "
                              f"best={best_score:.1f} "
                              f"| {text[:80]} *** IMPROVED ***")
                else:
                    # Decay alpha
                    alpha *= 0.9

            if early_stop and best_score >= success_threshold:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

            if (q + 1) % 200 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"alpha={alpha:.4f}")

        # ============================================
        # PHASE 3: Line search along best direction (remaining queries)
        # ============================================
        if verbose:
            print(f"\n  Phase 3: Line search "
                  f"({queries - self.query_count} queries)")

        # Try scaling the best perturbation
        scales = [0.5, 0.75, 1.25, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0,
                  0.25, 0.1, -0.25, -0.1, 3.0, -3.0, 5.0, -5.0]

        for scale in scales:
            if self.query_count >= queries:
                break

            scaled_pert = best_low_pert * scale
            candidate = self._apply_perturbation(
                clean_arr, scaled_pert, epsilon
            )
            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_text = text
                best_low_pert = scaled_pert
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} scale={scale:.2f}"
                          f" *** IMPROVED ***")

            if early_stop and best_score >= success_threshold:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

        # Try adding random noise to best direction
        while self.query_count < queries:
            noise = self.np_rng.randn(D, D, C).astype(np.float32) * 0.1
            new_pert = best_low_pert + noise
            candidate = self._apply_perturbation(
                clean_arr, new_pert, epsilon
            )
            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_text = text
                best_low_pert = new_pert
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} "
                          f"| {text[:80]} *** IMPROVED ***")

            if early_stop and best_score >= success_threshold:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

        return self._finish(clean_arr, best_low_pert, epsilon,
                            best_score, best_text, verbose)

    def _finish(self, clean_arr, low_pert, epsilon,
                best_score, best_text, verbose):
        """Generate final adversarial image and print results."""
        adv_arr = self._apply_perturbation(clean_arr, low_pert, epsilon)
        adv_pil = self._pil_from_array(adv_arr)

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
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Low-dimensional black-box attack on Ollama moondream"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_lowdim.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378)
    parser.add_argument("--low-dim", type=int, default=32,
                        help="Low-dim grid size (e.g. 32, 48, 64)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print(f"Low-dim: {args.low_dim}x{args.low_dim}")
    print()

    attack = LowDimAttack(
        seed=args.seed, num_predict=args.num_predict,
        low_dim=args.low_dim,
    )
    adv_pil, info = attack.attack(
        pil, epsilon=args.epsilon, queries=args.queries, verbose=True
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")

    # Verify with full 200-token description
    print("\n=== VERIFICATION (full 200-token description) ===")
    score, text, _ = ollama_score(
        adv_pil, host="http://127.0.0.1:11435", num_predict=200
    )
    has_dog = any(kw in text.lower() for kw in DOG_KEYWORDS)
    print(f"Dog keyword present: {has_dog}")
    print(f"Description: {text[:300]}")

    clean_arr = np.array(pil, dtype=np.float32) / 255.0
    adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
    linf = np.abs(adv_arr - clean_arr).max()
    print(f"\nL-inf: {linf:.8f} (budget: {args.epsilon:.8f})")
    print(f"Within budget: {linf <= args.epsilon + 1e-6}")
