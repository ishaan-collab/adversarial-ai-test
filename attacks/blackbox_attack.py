"""
Improved Black-Box Attack on Ollama moondream.

Key improvements over existing attacks:
  1. Character-position scoring (0-200+ range, vs. 0-20 token position)
     - Most perturbations won't change token position, but will change
       character position, giving the attack a usable gradient signal.
  2. Correct SPSA gradient accumulation
     - Previous code estimated gradient at best_low_pert every step,
       discarding intermediate steps. This version accumulates steps
       from the current point and reverts on failure.
  3. Multi-phase: random search -> SPSA -> low-dim square refinement
  4. Low-dim perturbation (32x32x3, bilinearly upscaled for smoothness)
  5. Targeted bonus (+50 if "cat" appears in output)

Usage:
    PYTHONPATH=. python attacks/blackbox_attack.py --image data/vlm/dog07.jpg
    PYTHONPATH=. python attacks/blackbox_attack.py --epsilon 16/255 --queries 10000
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
# DOG KEYWORDS
# ============================================================

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
# OLLAMA QUERY + SCORING
# ============================================================

def ollama_query(pil_img, host="http://127.0.0.1:11435",
                 model="moondream",
                 prompt=" Question: What do you see in this image?\n\n Answer:",
                 temperature=0.1, num_predict=20):
    """Query Ollama with streaming and return list of tokens."""
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


def compute_score(text, keywords=DOG_KEYWORDS, target=TARGET_KEYWORD):
    """
    Continuous score from Ollama output text.
    Higher = better for attacker.

    Uses CHARACTER position (not token position) for fine-grained signal:
    - If dog keyword found: score = char_position - dog_count * 5 + target_bonus
    - If no dog keyword: score = 200 + len(text) * 0.5 + target_bonus
    - Target bonus: +50 if "cat" appears

    Range: ~-20 (dog at char 0) to ~300 (no dog + cat + long text)
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


def ollama_score(pil_img, host="http://127.0.0.1:11435",
                 num_predict=20, retries=2):
    """Query Ollama and return (score, text, tokens)."""
    for attempt in range(retries + 1):
        try:
            tokens = ollama_query(
                pil_img, host=host, num_predict=num_predict
            )
            break
        except Exception:
            if attempt == retries:
                return -999.0, "<error>", []
            time.sleep(1)

    text = "".join(tokens)
    score = compute_score(text)
    return score, text, tokens


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
    """Apply upscaled perturbation to clean image, clipped to epsilon ball.

    The low_pert values are used directly (NOT normalized to full epsilon).
    This is critical for SPSA: plus/minus evaluations must differ by the
    actual delta, not be renormalized to the same magnitude.
    """
    H, W, C = clean_arr.shape
    pert_up = upscale_perturbation(low_pert, H, W)

    # Clip perturbation to epsilon ball (don't normalize!)
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
# IMPROVED BLACK-BOX ATTACK
# ============================================================

class BlackBoxAttack:
    """
    Improved black-box attack on Ollama moondream.

    Multi-phase approach:
      Phase 1: Random search in low-dim space (exploration, ~10% queries)
      Phase 2: SPSA gradient estimation with accumulation (~70% queries)
      Phase 3: Low-dim square refinement (fine-tuning, ~20% queries)

    Key fixes vs. existing lowdim_attack.py:
      - SPSA estimates gradient at current_pert, not best_low_pert
      - SPSA steps accumulate between evaluations
      - On failed evaluation: revert to best, decay alpha
      - Character-position scoring (0-200+ range vs. 0-20 token position)
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

    def attack(self, clean_pil, epsilon=8 / 255, queries=5000,
               verbose=True, early_stop=True):
        """
        Run the improved black-box attack.

        Args:
            clean_pil:   Clean PIL image.
            epsilon:     L-inf budget in [0, 1].
            queries:     Maximum Ollama queries.
            verbose:     Print progress.
            early_stop:  Stop if score >= SUCCESS_SCORE.

        Returns:
            adv_pil:  Best adversarial PIL Image found.
            info:     Dict with results.
        """
        self.query_count = 0
        self.history = []

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape
        D = self.low_dim

        # ============================================
        # Initial score
        # ============================================
        score, text, _ = ollama_score(
            pil_from_array(clean_arr), host=self.host,
            num_predict=self.num_predict
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
        phase1_end = min(100, queries // 10)
        if verbose:
            print(f"\n  Phase 1: Random search ({phase1_end} queries)")

        for q in range(phase1_end):
            if self.query_count >= queries:
                break

            low_pert = self.np_rng.randn(D, D, C).astype(np.float32) * epsilon * 0.5
            candidate = apply_perturbation(clean_arr, low_pert, epsilon)

            pil = pil_from_array(candidate)
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

            if early_stop and best_score >= SUCCESS_SCORE:
                return self._finish(clean_arr, best_low_pert, epsilon,
                                    best_score, best_text, verbose)

        # ============================================
        # Phase 2: SPSA with accumulation (~70% of queries)
        # ============================================
        phase2_end = max(phase1_end, min(int(queries * 0.8), queries - 100))
        if verbose:
            print(f"\n  Phase 2: SPSA ({phase2_end - phase1_end} iters)")

        alpha = epsilon * 0.5
        c_spsa = epsilon * 0.1
        eval_interval = 20

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
            score_plus, _, _ = ollama_score(
                pil_plus, host=self.host, num_predict=self.num_predict
            )

            minus_pert = current_pert - c_spsa * delta
            minus_img = apply_perturbation(clean_arr, minus_pert, epsilon)
            pil_minus = pil_from_array(minus_img)
            score_minus, _, _ = ollama_score(
                pil_minus, host=self.host, num_predict=self.num_predict
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
                score, text, _ = ollama_score(
                    pil, host=self.host, num_predict=self.num_predict
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

            if (q + 1) % 200 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"alpha={alpha:.4f} spsa_iters={spsa_iters}")

        # ============================================
        # Phase 3: Low-dim square refinement
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
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
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

            if no_improve > 0 and no_improve % 300 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"no_improve={no_improve}")

        return self._finish(clean_arr, best_low_pert, epsilon,
                            best_score, best_text, verbose)

    def _finish(self, clean_arr, low_pert, epsilon,
                best_score, best_text, verbose):
        """Generate final adversarial image and print results."""
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
        description="Improved black-box attack on Ollama moondream"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_blackbox_v3.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378)
    parser.add_argument("--low-dim", type=int, default=32)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print(f"Low-dim: {args.low_dim}x{args.low_dim}")
    print()

    attack = BlackBoxAttack(
        seed=args.seed, num_predict=args.num_predict,
        low_dim=args.low_dim,
    )
    adv_pil, info = attack.attack(
        pil, epsilon=args.epsilon, queries=args.queries, verbose=True
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")

    print("\n=== VERIFICATION (full 200-token description) ===")
    score, text, _ = ollama_score(
        adv_pil, host="http://127.0.0.1:11435", num_predict=200
    )
    has_dog = any(kw in text.lower() for kw in DOG_KEYWORDS)
    print(f"Dog keyword present: {has_dog}")
    print(f"Score: {score:.1f}")
    print(f"Description: {text[:300]}")

    clean_arr = np.array(pil, dtype=np.float32) / 255.0
    adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
    linf = np.abs(adv_arr - clean_arr).max()
    print(f"\nL-inf: {linf:.8f} (budget: {args.epsilon:.8f})")
    print(f"Within budget: {linf <= args.epsilon + 1e-6}")
