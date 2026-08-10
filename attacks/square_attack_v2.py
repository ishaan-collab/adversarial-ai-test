"""
Square Attack + Genetic Algorithm: Black-Box Adversarial Attack on Ollama moondream.

Two-phase approach:
  Phase 1: Square Attack with large structured perturbations
           - Push large regions to ±epsilon boundary
           - Multi-square perturbation per query
           - Refined scoring: position + count + text length

  Phase 2: Genetic refinement (if Phase 1 finds partial success)
           - Maintain population of perturbation patterns
           - Crossover + mutation
           - Select best-scoring individuals

Scoring: Continuous score from Ollama output (higher = better for attacker):
  - Base: position of first dog keyword (0-20)
  - Bonus: fewer dog keyword occurrences
  - Bonus: longer text without dog keyword

Usage:
    from attacks.square_attack_v2 import BlackBoxAttack

    attack = BlackBoxAttack(host="http://127.0.0.1:11435")
    adv_pil = attack.attack(clean_pil, epsilon=8/255, queries=5000)
"""

import io
import json
import time
import random
import base64
import requests
import numpy as np
from PIL import Image


# ============================================================
# DOG-RELATED KEYWORDS
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
    """Query Ollama with streaming and return list of tokens."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
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
    """
    Compute a continuous score from Ollama tokens.
    Higher = better for attacker (less dog-like).

    Score components:
    - dog_position: position of first dog keyword (0..num_predict)
      num_predict = no dog found (best)
    - dog_count: number of dog keyword occurrences (fewer = better)
    - text_length: length of text before first dog word (longer = better)

    Final score = dog_position * 10 - dog_count * 5 + text_before_dog * 0.1
    """
    text = "".join(tokens)
    text_lower = text.lower()

    # Find first dog keyword position
    dog_pos = num_predict
    char_pos = len(text)
    accumulated = ""
    for i, tok in enumerate(tokens):
        accumulated += tok
        lower = accumulated.lower()
        for kw in DOG_KEYWORDS:
            if kw in lower:
                recent = "".join(tokens[max(0, i - 3):i + 1]).lower()
                if kw in recent:
                    dog_pos = i
                    # Find character position of first dog keyword
                    idx = lower.find(kw)
                    if idx >= 0:
                        char_pos = idx
                    break
        if dog_pos < num_predict:
            break

    # Count dog keyword occurrences
    dog_count = 0
    for kw in DOG_KEYWORDS:
        dog_count += text_lower.count(kw)

    # Text before dog keyword
    text_before_dog = char_pos if dog_pos < num_predict else len(text)

    # Continuous score: higher = better for attacker
    score = dog_pos * 10.0 - dog_count * 5.0 + min(text_before_dog, 50) * 0.1

    return score, dog_pos, dog_count, text


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
                return -999, "<error>", []
            time.sleep(1)

    score, dog_pos, dog_count, text = compute_score(tokens, num_predict)
    return score, text, tokens


# ============================================================
# BLACK-BOX ATTACK
# ============================================================

class BlackBoxAttack:
    """
    Multi-strategy black-box attack on Ollama moondream.

    Phase 1: Square Attack with large perturbations
    Phase 2: SPSA gradient estimation (if Phase 1 stalls)
    """

    def __init__(self, host="http://127.0.0.1:11435",
                 model="moondream", num_predict=20, seed=42):
        self.host = host
        self.model = model
        self.num_predict = num_predict
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.query_count = 0
        self.history = []

    def _pil_from_array(self, arr):
        return Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        )

    def _apply_multi_square(self, arr, clean_arr, epsilon,
                            num_squares=1, min_size=5, max_size=50):
        """Apply multiple random square perturbations."""
        H, W, C = arr.shape
        result = arr.copy()
        lower = np.clip(clean_arr - epsilon, 0, 1)
        upper = np.clip(clean_arr + epsilon, 0, 1)

        for _ in range(num_squares):
            size = self.rng.randint(min_size, max_size)
            size = min(size, H, W)
            y0 = self.rng.randint(0, H - size)
            x0 = self.rng.randint(0, W - size)

            direction = self.np_rng.choice([-1, 1], size=C)
            for c in range(C):
                if direction[c] > 0:
                    result[y0:y0+size, x0:x0+size, c] = upper[
                        y0:y0+size, x0:x0+size, c
                    ]
                else:
                    result[y0:y0+size, x0:x0+size, c] = lower[
                        y0:y0+size, x0:x0+size, c
                    ]

        return np.clip(result, 0, 1)

    def _apply_full_noise(self, clean_arr, epsilon, density=0.1):
        """Apply random ±epsilon noise to a fraction of pixels."""
        H, W, C = clean_arr.shape
        noise = self.np_rng.choice(
            [-epsilon, 0, epsilon], size=(H, W, C), p=[density/2, 1-density, density/2]
        ).astype(np.float32)
        result = clean_arr + noise
        return np.clip(result, 0, 1)

    def attack(self, clean_pil, epsilon=8/255, queries=5000,
               verbose=True, early_stop=True):
        """
        Run multi-strategy black-box attack.

        Strategy schedule:
          - Queries 0-500:    Full-image random noise (density sweep)
          - Queries 500-2000: Square attack (large → small squares)
          - Queries 2000-5000: SPSA gradient estimation
        """
        self.query_count = 0
        self.history = []

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape

        lower = np.clip(clean_arr - epsilon, 0, 1)
        upper = np.clip(clean_arr + epsilon, 0, 1)

        # Initial score
        pil = self._pil_from_array(clean_arr)
        score, text, _ = ollama_score(
            pil, host=self.host, num_predict=self.num_predict
        )
        self.query_count += 1
        best_score = score
        best_arr = clean_arr.copy()
        best_text = text

        if verbose:
            print(f"  [init] score={best_score:.1f} | {best_text[:100]}")

        self.history.append({
            "query": 0, "score": best_score, "text": best_text[:200],
            "phase": "init",
        })

        no_improve = 0

        # ============================================
        # PHASE 1: Full-image random noise (density sweep)
        # Try different noise densities to find which works
        # ============================================
        phase1_end = min(500, queries // 5)

        if verbose:
            print(f"\n  Phase 1: Random noise ({phase1_end} queries)")

        densities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        for q in range(phase1_end):
            if self.query_count >= queries:
                break

            density = densities[q % len(densities)]
            candidate = self._apply_full_noise(clean_arr, epsilon, density)

            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_arr = candidate
                best_text = text
                no_improve = 0
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} density={density:.2f} "
                          f"| {text[:80]} *** IMPROVED ***")
            else:
                no_improve += 1

            self.history.append({
                "query": self.query_count, "score": score,
                "text": text[:200], "phase": "noise",
            })

            if early_stop and best_score >= self.num_predict * 10:
                if verbose:
                    print(f"  SUCCESS at query {self.query_count}!")
                    print(f"  Text: {best_text[:200]}")
                return self._pil_from_array(best_arr), self._info(
                    epsilon, best_score, best_text
                )

        # ============================================
        # PHASE 2: Square attack (large → small)
        # ============================================
        phase2_end = min(2000, queries * 2 // 5)

        if verbose:
            print(f"\n  Phase 2: Square attack "
                  f"({phase2_end - phase1_end} queries)")

        for q in range(phase1_end, phase2_end):
            if self.query_count >= queries:
                break

            progress = (q - phase1_end) / max(1, phase2_end - phase1_end)
            max_size = max(5, int(min(H, W) * (0.5 - 0.4 * progress)))
            min_size = max(3, int(min(H, W) * (0.05 + 0.1 * progress)))
            min_size = min(min_size, max_size)

            # Try 1-3 squares per query
            num_sq = self.rng.randint(1, 4)
            candidate = self._apply_multi_square(
                best_arr, clean_arr, epsilon,
                num_squares=num_sq,
                min_size=min_size, max_size=max_size,
            )

            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            if score > best_score:
                best_score = score
                best_arr = candidate
                best_text = text
                no_improve = 0
                if verbose:
                    print(f"  [{self.query_count:5d}] score={score:.1f} "
                          f"best={best_score:.1f} "
                          f"squares={num_sq} size={min_size}-{max_size} "
                          f"| {text[:80]} *** IMPROVED ***")
            else:
                no_improve += 1

            self.history.append({
                "query": self.query_count, "score": score,
                "text": text[:200], "phase": "square",
            })

            if (q + 1) % 200 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"no_improve={no_improve}")

            if early_stop and best_score >= self.num_predict * 10:
                if verbose:
                    print(f"  SUCCESS at query {self.query_count}!")
                    print(f"  Text: {best_text[:200]}")
                return self._pil_from_array(best_arr), self._info(
                    epsilon, best_score, best_text
                )

        # ============================================
        # PHASE 3: SPSA gradient estimation
        # ============================================
        if verbose:
            print(f"\n  Phase 3: SPSA "
                  f"({queries - self.query_count} queries remaining)")

        alpha = epsilon / 3.0
        c = epsilon / 5.0

        while self.query_count < queries:
            # SPSA: sample random direction
            delta = self.np_rng.choice(
                [-1, 1], size=(H, W, C)
            ).astype(np.float32)

            # Plus perturbation
            plus = np.clip(best_arr + c * delta, lower, upper)
            pil_plus = self._pil_from_array(plus)
            score_plus, _, _ = ollama_score(
                pil_plus, host=self.host, num_predict=self.num_predict
            )

            # Minus perturbation
            minus = np.clip(best_arr - c * delta, lower, upper)
            pil_minus = self._pil_from_array(minus)
            score_minus, _, _ = ollama_score(
                pil_minus, host=self.host, num_predict=self.num_predict
            )

            self.query_count += 2

            # Gradient estimate
            if abs(score_plus - score_minus) > 0.01:
                grad_est = (
                    (score_plus - score_minus) / (2 * c) * delta
                )
                # PGD step
                step = alpha * np.sign(grad_est)
                candidate = np.clip(
                    best_arr + step, lower, upper
                )
                candidate = np.clip(candidate, 0, 1)

                # Evaluate
                pil = self._pil_from_array(candidate)
                score, text, _ = ollama_score(
                    pil, host=self.host, num_predict=self.num_predict
                )
                self.query_count += 1

                if score > best_score:
                    best_score = score
                    best_arr = candidate
                    best_text = text
                    no_improve = 0
                    if verbose:
                        print(f"  [{self.query_count:5d}] score={score:.1f} "
                              f"best={best_score:.1f} "
                              f"| {text[:80]} *** IMPROVED ***")
                else:
                    no_improve += 1
            else:
                no_improve += 1

            # Decay alpha
            if self.query_count % 200 == 0:
                alpha *= 0.9

            if self.query_count % 200 == 0 and verbose:
                print(f"  [{self.query_count:5d}] best={best_score:.1f} "
                      f"no_improve={no_improve}")

            self.history.append({
                "query": self.query_count, "score": best_score,
                "text": best_text[:200], "phase": "spsa",
            })

            if early_stop and best_score >= self.num_predict * 10:
                if verbose:
                    print(f"  SUCCESS at query {self.query_count}!")
                    print(f"  Text: {best_text[:200]}")
                break

        if verbose:
            print(f"\n  Final: score={best_score:.1f} "
                  f"queries={self.query_count}")
            print(f"  Text: {best_text[:200]}")

        return self._pil_from_array(best_arr), self._info(
            epsilon, best_score, best_text
        )

    def _info(self, epsilon, best_score, best_text):
        return {
            "history": self.history,
            "best_score": best_score,
            "queries": self.query_count,
            "epsilon": epsilon,
            "best_text": best_text,
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Black-box adversarial attack on Ollama moondream"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_blackbox.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378)
    args = parser.parse_args()

    # Force line buffering
    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print()

    attack = BlackBoxAttack(
        seed=args.seed, num_predict=args.num_predict
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

    # L-inf check
    clean_arr = np.array(pil, dtype=np.float32) / 255.0
    adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
    linf = np.abs(adv_arr - clean_arr).max()
    print(f"\nL-inf: {linf:.8f} (budget: {args.epsilon:.8f})")
    print(f"Within budget: {linf <= args.epsilon + 1e-6}")
