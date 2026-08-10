"""
Square Attack: Query-Based Black-Box Adversarial Attack on Ollama moondream.

The Square Attack is a state-of-the-art score-based black-box attack that
perturbs images by modifying square-shaped regions. It requires no gradient
information — only the model's output score.

Strategy:
  1. Start from the clean image
  2. Pick a random square (location, size, color direction)
  3. Apply perturbation to that square (push pixels to ±epsilon boundary)
  4. Query Ollama, compute a score (higher = better for attacker)
  5. Keep change if score improves; revert otherwise
  6. Repeat for N queries

Scoring: Token-position based. We stream 20 tokens from Ollama at temp=0.1
and find the first dog-related word. Score = position (0-20). If no dog word
appears, score = 20 (success).

Usage:
    from attacks.square_attack import SquareAttack, ollama_score

    attack = SquareAttack(host="http://127.0.0.1:11435")
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
# DOG-RELATED KEYWORDS (for scoring)
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


def ollama_score(pil_img, host="http://127.0.0.1:11435",
                 num_predict=20, retries=2):
    """
    Score an image by querying Ollama.

    Returns:
        score: int 0..num_predict. Higher = better for attacker.
               num_predict means no dog keyword found (success).
        text:   str  full text response.
        tokens: list of token strings.
    """
    for attempt in range(retries + 1):
        try:
            tokens = ollama_query(
                pil_img, host=host, num_predict=num_predict
            )
            break
        except Exception:
            if attempt == retries:
                return 0, "<error>", []
            time.sleep(1)

    # Find first dog-related token
    dog_pos = num_predict  # default: not found
    accumulated = ""
    for i, tok in enumerate(tokens):
        accumulated += tok
        lower = accumulated.lower()
        for kw in DOG_KEYWORDS:
            if kw in lower:
                # Check that the keyword appeared in the last few tokens
                # (not from much earlier)
                recent = "".join(tokens[max(0, i - 3):i + 1]).lower()
                if kw in recent:
                    dog_pos = i
                    break
        if dog_pos < num_predict:
            break

    text = "".join(tokens)
    return dog_pos, text, tokens


# ============================================================
# SQUARE ATTACK
# ============================================================

class SquareAttack:
    """
    Square Attack on Ollama moondream (black-box, query-based).

    The attack modifies square-shaped regions of the image, pushing
    pixels to the ±epsilon boundary. Each candidate is scored by
    querying Ollama and checking how early "dog" appears in the output.
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
        """Convert [H,W,3] float array in [0,1] to PIL Image."""
        return Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        )

    def _random_square(self, H, W, min_size=3, max_size=None):
        """Pick a random square region within the image."""
        if max_size is None:
            max_size = min(H, W) // 2

        size = self.rng.randint(min_size, max_size)
        size = min(size, H, W)
        y0 = self.rng.randint(0, H - size)
        x0 = self.rng.randint(0, W - size)
        return y0, x0, size

    def _apply_square(self, arr, clean_arr, epsilon, y0, x0, size, direction):
        """
        Apply a square perturbation: push pixels to ±epsilon boundary.

        direction: [3] array of +1/-1 per channel.
        """
        result = arr.copy()
        lower = np.clip(clean_arr - epsilon, 0, 1)
        upper = np.clip(clean_arr + epsilon, 0, 1)

        for c in range(3):
            if direction[c] > 0:
                result[y0:y0+size, x0:x0+size, c] = upper[
                    y0:y0+size, x0:x0+size, c
                ]
            else:
                result[y0:y0+size, x0:x0+size, c] = lower[
                    y0:y0+size, x0:x0+size, c
                ]

        return np.clip(result, 0, 1)

    def attack(self, clean_pil, epsilon=8/255, queries=5000,
               init_arr=None, verbose=True, early_stop=True):
        """
        Run the Square Attack.

        Args:
            clean_pil:   PIL Image (clean image to attack).
            epsilon:     L-inf budget in [0, 1].
            queries:     Maximum number of Ollama queries.
            init_arr:    Optional initial perturbation [H,W,3] in [0,1].
                         If None, start from clean image.
            verbose:     Print progress.
            early_stop:  Stop early if score reaches num_predict.

        Returns:
            adv_pil:     Best adversarial PIL Image found.
            info:        Dict with attack history and metadata.
        """
        self.query_count = 0
        self.history = []

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape

        if init_arr is not None:
            arr = np.clip(init_arr, 0, 1)
            arr = np.clip(arr, clean_arr - epsilon, clean_arr + epsilon)
            arr = np.clip(arr, 0, 1)
        else:
            arr = clean_arr.copy()

        # Score the initial (clean) image
        pil = self._pil_from_array(arr)
        score, text, _ = ollama_score(
            pil, host=self.host, num_predict=self.num_predict
        )
        self.query_count += 1
        best_score = score
        best_arr = arr.copy()
        best_text = text

        if verbose:
            print(f"  [init] score={best_score}/{self.num_predict} "
                  f"| {best_text[:100]}")

        self.history.append({
            "query": 0, "score": best_score, "text": best_text[:200],
            "improved": False,
        })

        if early_stop and best_score >= self.num_predict:
            if verbose:
                print("  Already no dog keyword! Stopping.")
            return self._pil_from_array(best_arr), {
                "history": self.history, "best_score": best_score,
                "queries": self.query_count, "epsilon": epsilon,
            }

        no_improve_count = 0

        for q in range(queries):
            # Adaptive square size: start large, shrink over time
            progress = q / queries
            max_size = max(3, int(min(H, W) * (0.5 - 0.35 * progress)))
            min_size = max(2, int(min(H, W) * (0.03 + 0.05 * progress)))

            y0, x0, size = self._random_square(
                H, W, min_size=min_size, max_size=max_size
            )

            # Random direction per channel
            direction = self.np_rng.choice([-1, 1], size=3)

            candidate = self._apply_square(
                best_arr, clean_arr, epsilon, y0, x0, size, direction
            )

            pil = self._pil_from_array(candidate)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1

            improved = score > best_score

            if improved:
                best_arr = candidate
                best_score = score
                best_text = text
                no_improve_count = 0
                marker = " *** IMPROVED ***"
            else:
                no_improve_count += 1
                marker = ""

            self.history.append({
                "query": q + 1, "score": score, "text": text[:200],
                "improved": improved,
            })

            if verbose and (improved or (q + 1) % 100 == 0):
                print(f"  [{q+1:5d}/{queries}] score={score}/{self.num_predict} "
                      f"best={best_score}/{self.num_predict} "
                      f"no_improve={no_improve_count}{marker}")

            if early_stop and best_score >= self.num_predict:
                if verbose:
                    print(f"  SUCCESS! No dog keyword at query {q+1}.")
                    print(f"  Text: {best_text[:200]}")
                break

        if verbose:
            print(f"\n  Final: score={best_score}/{self.num_predict} "
                  f"queries={self.query_count}")
            print(f"  Text: {best_text[:200]}")

        return self._pil_from_array(best_arr), {
            "history": self.history,
            "best_score": best_score,
            "queries": self.query_count,
            "epsilon": epsilon,
            "best_text": best_text,
        }


# ============================================================
# SPSA GRADIENT ESTIMATION (Fallback)
# ============================================================

def spsa_attack(clean_pil, epsilon=8/255, queries=2000,
                host="http://127.0.0.1:11435",
                num_predict=20, alpha=None, seed=42, verbose=True):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) attack.

    Estimates the gradient of the score function using random perturbations,
    then takes a PGD step in the estimated gradient direction.

    More query-efficient than random search when the gradient is informative.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
    H, W, C = clean_arr.shape

    if alpha is None:
        alpha = epsilon / 4

    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)

    # Start from clean
    arr = clean_arr.copy()

    # Initial score
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    score, text, _ = ollama_score(pil, host=host, num_predict=num_predict)
    best_score = score
    best_arr = arr.copy()
    query_count = 1

    if verbose:
        print(f"  [init] score={best_score}/{num_predict} | {text[:80]}")

    for q in range(queries):
        # SPSA: sample random direction
        delta = np_rng.choice([-1, 1], size=(H, W, C)).astype(np.float32)

        # Perturbation magnitude for gradient estimation
        c = epsilon / 3.0

        # Plus perturbation
        plus = np.clip(arr + c * delta, lower, upper)
        pil_plus = Image.fromarray((plus * 255).astype(np.uint8))
        score_plus, _, _ = ollama_score(
            pil_plus, host=host, num_predict=num_predict
        )

        # Minus perturbation
        minus = np.clip(arr - c * delta, lower, upper)
        pil_minus = Image.fromarray((minus * 255).astype(np.uint8))
        score_minus, _, _ = ollama_score(
            pil_minus, host=host, num_predict=num_predict
        )

        query_count += 2

        # Gradient estimate
        grad_est = (score_plus - score_minus) / (2 * c) * delta

        # PGD step: maximize score (go in gradient direction)
        step = alpha * np.sign(grad_est)
        arr = np.clip(arr + step, lower, upper)
        arr = np.clip(arr, 0, 1)

        # Evaluate every 10 iterations
        if (q + 1) % 10 == 0 or q == 0:
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            score, text, _ = ollama_score(
                pil, host=host, num_predict=num_predict
            )
            query_count += 1

            if score > best_score:
                best_score = score
                best_arr = arr.copy()
                marker = " *** IMPROVED ***"
            else:
                marker = ""

            if verbose:
                print(f"  [{q+1:5d}/{queries}] score={score}/{num_predict} "
                      f"best={best_score}/{num_predict} "
                      f"queries={query_count}{marker}")

            if best_score >= num_predict:
                if verbose:
                    print(f"  SUCCESS at query {query_count}!")
                    print(f"  Text: {text[:200]}")
                break

    return Image.fromarray((best_arr * 255).astype(np.uint8)), {
        "best_score": best_score,
        "queries": query_count,
        "epsilon": epsilon,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(
        description="Square Attack: Black-box adversarial attack on Ollama"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_square.png")
    parser.add_argument("--method", choices=["square", "spsa"],
                        default="square")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378,
                        help="Resize image to this size for attack")
    args = parser.parse_args()

    # Load image
    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Method: {args.method}")
    print(f"Max queries: {args.queries}")
    print()

    if args.method == "square":
        attack = SquareAttack(
            seed=args.seed, num_predict=args.num_predict
        )
        adv_pil, info = attack.attack(
            pil, epsilon=args.epsilon, queries=args.queries, verbose=True
        )
    else:
        adv_pil, info = spsa_attack(
            pil, epsilon=args.epsilon, queries=args.queries,
            num_predict=args.num_predict, seed=args.seed, verbose=True
        )

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")

    # Verify with full description
    print("\n=== VERIFICATION (full 200-token description) ===")
    score, text, _ = ollama_score(
        adv_pil, host="http://127.0.0.1:11435", num_predict=200
    )
    has_dog = any(kw in text.lower() for kw in DOG_KEYWORDS)
    print(f"Dog keyword present: {has_dog}")
    print(f"Score: {score}/200")
    print(f"Description: {text[:300]}")

    # L-inf check
    clean_arr = np.array(pil, dtype=np.float32) / 255.0
    adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
    linf = np.abs(adv_arr - clean_arr).max()
    print(f"\nL-inf: {linf:.8f} (budget: {args.epsilon:.8f})")
    print(f"Within budget: {linf <= args.epsilon + 1e-6}")
