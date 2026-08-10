"""
Square Attack v2 against GLM-5V-Turbo (Zhipu BigModel) as oracle.

Improvements over v1
--------------------
* **Targeted mode** (``--target cat|bird|car|...``). Bonus score when
  the response contains any non-dog animal/object word, so the optimiser
  has a real gradient signal instead of just "no dog".
* **Six phases** instead of three:
    0. Big-block attack (large coarse squares)
    1. Random full-image noise (density sweep)
    2. Multi-square with shrinking size
    3. Color / HSV jitter attack
    4. Momentum multi-square
    5. SPSA gradient estimation (fine-tuning)
* **Perturbation mask persistence** — successful squares are stamped
  onto a persistent perturbation buffer, so good edits survive the
  next round instead of being resampled from scratch.
* **Larger default budget** (400 queries, ε = 16/255) — real GLM-5V-Turbo
  needs more queries than ResNet50 to crack; 30 queries are not enough.
* **Adaptive score**:
    - SUCCESS_SCORE when no dog keyword is found
    - TARGET_BONUS  when a target keyword (e.g. "cat") is found
    - otherwise the character-position score

Usage
-----
    export ZHIPU_API_KEY=...
    PYTHONPATH=. python attacks/square_attack_glm5.py --target cat --queries 400
    PYTHONPATH=. python attacks/square_attack_glm5.py --target bird --epsilon 16/255
    PYTHONPATH=. python attacks/square_attack_glm5.py --mock-host http://127.0.0.1:11880 --queries 60
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from models.api_adapter import DEFAULT_KEYWORDS, SUCCESS_SCORE
from models.frontier_adapter import (
    GLM5Adapter,
    character_score,
)


# ============================================================
# SCORING
# ============================================================

# Words we want to elicit from the model (anti-targets).
# These earn TARGET_BONUS in the score when present in the response.
ANTI_TARGET_GROUPS = {
    "cat":   ["cat", "kitten", "feline", "tabby", "siamese", "persian"],
    "bird":  ["bird", "parrot", "sparrow", "robin", "pigeon", "eagle",
              "hawk", "owl", "crow", "duck", "goose", "chicken"],
    "car":   ["car", "vehicle", "automobile", "sedan", "truck", "suv",
              "toyota", "honda", "tesla"],
    "tree":  ["tree", "forest", "palm", "oak", "pine", "plant", "leafy"],
    "food":  ["food", "pizza", "burger", "sandwich", "bread", "fruit",
              "cake", "meal", "dish"],
    "random":["abstract", "texture", "pattern", "blur", "noise", "shape",
              "object", "artifact", "design", "colorful"],
}

TARGET_BONUS = 50.0  # bonus per match of a target keyword


def contains_any(text: str, words) -> bool:
    text = (text or "").lower()
    return any(w in text for w in words)


def contains_dog(text: str) -> bool:
    return contains_any(text, DEFAULT_KEYWORDS)


def anti_target_score(text: str, target_group: Optional[str]) -> float:
    """Return TARGET_BONUS if any anti-target keyword appears in ``text``,
    else 0. If ``target_group`` is None, returns 0."""
    if target_group is None or target_group not in ANTI_TARGET_GROUPS:
        return 0.0
    if contains_any(text, ANTI_TARGET_GROUPS[target_group]):
        return TARGET_BONUS
    return 0.0


# ============================================================
# ATTACK
# ============================================================

class GLM5SquareAttackV2:
    """Aggressive Square Attack against GLM-5V-Turbo.

    Maintains two arrays:
      * ``best_arr``  – current best adversarial image in [0,1]
      * ``delta_arr`` – persistent perturbation mask (residual w.r.t. clean)
    Successful candidates contribute their delta to ``delta_arr``,
    so good edits accumulate over time.
    """

    def __init__(self, adapter: GLM5Adapter, image_size: int = 378,
                 seed: int = 42, sleep_s: float = 0.5,
                 max_tokens: int = 30, target_group: Optional[str] = None,
                 verbose: bool = True):
        self.adapter = adapter
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.sleep_s = sleep_s
        self.max_tokens = max_tokens
        self.target_group = target_group
        self.api_queries = 0
        self.history = []

    # --------------------------------------------------------
    # IMAGE HELPERS
    # --------------------------------------------------------

    def _pil_from_array(self, arr: np.ndarray) -> Image.Image:
        return Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        )

    def _query(self, arr: np.ndarray) -> Tuple[float, str, float]:
        """One API call -> (char_score, text, target_bonus)."""
        pil = self._pil_from_array(arr)
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize((self.image_size, self.image_size),
                             Image.LANCZOS)
        text = self.adapter.query(pil, max_tokens=self.max_tokens)
        self.api_queries += 1
        char = character_score(text)
        bonus = anti_target_score(text, self.target_group)
        return char, text, bonus

    def _combined_score(self, char: float, bonus: float) -> float:
        return char + bonus

    def _log(self, score: float, text: str, phase: str, note: str = "",
             char: float = 0.0, bonus: float = 0.0) -> None:
        self.history.append({
            "query": self.api_queries,
            "score": score,
            "char_score": char,
            "target_bonus": bonus,
            "text": text[:200],
            "phase": phase,
            "note": note,
            "contains_dog": contains_dog(text),
            "target_hit": bonus > 0,
        })

    # --------------------------------------------------------
    # PERTURBATION GENERATORS
    # Each returns ``new_arr`` clipped to [0,1] and to the ε-ball.
    # --------------------------------------------------------

    def _clip_to_ball(self, arr: np.ndarray, clean: np.ndarray,
                      eps: float) -> np.ndarray:
        lower = np.clip(clean - eps, 0, 1)
        upper = np.clip(clean + eps, 0, 1)
        return np.clip(np.clip(arr, lower, upper), 0, 1)

    def _random_full_noise(self, clean: np.ndarray, eps: float,
                           density: float) -> np.ndarray:
        H, W, C = clean.shape
        noise = self.np_rng.choice(
            [-eps, 0.0, eps], size=(H, W, C),
            p=[density / 2, 1 - density, density / 2],
        ).astype(np.float32)
        return np.clip(clean + noise, 0, 1)

    def _multi_square(self, base: np.ndarray, clean: np.ndarray,
                      eps: float, num_squares: int,
                      min_size: int, max_size: int) -> np.ndarray:
        H, W, C = base.shape
        lower = np.clip(clean - eps, 0, 1)
        upper = np.clip(clean + eps, 0, 1)
        result = base.copy()
        for _ in range(num_squares):
            size = self.rng.randint(min_size, max_size)
            size = min(size, H, W)
            if size <= 0:
                continue
            y0 = self.rng.randint(0, H - size)
            x0 = self.rng.randint(0, W - size)
            direction = self.np_rng.choice([-1, 1], size=C)
            for c in range(C):
                if direction[c] > 0:
                    result[y0:y0 + size, x0:x0 + size, c] = \
                        upper[y0:y0 + size, x0:x0 + size, c]
                else:
                    result[y0:y0 + size, x0:x0 + size, c] = \
                        lower[y0:y0 + size, x0:x0 + size, c]
        return np.clip(result, 0, 1)

    def _big_block_attack(self, clean: np.ndarray, eps: float) -> np.ndarray:
        """Phase 0: a single big block covering 30-60% of the image."""
        H, W, C = clean.shape
        result = clean.copy()
        frac = self.rng.uniform(0.3, 0.6)
        # Random rectangular region
        rh = max(8, int(H * frac * self.rng.uniform(0.5, 1.0)))
        rw = max(8, int(W * frac * self.rng.uniform(0.5, 1.0)))
        y0 = self.rng.randint(0, max(1, H - rh))
        x0 = self.rng.randint(0, max(1, W - rw))
        direction = self.np_rng.choice([-1, 1], size=C)
        lower = np.clip(clean - eps, 0, 1)
        upper = np.clip(clean + eps, 0, 1)
        for c in range(C):
            if direction[c] > 0:
                result[y0:y0 + rh, x0:x0 + rw, c] = \
                    upper[y0:y0 + rh, x0:x0 + rw, c]
            else:
                result[y0:y0 + rh, x0:x0 + rw, c] = \
                    lower[y0:y0 + rh, x0:x0 + rw, c]
        return np.clip(result, 0, 1)

    def _color_jitter(self, base: np.ndarray, clean: np.ndarray,
                      eps: float, strength: float = 0.5) -> np.ndarray:
        """Phase 3: shift HSV channels uniformly across the whole image."""
        # Convert to HSV in [0,1], shift H/S/V, convert back
        pil = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
        hsv = pil.convert("HSV")
        arr = np.asarray(hsv, dtype=np.float32) / 255.0  # H,S,V in [0,1]
        h_shift = self.rng.uniform(-0.10, 0.10) * strength
        s_shift = self.rng.uniform(-0.20, 0.20) * strength
        v_shift = self.rng.uniform(-0.20, 0.20) * strength
        arr[..., 0] = (arr[..., 0] + h_shift) % 1.0
        arr[..., 1] = np.clip(arr[..., 1] + s_shift, 0, 1)
        arr[..., 2] = np.clip(arr[..., 2] + v_shift, 0, 1)
        out = Image.fromarray((arr * 255).astype(np.uint8), mode="HSV").convert("RGB")
        out_arr = np.asarray(out, dtype=np.float32) / 255.0
        return self._clip_to_ball(out_arr, clean, eps)

    def _spsa_step(self, base: np.ndarray, clean: np.ndarray,
                   eps: float, c: float):
        H, W, C = base.shape
        delta = self.np_rng.choice([-1.0, 1.0], size=(H, W, C)).astype(np.float32)
        plus = self._clip_to_ball(base + c * delta, clean, eps)
        minus = self._clip_to_ball(base - c * delta, clean, eps)
        c_plus, t_plus, b_plus = self._query(plus)
        c_minus, t_minus, b_minus = self._query(minus)
        s_plus = self._combined_score(c_plus, b_plus)
        s_minus = self._combined_score(c_minus, b_minus)
        if abs(s_plus - s_minus) < 1e-3:
            return None
        grad = (s_plus - s_minus) / (2.0 * c) * delta
        step = c * 2.0 * np.sign(grad)
        cand = self._clip_to_ball(base + step, clean, eps)
        return cand

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def attack(self, clean_pil: Image.Image, epsilon: float = 16 / 255,
               queries: int = 400, early_stop: bool = True
               ) -> Tuple[Image.Image, dict]:
        clean_pil = clean_pil.convert("RGB")
        clean_pil = clean_pil.resize((self.image_size, self.image_size),
                                     Image.LANCZOS)
        clean = np.asarray(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean.shape

        if self.verbose:
            tgt = f" target={self.target_group}" if self.target_group else " untargeted"
            print(f"[glm5-square-v2] image={H}x{W}  "
                  f"epsilon={epsilon:.5f} ({epsilon * 255:.1f}/255)  "
                  f"budget={queries}{tgt}")

        # Persistent perturbation buffer
        delta = np.zeros_like(clean)

        # Baseline
        best_char, best_text, best_bonus = self._query(clean)
        best_score = self._combined_score(best_char, best_bonus)
        best_arr = clean.copy()
        self._log(best_score, best_text, "init", "", best_char, best_bonus)
        if self.verbose:
            print(f"  [init  q={self.api_queries:4d}] "
                  f"score={best_score:7.2f} "
                  f"(char={best_char:6.1f}, bonus={best_bonus:4.1f}) "
                  f"dog={contains_dog(best_text)} | {best_text[:80]!r}")

        def done():
            # success = no dog keyword AND (target hit OR score >= SUCCESS_SCORE)
            no_dog = not contains_dog(best_text)
            if not no_dog:
                return False
            if self.target_group:
                return best_bonus > 0
            return best_score >= SUCCESS_SCORE

        if done():
            if self.verbose:
                print("  [init] already successful.")
            return self._pil_from_array(best_arr), self._info(
                clean, best_arr, best_char, best_bonus, best_text,
                epsilon, queries,
            )

        # Phase budget allocation
        n_p0 = max(1, int(queries * 0.08))
        n_p1 = max(1, int(queries * 0.17))
        n_p2 = max(1, int(queries * 0.30))
        n_p3 = max(1, int(queries * 0.10))
        n_p4 = max(1, int(queries * 0.15))
        n_p5 = queries - (n_p0 + n_p1 + n_p2 + n_p3 + n_p4)
        n_p5 = max(1, n_p5)

        # ------------------ Phase 0: big blocks ------------------
        if self.verbose:
            print(f"\n  Phase 0: big-block ({n_p0} queries)")
        for _ in range(n_p0):
            if self.api_queries >= queries:
                break
            cand = self._big_block_attack(clean, epsilon)
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = "big_block"
            if score > best_score:
                # Accumulate the delta into persistent buffer
                delta = np.clip(cand - clean, -epsilon, epsilon)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "block", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [blk  q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 0 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            time.sleep(self.sleep_s)

        # ------------------ Phase 1: full-image noise ------------------
        if self.verbose:
            print(f"\n  Phase 1: random full-image noise ({n_p1} queries)")
        densities = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
        for q in range(n_p1):
            if self.api_queries >= queries:
                break
            density = densities[q % len(densities)]
            cand = self._random_full_noise(clean, epsilon, density)
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = f"density={density:.2f}"
            if score > best_score:
                delta = np.clip(cand - clean, -epsilon, epsilon)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "noise", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [nois q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 1 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            time.sleep(self.sleep_s)

        # ------------------ Phase 2: multi-square shrinking ------------------
        if self.verbose:
            print(f"\n  Phase 2: multi-square shrinking ({n_p2} queries)")
        for q in range(n_p2):
            if self.api_queries >= queries:
                break
            progress = q / max(1, n_p2 - 1)
            max_size = max(5, int(min(H, W) * (0.5 - 0.4 * progress)))
            min_size = max(3, int(min(H, W) * (0.05 + 0.10 * progress)))
            min_size = min(min_size, max_size)
            num_sq = self.rng.randint(1, 4)
            cand = self._multi_square(best_arr, clean, epsilon,
                                      num_sq, min_size, max_size)
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = f"sq={num_sq} sz={min_size}-{max_size}"
            if score > best_score:
                delta = np.clip(cand - clean, -epsilon, epsilon)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "square", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [sq   q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 2 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            time.sleep(self.sleep_s)

        # ------------------ Phase 3: color jitter ------------------
        if self.verbose:
            print(f"\n  Phase 3: HSV color jitter ({n_p3} queries)")
        for q in range(n_p3):
            if self.api_queries >= queries:
                break
            strength = 0.3 + 0.7 * (q / max(1, n_p3 - 1))
            cand = self._color_jitter(best_arr, clean, epsilon, strength)
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = f"strength={strength:.2f}"
            if score > best_score:
                delta = np.clip(cand - clean, -epsilon, epsilon)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "color", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [col  q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 3 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            time.sleep(self.sleep_s)

        # ------------------ Phase 4: momentum multi-square ------------------
        if self.verbose:
            print(f"\n  Phase 4: momentum multi-square ({n_p4} queries)")
        momentum = np.zeros_like(clean)
        for q in range(n_p4):
            if self.api_queries >= queries:
                break
            progress = q / max(1, n_p4 - 1)
            max_size = max(5, int(min(H, W) * (0.30 - 0.25 * progress)))
            min_size = max(3, int(min(H, W) * (0.03 + 0.07 * progress)))
            min_size = min(min_size, max_size)
            num_sq = self.rng.randint(2, 6)
            cand = self._multi_square(best_arr, clean, epsilon,
                                      num_sq, min_size, max_size)
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = f"sq={num_sq} sz={min_size}-{max_size}"
            if score > best_score:
                # Update momentum in the direction of improvement
                delta = np.clip(cand - clean, -epsilon, epsilon)
                new_delta = 0.7 * delta + 0.3 * (cand - best_arr)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "momentum", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [mom  q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 4 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            time.sleep(self.sleep_s)

        # ------------------ Phase 5: SPSA ------------------
        c = epsilon / 4.0
        if self.verbose:
            print(f"\n  Phase 5: SPSA (up to {queries - self.api_queries} queries)")
        while self.api_queries < queries:
            cand = self._spsa_step(best_arr, clean, epsilon, c)
            if cand is None:
                continue
            char, text, bonus = self._query(cand)
            score = self._combined_score(char, bonus)
            note = f"spsa_c={c:.4f}"
            if score > best_score:
                delta = np.clip(cand - clean, -epsilon, epsilon)
                best_arr, best_score, best_char, best_bonus, best_text = (
                    cand, score, char, bonus, text,
                )
                note += " IMPROVED"
            self._log(score, text, "spsa", note, char, bonus)
            if self.verbose and note.endswith("IMPROVED"):
                print(f"  [spsa q={self.api_queries:4d}] "
                      f"score={score:7.2f} best={best_score:7.2f} "
                      f"{note} | {text[:70]!r}")
            if early_stop and done():
                if self.verbose:
                    print(f"  SUCCESS in Phase 5 at q={self.api_queries}")
                return self._pil_from_array(best_arr), self._info(
                    clean, best_arr, best_char, best_bonus, best_text,
                    epsilon, queries,
                )
            if self.api_queries % 10 == 0:
                c *= 0.97
            time.sleep(self.sleep_s)

        if self.verbose:
            print(f"\n  Done. best_score={best_score:.2f} "
                  f"(char={best_char:.1f}, bonus={best_bonus:.1f}) "
                  f"queries={self.api_queries} "
                  f"dog={contains_dog(best_text)}")
            print(f"  Text: {best_text[:200]}")
        return self._pil_from_array(best_arr), self._info(
            clean, best_arr, best_char, best_bonus, best_text, epsilon, queries,
        )

    # --------------------------------------------------------

    def _info(self, clean, best_arr, char, bonus, text,
              epsilon, queries) -> dict:
        linf = float(np.abs(best_arr - clean).max())
        score = self._combined_score(char, bonus)
        return {
            "model": self.adapter.model_name,
            "host": self.adapter.host,
            "epsilon": epsilon,
            "linf": linf,
            "best_score": score,
            "best_char_score": char,
            "best_target_bonus": bonus,
            "best_text": text,
            "contains_dog": contains_dog(text),
            "target_group": self.target_group,
            "queries_used": self.api_queries,
            "queries_budget": queries,
            "history": self.history,
        }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=("Square Attack v2 against GLM-5V-Turbo (Zhipu). "
                     "Six phases; supports targeted mode."),
    )
    parser.add_argument("--image", default="dog.jpg",
                        help="Clean image to attack (default: dog.jpg).")
    parser.add_argument("--output", default="outputs/adv_glm5_dog.png",
                        help="Where to save the adversarial image.")
    parser.add_argument("--epsilon", type=float, default=16 / 255,
                        help="L-inf budget in [0,1] (default: 16/255).")
    parser.add_argument("--queries", type=int, default=400,
                        help="Max API calls (default: 400).")
    parser.add_argument("--image-size", type=int, default=378)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between API calls.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=list(ANTI_TARGET_GROUPS.keys()),
                        default="cat",
                        help="Anti-target group to elicit. Default 'cat'. "
                             "Pass empty string to disable targeted mode.")
    parser.add_argument("--api-key", default=os.environ.get("ZHIPU_API_KEY"))
    parser.add_argument("--mock-host", default=os.environ.get("FRONTIER_MOCK_HOST"),
                        help="Use a local mock GLM server (e.g. "
                             "http://127.0.0.1:11880) instead of the real API.")
    parser.add_argument("--report", default=None,
                        help="Optional JSON path for full attack report.")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)
    if not args.api_key and not args.mock_host:
        print("ERROR: set ZHIPU_API_KEY or pass --mock-host.")
        sys.exit(1)

    target_group = args.target if args.target else None

    if args.mock_host:
        adapter = GLM5Adapter(
            name="glm-5v-turbo",
            host=args.mock_host,
            api_key="mock-key",
            image_size=args.image_size,
        )
    else:
        adapter = GLM5Adapter(
            name="glm-5v-turbo",
            api_key=args.api_key,
            image_size=args.image_size,
        )

    clean_pil = Image.open(args.image).convert("RGB")
    print(f"Image  : {args.image}  -> {args.image_size}x{args.image_size}")
    print(f"Model  : {adapter.model_name}  host={adapter.host}")
    print(f"Budget : {args.queries} queries, "
          f"epsilon={args.epsilon:.5f} ({args.epsilon * 255:.1f}/255)")
    print(f"Target : {target_group or '(untargeted)'}")
    print()

    attack = GLM5SquareAttackV2(
        adapter=adapter,
        image_size=args.image_size,
        seed=args.seed,
        sleep_s=args.sleep,
        max_tokens=args.max_tokens,
        target_group=target_group,
    )
    adv_pil, info = attack.attack(
        clean_pil,
        epsilon=args.epsilon,
        queries=args.queries,
        early_stop=not args.no_early_stop,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)

    # Verify with a fresh, longer query
    print("\n=== VERIFICATION (longer max_tokens) ===")
    verify_pil = adv_pil.resize(
        (args.image_size, args.image_size), Image.LANCZOS,
    )
    verify_text = adapter.query(verify_pil, max_tokens=120)
    verify_has_dog = contains_dog(verify_text)
    verify_bonus = anti_target_score(verify_text, target_group)
    verify_score = character_score(verify_text) + verify_bonus
    no_dog = not verify_has_dog
    success = no_dog and (verify_bonus > 0 if target_group else verify_score >= SUCCESS_SCORE)
    label = ("SUCCESS" if success else
             ("PARTIAL (no dog but no target either)" if no_dog else "FAIL"))
    print(f"  Score       : {verify_score:.2f}  [{label}]")
    print(f"  Dog keyword : {verify_has_dog}")
    if target_group:
        print(f"  Target hit  : {verify_bonus > 0}  ({verify_bonus:.1f})")
    print(f"  Response    : {verify_text[:300]!r}")
    print(f"  L-inf       : {info['linf']:.6f} "
          f"(budget {args.epsilon:.6f}, within: "
          f"{info['linf'] <= args.epsilon + 1e-6})")
    print(f"  Saved       : {args.output}")

    info["verification_text"] = verify_text
    info["verification_has_dog"] = verify_has_dog
    info["verification_target_bonus"] = verify_bonus
    info["verification_score"] = verify_score

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"  Report      : {args.report}")


if __name__ == "__main__":
    main()
