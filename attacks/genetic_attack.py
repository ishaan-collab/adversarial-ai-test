"""
Genetic Algorithm Black-Box Attack on Ollama moondream.

Uses a population of low-dimensional perturbation patterns, evolved via
crossover and mutation. Better suited for discontinuous score landscapes
than gradient-based methods (SPSA).

Key design choices:
  - Low-dim perturbation (16x16 or 32x32, bilinearly upscaled)
  - Population of 30 individuals
  - Tournament selection + uniform crossover
  - Gaussian mutation with adaptive rate
  - Elite preservation (top 5 unchanged)
  - Continuous scoring (token position + dog count + text length)
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


DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]


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
    accumulated = ""
    for i, tok in enumerate(tokens):
        accumulated += tok
        lower = accumulated.lower()
        for kw in DOG_KEYWORDS:
            if kw in lower:
                recent = "".join(tokens[max(0, i - 3):i + 1]).lower()
                if kw in recent:
                    dog_pos = i
                    break
        if dog_pos < num_predict:
            break

    dog_count = sum(text_lower.count(kw) for kw in DOG_KEYWORDS)
    no_dog = 1 if dog_pos >= num_predict else 0

    score = (
        dog_pos * 10.0
        - dog_count * 5.0
        + no_dog * 100.0
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

    pert_max = np.abs(pert_up).max()
    if pert_max > 0:
        pert_up = pert_up / pert_max * epsilon

    result = clean_arr + pert_up
    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)
    return np.clip(result, lower, upper)


class GeneticAttack:
    """
    Genetic algorithm black-box attack.

    Population of low-dim perturbation patterns evolved via
    crossover + mutation, scored by Ollama queries.
    """

    def __init__(self, host="http://127.0.0.1:11435",
                 model="moondream", num_predict=20,
                 low_dim=16, seed=42):
        self.host = host
        self.model = model
        self.num_predict = num_predict
        self.low_dim = low_dim
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.query_count = 0

    def _pil(self, arr):
        return Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        )

    def _random_individual(self):
        """Random low-dim perturbation, Gaussian."""
        D = self.low_dim
        return self.np_rng.randn(D, D, 3).astype(np.float32) * 0.5

    def _crossover(self, parent1, parent2):
        """Uniform crossover: each element from random parent."""
        mask = self.np_rng.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2)
        return child.astype(np.float32)

    def _mutate(self, individual, rate=0.3, strength=0.5):
        """Gaussian mutation on random subset of elements."""
        result = individual.copy()
        D = self.low_dim
        num_mutations = int(rate * D * D * 3)
        for _ in range(num_mutations):
            i = self.rng.randint(0, D - 1)
            j = self.rng.randint(0, D - 1)
            c = self.rng.randint(0, 2)
            result[i, j, c] += self.np_rng.randn() * strength
        return result

    def attack(self, clean_pil, epsilon=8/255, queries=5000,
               population_size=30, elite_size=5,
               mutation_rate=0.3, mutation_strength=0.5,
               verbose=True, early_stop=True):
        """
        Run genetic algorithm attack.

        Args:
            clean_pil:       Clean PIL image.
            epsilon:         L-inf budget.
            queries:         Max Ollama queries.
            population_size: Number of individuals per generation.
            elite_size:      Number of top individuals preserved.
            mutation_rate:   Fraction of genes mutated.
            mutation_strength: Std dev of mutation noise.
            verbose:         Print progress.
            early_stop:      Stop if no dog keyword found.

        Returns:
            adv_pil:  Best adversarial PIL image.
            info:     Dict with results.
        """
        self.query_count = 0
        D = self.low_dim

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
        H, W, C = clean_arr.shape

        success_threshold = self.num_predict * 10.0 + 100.0

        # Initialize population
        population = [self._random_individual() for _ in range(population_size)]

        # Score initial population
        scored_pop = []
        for i, ind in enumerate(population):
            if self.query_count >= queries:
                break
            adv_arr = apply_perturbation(clean_arr, ind, epsilon)
            pil = self._pil(adv_arr)
            score, text, _ = ollama_score(
                pil, host=self.host, num_predict=self.num_predict
            )
            self.query_count += 1
            scored_pop.append((score, ind, text))

            if verbose and (i < 5 or score > 100):
                print(f"  [init {i+1:2d}/{population_size}] "
                      f"score={score:.1f} | {text[:80]}")

            if early_stop and score >= success_threshold:
                if verbose:
                    print(f"  SUCCESS at init {i+1}!")
                    print(f"  Text: {text[:200]}")
                adv_arr = apply_perturbation(clean_arr, ind, epsilon)
                return self._pil(adv_arr), {
                    "best_score": score, "queries": self.query_count,
                    "epsilon": epsilon, "best_text": text,
                }

        scored_pop.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_pop[0][0]
        best_ind = scored_pop[0][1]
        best_text = scored_pop[0][2]

        if verbose:
            print(f"  Initial best: score={best_score:.1f}")

        # Evolve
        generation = 0
        while self.query_count < queries:
            generation += 1

            # Create offspring
            offspring = []

            # Elites (preserved unchanged)
            for i in range(min(elite_size, len(scored_pop))):
                offspring.append(scored_pop[i][1].copy())

            # Fill rest with crossover + mutation
            while len(offspring) < population_size:
                if self.query_count >= queries:
                    break

                # Tournament selection (size 3)
                tournament = self.rng.sample(
                    scored_pop, min(3, len(scored_pop))
                )
                tournament.sort(key=lambda x: x[0], reverse=True)
                parent1 = tournament[0][1]

                tournament2 = self.rng.sample(
                    scored_pop, min(3, len(scored_pop))
                )
                tournament2.sort(key=lambda x: x[0], reverse=True)
                parent2 = tournament2[0][1]

                # Crossover
                child = self._crossover(parent1, parent2)

                # Adaptive mutation
                # Increase mutation if stuck, decrease if improving
                mut_str = mutation_strength
                mut_rate = mutation_rate

                child = self._mutate(child, rate=mut_rate, strength=mut_str)
                offspring.append(child)

            # Score offspring (only the new ones)
            scored_offspring = list(scored_pop[:elite_size])
            for i in range(elite_size, len(offspring)):
                if self.query_count >= queries:
                    break

                ind = offspring[i]
                adv_arr = apply_perturbation(clean_arr, ind, epsilon)
                pil = self._pil(adv_arr)
                score, text, _ = ollama_score(
                    pil, host=self.host, num_predict=self.num_predict
                )
                self.query_count += 1
                scored_offspring.append((score, ind, text))

                if early_stop and score >= success_threshold:
                    if verbose:
                        print(f"  SUCCESS at gen {generation}, "
                              f"query {self.query_count}!")
                        print(f"  Text: {text[:200]}")
                    adv_arr = apply_perturbation(
                        clean_arr, ind, epsilon
                    )
                    return self._pil(adv_arr), {
                        "best_score": score,
                        "queries": self.query_count,
                        "epsilon": epsilon,
                        "best_text": text,
                    }

            scored_offspring.sort(key=lambda x: x[0], reverse=True)
            scored_pop = scored_offspring[:population_size]

            new_best = scored_pop[0][0]
            improved = new_best > best_score
            if improved:
                best_score = new_best
                best_ind = scored_pop[0][1]
                best_text = scored_pop[0][2]

            if verbose:
                marker = " *** IMPROVED ***" if improved else ""
                print(f"  [gen {generation:3d}] queries={self.query_count:5d} "
                      f"best={best_score:.1f} "
                      f"top_score={new_best:.1f}{marker}")
                if improved:
                    print(f"    | {best_text[:100]}")

        # Final result
        adv_arr = apply_perturbation(clean_arr, best_ind, epsilon)
        linf = np.abs(adv_arr - clean_arr).max()

        if verbose:
            print(f"\n  Final: score={best_score:.1f} "
                  f"queries={self.query_count}")
            print(f"  L-inf: {linf:.8f} (budget: {epsilon:.8f})")
            print(f"  Text: {best_text[:200]}")

        return self._pil(adv_arr), {
            "best_score": best_score,
            "queries": self.query_count,
            "epsilon": epsilon,
            "best_text": best_text,
            "linf": float(linf),
            "generations": generation,
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Genetic Algorithm black-box attack on Ollama"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_genetic.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378)
    parser.add_argument("--low-dim", type=int, default=16)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elite", type=int, default=5)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print(f"Low-dim: {args.low_dim}x{args.low_dim}")
    print(f"Population: {args.population}, Elite: {args.elite}")
    print()

    attack = GeneticAttack(
        seed=args.seed, num_predict=args.num_predict,
        low_dim=args.low_dim,
    )
    adv_pil, info = attack.attack(
        pil, epsilon=args.epsilon, queries=args.queries,
        population_size=args.population, elite_size=args.elite,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adv_pil.save(args.output)
    print(f"\nSaved: {args.output}")

    print("\n=== VERIFICATION (200 tokens) ===")
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
