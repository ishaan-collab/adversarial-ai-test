"""
Black-box attack evaluation on Ollama moondream.

Runs query-based black-box attacks on dog images and evaluates:
  1. Attack success (dog keyword removed from Ollama output)
  2. Query efficiency (queries needed for success)
  3. Perturbation budget (L-inf distance)
  4. Output text comparison (clean vs. adversarial)

Supported attacks:
  - v3:      Improved low-dim SPSA (character-position scoring, fixed SPSA)
  - lowdim:  Original low-dim SPSA
  - square:  Square Attack v2 (multi-phase)
  - genetic: Genetic algorithm with low-dim perturbation

Usage:
    PYTHONPATH=. python experiments/blackbox_eval.py --attack v3 --image dog07
    PYTHONPATH=. python experiments/blackbox_eval.py --attack v3 --image all
    PYTHONPATH=. python experiments/blackbox_eval.py --attack v3 --epsilon 0.0627
    PYTHONPATH=. python experiments/blackbox_eval.py --attack lowdim --image dog07 --queries 10000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# IMAGE DATASET
# ============================================================

IMAGES = [
    ("dog.jpg", "dog"),
    ("data/vlm/dog01.jpg", "dog01"),
    ("data/vlm/dog02.jpg", "dog02"),
    ("data/vlm/dog03.jpg", "dog03"),
    ("data/vlm/dog04.jpg", "dog04"),
    ("data/vlm/dog05.jpg", "dog05"),
    ("data/vlm/dog06.jpg", "dog06"),
    ("data/vlm/dog07.jpg", "dog07"),
    ("data/vlm/dog08.jpg", "dog08"),
    ("data/vlm/dog09.jpg", "dog09"),
    ("data/vlm/dog10.jpg", "dog10"),
]

ATTACK_SIZE = 378


# ============================================================
# ATTACK RUNNER
# ============================================================

def run_attack(attack_name, clean_pil, epsilon, queries, seed,
               num_predict=20, low_dim=32, host="http://127.0.0.1:11435"):
    """Run the specified attack and return (adv_pil, info)."""

    if attack_name == "v3":
        from attacks.blackbox_attack import BlackBoxAttack
        attack = BlackBoxAttack(
            host=host, seed=seed, num_predict=num_predict,
            low_dim=low_dim,
        )
        return attack.attack(
            clean_pil, epsilon=epsilon, queries=queries, verbose=True
        )

    elif attack_name == "lowdim":
        from attacks.lowdim_attack import LowDimAttack
        attack = LowDimAttack(
            host=host, seed=seed, num_predict=num_predict,
            low_dim=low_dim,
        )
        return attack.attack(
            clean_pil, epsilon=epsilon, queries=queries, verbose=True
        )

    elif attack_name == "square":
        from attacks.square_attack_v2 import BlackBoxAttack as SquareV2
        attack = SquareV2(
            host=host, seed=seed, num_predict=num_predict,
        )
        return attack.attack(
            clean_pil, epsilon=epsilon, queries=queries, verbose=True
        )

    elif attack_name == "genetic":
        from attacks.genetic_attack import GeneticAttack
        attack = GeneticAttack(
            host=host, seed=seed, num_predict=num_predict,
            low_dim=low_dim,
        )
        return attack.attack(
            clean_pil, epsilon=epsilon, queries=queries, verbose=True
        )

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def verify_adversarial(adv_pil, host="http://127.0.0.1:11435",
                       num_predict=200):
    """Verify adversarial image with full 200-token description."""
    from attacks.blackbox_attack import ollama_score, DOG_KEYWORDS

    score, text, _ = ollama_score(
        adv_pil, host=host, num_predict=num_predict
    )
    has_dog = any(kw in text.lower() for kw in DOG_KEYWORDS)

    return {
        "score": score,
        "text": text,
        "has_dog": has_dog,
        "success": not has_dog,
    }


def get_clean_description(clean_pil, host="http://127.0.0.1:11435"):
    """Get clean image description for comparison."""
    from attacks.blackbox_attack import ollama_score
    score, text, _ = ollama_score(clean_pil, host=host, num_predict=200)
    return text


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Black-box attack evaluation on Ollama moondream"
    )
    parser.add_argument(
        "--attack", default="v3",
        choices=["v3", "lowdim", "square", "genetic"],
        help="Attack method",
    )
    parser.add_argument(
        "--epsilon", type=float, default=8 / 255,
        help="L-inf budget (default: 8/255)",
    )
    parser.add_argument(
        "--queries", type=int, default=5000,
        help="Max Ollama queries per image",
    )
    parser.add_argument(
        "--image", default="dog07",
        help="Image name (e.g. dog07) or 'all'",
    )
    parser.add_argument(
        "--output-dir", default="outputs/blackbox_eval",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--low-dim", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select images
    if args.image == "all":
        images = IMAGES
    else:
        images = [
            (path, name) for path, name in IMAGES
            if name == args.image
        ]
        if not images:
            print(f"Image '{args.image}' not found. Available:")
            for _, name in IMAGES:
                print(f"  {name}")
            return

    print("=" * 60)
    print("Black-Box Attack Evaluation")
    print("=" * 60)
    print(f"Attack:     {args.attack}")
    print(f"Epsilon:    {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")
    print(f"Queries:    {args.queries}")
    print(f"Images:     {len(images)}")
    print(f"Output:     {args.output_dir}")
    print(f"Seed:       {args.seed}")
    print(f"Low-dim:    {args.low_dim}x{args.low_dim}")
    print()

    results = []
    total_start = time.time()

    for img_path, name in images:
        print(f"\n{'=' * 60}")
        print(f"Attacking: {name} ({img_path})")
        print(f"{'=' * 60}")

        pil = Image.open(img_path).convert("RGB")
        pil = pil.resize((ATTACK_SIZE, ATTACK_SIZE), Image.LANCZOS)

        # Get clean description
        print("Getting clean description...")
        clean_text = get_clean_description(pil)
        print(f"Clean: {clean_text[:150]}")

        # Run attack
        t0 = time.time()
        try:
            adv_pil, info = run_attack(
                args.attack, pil, args.epsilon, args.queries, args.seed,
                num_predict=args.num_predict, low_dim=args.low_dim,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "image": name,
                "path": img_path,
                "error": str(e),
                "success": False,
            })
            with open(f"{args.output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=2)
            continue

        elapsed = time.time() - t0

        # Save adversarial image
        adv_path = f"{args.output_dir}/adv_{name}.png"
        adv_pil.save(adv_path)
        print(f"\nSaved: {adv_path}")

        # Verify with full 200-token description
        print("\nVerifying with full 200-token description...")
        verification = verify_adversarial(adv_pil)

        # L-inf check
        clean_arr = np.array(pil, dtype=np.float32) / 255.0
        adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
        linf = float(np.abs(adv_arr - clean_arr).max())

        result = {
            "image": name,
            "path": img_path,
            "attack": args.attack,
            "epsilon": args.epsilon,
            "queries_used": info.get("queries", 0),
            "max_queries": args.queries,
            "attack_score": info.get("best_score", 0),
            "verify_score": verification["score"],
            "success": verification["success"],
            "has_dog": verification["has_dog"],
            "clean_text": clean_text[:300],
            "adv_text": verification["text"][:300],
            "linf": linf,
            "within_budget": linf <= args.epsilon + 1e-6,
            "elapsed": elapsed,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        print(f"\n--- Result for {name} ---")
        print(f"Success:     {result['success']}")
        print(f"Attack score: {result['attack_score']:.1f}")
        print(f"Verify score: {result['verify_score']:.1f}")
        print(f"Queries:     {result['queries_used']}")
        print(f"L-inf:       {result['linf']:.8f} "
              f"(budget: {args.epsilon:.8f})")
        print(f"Within budget: {result['within_budget']}")
        print(f"Time:        {elapsed:.1f}s")
        print(f"Clean text:  {result['clean_text'][:100]}")
        print(f"Adv text:    {result['adv_text'][:100]}")

        # Save intermediate results
        with open(f"{args.output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ============================================
    # Summary
    # ============================================
    total_elapsed = time.time() - total_start
    successes = sum(1 for r in results if r.get("success"))
    total_queries = sum(r.get("queries_used", 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Attack:    {args.attack}")
    print(f"Epsilon:   {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")
    print(f"Success:   {successes}/{len(results)}")
    print(f"Queries:   {total_queries} total")
    print(f"Time:      {total_elapsed:.1f}s total")
    print()
    print(f"{'Image':<12} {'Success':<8} {'Score':<8} {'Queries':<8} "
          f"{'L-inf':<10} {'Time':<8}")
    print("-" * 60)
    for r in results:
        status = "YES" if r.get("success") else "NO"
        print(f"{r['image']:<12} {status:<8} "
              f"{r.get('verify_score', 0):<8.1f} "
              f"{r.get('queries_used', 0):<8} "
              f"{r.get('linf', 0):<10.6f} "
              f"{r.get('elapsed', 0):<8.1f}")

    # Save final results with summary
    summary = {
        "attack": args.attack,
        "epsilon": args.epsilon,
        "max_queries": args.queries,
        "seed": args.seed,
        "total_images": len(results),
        "successes": successes,
        "total_queries": total_queries,
        "total_elapsed": total_elapsed,
        "results": results,
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
