"""
Universal VLM Attack Demo

Demonstrates the full framework: load any VLM, run white-box or black-box
attack, evaluate results.

Usage:
    # White-box attack on LLaVA-1.5-7B
    PYTHONPATH=. python experiments/universal_attack_demo.py \
        --model llava-1.5-7b --mode whitebox \
        --image data/vlm/dog03.jpg --target "A cat sitting on a couch"

    # Black-box attack on Qwen 35B via API
    PYTHONPATH=. python experiments/universal_attack_demo.py \
        --model qwen-vyas --mode blackbox \
        --image data/vlm/dog03.jpg --epsilon 0.125 --queries 300

    # List available models
    PYTHONPATH=. python experiments/universal_attack_demo.py --list
"""

import sys
import os
import time
import argparse
import json
import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="Universal VLM Attack Demo"
    )
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    parser.add_argument("--model", default="llava-1.5-7b",
                        help="Model name (see --list)")
    parser.add_argument("--mode", default="whitebox",
                        choices=["whitebox", "blackbox"],
                        help="Attack mode")
    parser.add_argument("--image", default="data/vlm/dog03.jpg")
    parser.add_argument("--target", default="A cat sitting on a couch")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=1 / 255)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--queries", type=int, default=300,
                        help="Max queries (black-box only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/universal_demo")
    args = parser.parse_args()

    if args.list:
        from models.vlm_registry import list_models
        list_models()
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("UNIVERSAL VLM ATTACK DEMO")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Mode:       {args.mode}")
    print(f"  Image:      {args.image}")
    print(f"  Target:     \"{args.target}\"")
    print(f"  Epsilon:    {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")
    if args.mode == "whitebox":
        print(f"  Iterations: {args.iterations}")
        print(f"  Momentum:   {args.momentum}")
    else:
        print(f"  Max queries: {args.queries}")
    print()

    # Load model
    print("[1/5] Loading model...")
    from models.vlm_registry import get_vlm
    adapter = get_vlm(args.model, mode=args.mode)
    print(f"  Loaded: {adapter.name}")

    # Load image
    print("\n[2/5] Loading image...")
    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((adapter.image_size, adapter.image_size), Image.LANCZOS)
    print(f"  Size: {adapter.image_size}x{adapter.image_size}")

    # Clean baseline
    print("\n[3/5] Clean description:")
    clean_desc = adapter.describe(pil, max_tokens=100)
    print(f"  {clean_desc[:200]}")

    # Run attack
    print(f"\n[4/5] Running {'white-box PGD' if args.mode == 'whitebox' else 'black-box SPSA'} attack...")

    if args.mode == "whitebox":
        from attacks.vlm_pgd_universal import UniversalPGDAttack
        attack = UniversalPGDAttack(
            adapter=adapter,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iterations=args.iterations,
            momentum=args.momentum,
            seed=args.seed,
        )
        t0 = time.time()
        adv_pil = attack.attack_pil(pil, args.target, verbose=True)
        elapsed = time.time() - t0
    else:
        from attacks.blackbox_universal import UniversalBlackBoxAttack
        attack = UniversalBlackBoxAttack(
            adapter=adapter,
            epsilon=args.epsilon,
            queries=args.queries,
            seed=args.seed,
        )
        t0 = time.time()
        adv_pil = attack.attack(pil, verbose=True)
        elapsed = time.time() - t0

    print(f"\n  Attack completed in {elapsed:.1f}s")

    # Evaluate
    print("\n[5/5] Adversarial description:")
    adv_desc = adapter.describe(adv_pil, max_tokens=200)
    print(f"  {adv_desc[:300]}")

    # Score
    target_words = args.target.lower().split()
    adv_lower = adv_desc.lower()
    target_matches = sum(1 for w in target_words if w in adv_lower)
    target_pct = target_matches / len(target_words) * 100

    clean_lower = clean_desc.lower()
    clean_target_matches = sum(1 for w in target_words if w in clean_lower)

    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Clean output:      {clean_desc[:100]}")
    print(f"  Adversarial output: {adv_desc[:100]}")
    print(f"  Target:            \"{args.target}\"")
    print(f"  Target words in adv: {target_matches}/{len(target_words)} ({target_pct:.0f}%)")
    print(f"  Target words in clean: {clean_target_matches}/{len(target_words)}")
    print(f"  Attack time:       {elapsed:.1f}s")
    print(f"  Epsilon:           {args.epsilon:.6f} ({args.epsilon * 255:.1f}/255)")

    # Save
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    adv_path = os.path.join(args.output_dir, f"adv_{img_name}_{args.model}.png")
    adv_pil.save(adv_path)
    print(f"\n  Saved: {adv_path}")

    # Save results
    results = {
        "model": args.model,
        "mode": args.mode,
        "image": args.image,
        "target": args.target,
        "epsilon": args.epsilon,
        "clean_desc": clean_desc,
        "adv_desc": adv_desc,
        "target_word_matches": target_matches,
        "target_word_total": len(target_words),
        "target_pct": target_pct,
        "attack_time_s": elapsed,
        "iterations": args.iterations if args.mode == "whitebox" else None,
        "queries": getattr(adapter, 'query_count', None),
    }
    results_path = os.path.join(args.output_dir, f"results_{img_name}_{args.model}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
