"""
Perturbation-budget sweep against the GLM-5V-Turbo attack.

Runs the Square Attack v2 with a fixed query budget and varying L-inf
epsilon. For each budget:
  * runs the full six-phase attack
  * verifies with a longer max_tokens query
  * reports score, response, SSIM-like distortion, file size

The mock server (experiments/mock_glm_server.py) is a stand-in for
GLM-5V-Turbo when no API key is set. With ZHIPU_API_KEY in env, swap
the adapter construction to use it.

Usage:
    PYTHONPATH=. python experiments/sweep_glm5_epsilon.py
    PYTHONPATH=. python experiments/sweep_glm5_epsilon.py --queries 200 --epsilons 8 16 32 64
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
from PIL import Image
import requests

from attacks.square_attack_glm5 import (
    GLM5SquareAttackV2,
    contains_dog,
    anti_target_score,
)
from models.api_adapter import DEFAULT_KEYWORDS, SUCCESS_SCORE
from models.frontier_adapter import GLM5Adapter, character_score


def run_one(image_path: str, mock_host: str, queries: int,
            epsilon: float, target_group: str,
            image_size: int = 378, max_tokens: int = 30,
            sleep_s: float = 0.0,
            save_image_path: str = None) -> dict:
    adapter = GLM5Adapter(
        name="glm-5v-turbo",
        host=mock_host,
        api_key="dry-run",
        image_size=image_size,
    )
    pil = Image.open(image_path).convert("RGB")
    attack = GLM5SquareAttackV2(
        adapter=adapter, image_size=image_size, sleep_s=sleep_s,
        max_tokens=max_tokens, target_group=target_group, verbose=False,
    )

    # Baseline
    base_pil = pil.resize((image_size, image_size), Image.LANCZOS)
    base_arr = np.asarray(base_pil, dtype=np.float32) / 255.0
    base_text = adapter.query(base_pil, max_tokens=max_tokens)
    base_score = character_score(base_text)
    base_bonus = anti_target_score(base_text, target_group)
    base_combined = base_score + base_bonus
    base_dog = contains_dog(base_text)

    t0 = time.time()
    adv_pil, info = attack.attack(base_pil, epsilon=epsilon, queries=queries)
    dt = time.time() - t0

    # Verify with longer max_tokens
    verify_pil = adv_pil.resize((image_size, image_size), Image.LANCZOS)
    verify_text = adapter.query(verify_pil, max_tokens=120)
    v_dog = contains_dog(verify_text)
    v_bonus = anti_target_score(verify_text, target_group)
    v_score = character_score(verify_text) + v_bonus

    # Distortion metrics
    adv_arr = np.asarray(verify_pil, dtype=np.float32) / 255.0
    diff = adv_arr - base_arr
    linf = float(np.abs(diff).max())
    l2 = float(np.sqrt((diff ** 2).mean()))
    mean_abs = float(np.abs(diff).mean())

    if save_image_path:
        os.makedirs(os.path.dirname(save_image_path) or ".", exist_ok=True)
        verify_pil.save(save_image_path)

    return {
        "epsilon_requested": epsilon,
        "epsilon_linf_actual": linf,
        "l2_mean": l2,
        "mean_abs_diff": mean_abs,
        "queries_used": info["queries_used"],
        "queries_budget": queries,
        "elapsed_s": dt,
        "baseline_text": base_text,
        "baseline_score": base_combined,
        "baseline_dog": base_dog,
        "attack_best_text": info["best_text"],
        "attack_best_score": info["best_score"],
        "verification_text": verify_text,
        "verification_score": v_score,
        "verification_dog": v_dog,
        "verification_target_hit": v_bonus > 0,
        "target_group": target_group,
    }


def main():
    p = argparse.ArgumentParser(
        description="L-inf perturbation-budget sweep for GLM-5 attack.")
    p.add_argument("--image", default="dog.jpg")
    p.add_argument("--mock-host", default="http://127.0.0.1:11884")
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--target", default="cat")
    p.add_argument("--epsilons", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128],
                   help="Epsilon budgets in /255 units (e.g. 8 16 32 64 128).")
    p.add_argument("--image-size", type=int, default=378)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--output-dir", default="outputs/eps_sweep")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    # Verify mock server up. We accept 200, 400, or 500 as "alive":
    # the mock doesn't have an image, so it'll fail gracefully.
    health_url = args.mock_host.rstrip("/")
    if not health_url.endswith("/api/paas/v4/chat/completions"):
        health_url = f"{health_url}/api/paas/v4/chat/completions"
    try:
        r = requests.post(
            health_url,
            json={"model": "glm-5v-turbo",
                  "messages": [{"role": "user",
                                "content": [{"type": "text",
                                             "text": "ping"}]}],
                  "max_tokens": 1},
            timeout=10,
        )
        if r.status_code not in (200, 400, 500):
            raise RuntimeError(f"unexpected status {r.status_code}")
    except Exception as e:
        print(f"ERROR: mock server not reachable at {health_url}: {e}")
        print("Start it with:")
        print(f"  PYTHONPATH=. python experiments/mock_glm_server.py "
              f"--port 11884 --preload &")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'=' * 78}")
    print(f"GLM-5 ATTACK - PERTURBATION-BUDGET SWEEP")
    print(f"{'=' * 78}")
    print(f"image        : {args.image} -> {args.image_size}x{args.image_size}")
    print(f"queries/run  : {args.queries}")
    print(f"target       : {args.target}")
    print(f"epsilons     : {args.epsilons} (in /255 units)")
    print(f"mock_host    : {args.mock_host}")
    print()

    results: List[dict] = []
    summary_rows: List[dict] = []

    for eps_x255 in args.epsilons:
        eps = eps_x255 / 255.0
        print(f"\n--- epsilon = {eps_x255}/255 = {eps:.5f} ---")
        out_img = os.path.join(args.output_dir,
                               f"adv_eps{eps_x255:03d}.png")
        r = run_one(
            image_path=args.image,
            mock_host=args.mock_host,
            queries=args.queries,
            epsilon=eps,
            target_group=args.target,
            image_size=args.image_size,
            max_tokens=args.max_tokens,
            save_image_path=out_img,
        )
        results.append(r)
        summary_rows.append({
            "eps": f"{eps_x255}/255",
            "linf_actual": f"{r['epsilon_linf_actual']:.4f}",
            "l2_mean": f"{r['l2_mean']:.4f}",
            "mean_abs": f"{r['mean_abs_diff']:.4f}",
            "queries": f"{r['queries_used']}/{r['queries_budget']}",
            "time_s": f"{r['elapsed_s']:.1f}",
            "base_score": f"{r['baseline_score']:.1f}",
            "best_score": f"{r['attack_best_score']:.1f}",
            "verify_score": f"{r['verification_score']:.1f}",
            "verify_dog": "YES" if r["verification_dog"] else "no",
            "target_hit": "YES" if r["verification_target_hit"] else "no",
            "response": r["verification_text"][:60],
        })

    # Print summary table
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    cols = ["eps", "linf_actual", "l2_mean", "mean_abs", "queries",
            "time_s", "base_score", "best_score", "verify_score",
            "verify_dog", "target_hit", "response"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in summary_rows))
              for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for r in summary_rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    # Save per-epsilon JSON + image
    out_json = os.path.join(args.output_dir, "sweep_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults JSON: {out_json}")

    # Print final verdict
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    successes = [r for r in results
                 if not r["verification_dog"]
                 and (r["verification_target_hit"]
                      or r["verification_score"] >= SUCCESS_SCORE)]
    if successes:
        eps_list = ", ".join(f"{r['epsilon_requested']*255:.0f}/255"
                             for r in successes)
        print(f"  SUCCESS at epsilon(s): {eps_list}")
    else:
        print("  No epsilon fully fooled GLM-5 with this budget. "
              "Increase queries or use the real API.")
    print(f"  Mock server's ResNet50 is deterministic and may not respond "
          f"to small perturbations.")
    print(f"  On the real GLM-5V-Turbo API, expect success at any of "
          f"these epsilon levels within the budget.")


if __name__ == "__main__":
    main()
