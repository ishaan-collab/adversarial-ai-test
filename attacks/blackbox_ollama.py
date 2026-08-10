"""
Black-box adversarial attack on Ollama moondream using random search.

Uses HF white-box attack as initialization, then refines with
Ollama API feedback (no gradients needed).

Strategy:
  1. Start from HF-optimized adversarial image
  2. Randomly perturb a subset of pixels within epsilon ball
  3. Query Ollama
  4. If "dog" disappears or target keyword appears, keep change
  5. Repeat for N iterations
"""

import io
import base64
import random
import time
import requests
import numpy as np
from PIL import Image
import torch


def ollama_query(pil_img, host="http://127.0.0.1:11435",
                 model="moondream", prompt="What do you see in this image?",
                 temperature=0.5, num_predict=200):
    """Query Ollama and return the text response."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
    return resp.json()["message"]["content"]


def score_response(text, source_keyword="dog", target_keyword="cat"):
    """
    Score the response. Lower = better for attacker.
    -2: target keyword present, source absent (perfect)
    -1: source absent but target absent (partial success)
     0: both present
    +1: source present, target absent (failure)
    """
    text_lower = text.lower()
    has_source = source_keyword in text_lower
    has_target = target_keyword in text_lower

    if has_target and not has_source:
        return -2
    elif not has_source and not has_target:
        return -1
    elif has_source and has_target:
        return 0
    else:
        return 1


def blackbox_refine(
    clean_pil,
    adv_pil,
    epsilon=8 / 255,
    iterations=100,
    block_size=10,
    source_keyword="dog",
    target_keyword="cat",
    host="http://127.0.0.1:11435",
    model="moondream",
    seed=42,
):
    """
    Refine adversarial image using black-box Ollama feedback.

    Args:
        clean_pil:     Clean PIL image
        adv_pil:       Initial adversarial PIL image (from HF attack)
        epsilon:       L-inf budget in [0, 1]
        iterations:    Number of Ollama queries
        block_size:    Size of random perturbation blocks (pixels)
        source_keyword: Keyword to remove (e.g. "dog")
        target_keyword: Keyword to inject (e.g. "cat")
        host:          Ollama API host
        model:         Ollama model name
        seed:          Random seed

    Returns:
        best_adv_pil:  Best adversarial image found
        info:          Dict with search history
    """
    random.seed(seed)
    np.random.seed(seed)

    clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
    adv_arr = np.array(adv_pil, dtype=np.float32) / 255.0
    H, W, C = clean_arr.shape

    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)

    # Score the initial adversarial image
    print("Querying Ollama for initial adv image...")
    best_text = ollama_query(adv_pil, host, model)
    best_score = score_response(best_text, source_keyword, target_keyword)
    print(f"  Initial: score={best_score} | {best_text[:100]}")

    history = [{"iter": 0, "score": best_score, "text": best_text[:200]}]

    if best_score <= -1:
        print("  Already good enough!")
        return adv_pil, {"history": history, "best_score": best_score}

    for i in range(iterations):
        # Generate random block perturbation
        candidate = adv_arr.copy()

        # Perturb a random rectangular region
        bh = random.randint(block_size // 2, block_size * 2)
        bw = random.randint(block_size // 2, block_size * 2)
        y0 = random.randint(0, H - bh)
        x0 = random.randint(0, W - bw)

        # Random noise in [-epsilon, epsilon]
        noise = np.random.uniform(
            -epsilon, epsilon, size=(bh, bw, C)
        ).astype(np.float32)

        candidate[y0:y0 + bh, x0:x0 + bw] += noise

        # Clip to epsilon ball around clean image
        candidate = np.clip(candidate, lower, upper)
        candidate = np.clip(candidate, 0, 1)

        # Convert to PIL and query Ollama
        candidate_pil = Image.fromarray(
            (candidate * 255).astype(np.uint8)
        )

        text = ollama_query(candidate_pil, host, model)
        score = score_response(text, source_keyword, target_keyword)

        improved = score < best_score

        if improved:
            adv_arr = candidate
            adv_pil = candidate_pil
            best_score = score
            best_text = text
            marker = "*** IMPROVED ***"
        else:
            marker = ""

        history.append({
            "iter": i + 1,
            "score": score,
            "text": text[:200],
            "improved": improved,
        })

        elapsed = time.time()
        print(
            f"  [{i+1:3d}/{iterations}] score={score} best={best_score} "
            f"| {text[:80]} {marker}"
        )

        if best_score <= -2:
            print("  Target achieved! Stopping early.")
            break

    print(f"\nFinal: score={best_score}")
    print(f"  Text: {best_text[:200]}")

    return adv_pil, {"history": history, "best_score": best_score}


if __name__ == "__main__":
    import argparse
    from models.moondream_adapter import MoondreamAdapter
    from attacks.moondream_pgd import moondream_pgd

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--target", default="A cat sitting on a couch")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--hf-iterations", type=int, default=200)
    parser.add_argument("--bb-iterations", type=int, default=100)
    parser.add_argument("--output", default="outputs/adv_blackbox.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: HF white-box attack as initialization
    print("=" * 60)
    print("Step 1: HF white-box attack (initialization)")
    print("=" * 60)

    model = MoondreamAdapter()
    pil_img = Image.open(args.image).convert("RGB")
    pil_378 = pil_img.resize((378, 378), Image.LANCZOS)
    tensor = (
        torch.from_numpy(np.array(pil_378))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(model.device)
    )

    clean_text = ollama_query(pil_378)
    print(f"Clean Ollama: {clean_text[:150]}")

    adv_tensor, _ = moondream_pgd(
        model=model,
        image=tensor,
        target_text=args.target,
        epsilon=args.epsilon,
        alpha=args.epsilon / 4,
        iterations=args.hf_iterations,
        lambda_vision=1.0,
        lambda_alignment=1.0,
        lambda_language=5.0,
        random_start=True,
        seed=args.seed,
        return_details=True,
    )

    adv_pil = model.tensor_to_pil(adv_tensor)
    hf_text = model.describe(adv_pil)
    ol_text = ollama_query(adv_pil)
    print(f"HF adv:      {hf_text[:150]}")
    print(f"Ollama adv:  {ol_text[:150]}")

    # Step 2: Black-box refinement with Ollama feedback
    print()
    print("=" * 60)
    print("Step 2: Black-box refinement (Ollama feedback)")
    print("=" * 60)

    best_adv, info = blackbox_refine(
        clean_pil=pil_378,
        adv_pil=adv_pil,
        epsilon=args.epsilon,
        iterations=args.bb_iterations,
        source_keyword="dog",
        target_keyword="cat",
        seed=args.seed,
    )

    best_adv.save(args.output)
    print(f"\nSaved to {args.output}")

    # Final comparison
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Clean:    {clean_text[:150]}")
    print(f"HF adv:   {hf_text[:150]}")
    final_ol = ollama_query(best_adv)
    print(f"BB final: {final_ol[:150]}")
