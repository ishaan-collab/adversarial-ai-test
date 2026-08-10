"""
HF Gradient-Guided Black-Box Attack on Ollama moondream.

Uses HF white-box gradient to identify the most important pixels,
then does coordinate descent on those pixels using Ollama feedback.

Strategy:
  1. Run HF attack, get the gradient
  2. Select top-K pixels by gradient magnitude
  3. For each pixel, try +epsilon and -epsilon
  4. Keep whichever improves Ollama score
  5. Multiple passes over all selected pixels

This reduces the search space from 428K continuous variables to
K binary variables, making black-box optimization tractable.
"""

import io
import json
import time
import base64
import requests
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.moondream_adapter import MoondreamAdapter
from attacks.moondream_pgd import moondream_pgd


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
                recent = "".join(
                    tokens[max(0, i - 3):i + 1]
                ).lower()
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


def pil_from_array(arr):
    return Image.fromarray(
        (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    )


def gradient_guided_attack(
    clean_pil, epsilon=8/255, max_queries=5000,
    top_k=500, num_passes=3, hf_model=None,
    target_text="A cat sitting on a couch",
    host="http://127.0.0.1:11435",
    num_predict=20, seed=42, verbose=True,
):
    """
    HF gradient-guided black-box attack.

    1. Run HF white-box attack to get gradient
    2. Select top-K pixels by gradient magnitude
    3. Coordinate descent: for each pixel, try +/- epsilon
    4. Multiple passes
    """
    rng = np.random.RandomState(seed)
    query_count = 0

    clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
    H, W, C = clean_arr.shape

    device = hf_model.device if hf_model else "cuda"
    tensor = torch.from_numpy(clean_arr).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(device)

    # ============================================
    # Step 1: Get HF gradient
    # ============================================
    if verbose:
        print("Step 1: Computing HF gradient...")

    tensor.requires_grad_(True)

    # Get features for loss computation
    features = hf_model.get_all_features(tensor)

    # Tokenize target
    target_ids = hf_model._tokenize(target_text)
    target_ids = target_ids.to(device)

    # Multi-token CE loss
    l_lang, _ = hf_model.get_multi_token_loss(tensor, target_ids)
    l_vision = F.mse_loss(
        features["vision_features"].float(),
        features["vision_features"].float().detach(),
    )
    l_alignment = F.mse_loss(
        features["connector_tokens"].float(),
        features["connector_tokens"].float().detach(),
    )

    loss = l_lang - 0.1 * l_vision - 0.1 * l_alignment

    grad = torch.autograd.grad(
        loss, tensor, retain_graph=False, create_graph=False
    )[0]

    grad_np = grad[0].permute(1, 2, 0).detach().cpu().numpy()
    grad_magnitude = np.abs(grad_np).mean(axis=2)

    # ============================================
    # Step 2: Select top-K pixels
    # ============================================
    flat_idx = np.argsort(grad_magnitude.ravel())[::-1][:top_k]
    pixel_coords = [(idx // W, idx % W) for idx in flat_idx]

    if verbose:
        print(f"  Gradient shape: {grad_np.shape}")
        print(f"  Top-{top_k} pixels selected")
        print(f"  Gradient magnitude range: "
              f"{grad_magnitude.min():.6f} - "
              f"{grad_magnitude.max():.6f}")

    # ============================================
    # Step 3: Initialize perturbation
    # ============================================
    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)

    # Start from clean
    adv_arr = clean_arr.copy()

    # Also try: start from HF adversarial direction
    # Use the gradient sign as initial perturbation
    grad_sign = np.sign(grad_np)
    adv_arr_hf = np.clip(clean_arr + epsilon * grad_sign, lower, upper)

    # Score both
    score_clean, text_clean, _ = ollama_score(
        pil_from_array(adv_arr), host=host, num_predict=num_predict
    )
    query_count += 1
    score_hf, text_hf, _ = ollama_score(
        pil_from_array(adv_arr_hf), host=host, num_predict=num_predict
    )
    query_count += 1

    if verbose:
        print(f"\n  Clean score: {score_clean:.1f} | {text_clean[:80]}")
        print(f"  HF+grad score: {score_hf:.1f} | {text_hf[:80]}")

    # Use better starting point
    if score_hf > score_clean:
        adv_arr = adv_arr_hf.copy()
        best_score = score_hf
        best_text = text_hf
        if verbose:
            print("  Starting from HF gradient direction")
    else:
        # Try negated gradient
        adv_arr_neg = np.clip(
            clean_arr - epsilon * grad_sign, lower, upper
        )
        score_neg, text_neg, _ = ollama_score(
            pil_from_array(adv_arr_neg), host=host,
            num_predict=num_predict
        )
        query_count += 1

        if score_neg > score_clean:
            adv_arr = adv_arr_neg.copy()
            best_score = score_neg
            best_text = text_neg
            if verbose:
                print(f"  Neg gradient score: {score_neg:.1f} | "
                      f"{text_neg[:80]}")
                print("  Starting from NEGATED HF gradient")
        else:
            best_score = score_clean
            best_text = text_clean
            if verbose:
                print("  Starting from clean")

    success_threshold = num_predict * 10.0 + 100.0

    # ============================================
    # Step 4: Coordinate descent
    # ============================================
    if verbose:
        print(f"\nStep 2: Coordinate descent "
              f"({top_k} pixels, {num_passes} passes)")

    # Shuffle pixel order for each pass
    for pass_num in range(num_passes):
        if query_count >= max_queries:
            break

        order = list(range(len(pixel_coords)))
        rng.shuffle(order)

        improvements = 0

        for idx_pos in order:
            if query_count >= max_queries:
                break

            y, x = pixel_coords[idx_pos]

            # Current values
            current = adv_arr[y, x].copy()

            # Try +epsilon for all channels
            candidate = adv_arr.copy()
            candidate[y, x] = upper[y, x]
            pil = pil_from_array(candidate)
            score_plus, text_plus, _ = ollama_score(
                pil, host=host, num_predict=num_predict
            )
            query_count += 1

            improved = False

            if score_plus > best_score:
                adv_arr = candidate
                best_score = score_plus
                best_text = text_plus
                improvements += 1
                improved = True

                if verbose:
                    print(f"  [q={query_count:5d} p={pass_num+1} "
                          f"pix=({y},{x})] score={best_score:.1f} "
                          f"| {best_text[:60]} *** +eps ***")

                if best_score >= success_threshold:
                    if verbose:
                        print(f"  SUCCESS!")
                        print(f"  Text: {best_text[:200]}")
                    return pil_from_array(adv_arr), {
                        "best_score": best_score,
                        "queries": query_count,
                        "epsilon": epsilon,
                        "best_text": best_text,
                    }
            else:
                # Try -epsilon
                candidate = adv_arr.copy()
                candidate[y, x] = lower[y, x]
                pil = pil_from_array(candidate)
                score_minus, text_minus, _ = ollama_score(
                    pil, host=host, num_predict=num_predict
                )
                query_count += 1

                if score_minus > best_score:
                    adv_arr = candidate
                    best_score = score_minus
                    best_text = text_minus
                    improvements += 1
                    improved = True

                    if verbose:
                        print(f"  [q={query_count:5d} p={pass_num+1} "
                              f"pix=({y},{x})] score={best_score:.1f} "
                              f"| {best_text[:60]} *** -eps ***")

                    if best_score >= success_threshold:
                        if verbose:
                            print(f"  SUCCESS!")
                            print(f"  Text: {best_text[:200]}")
                        return pil_from_array(adv_arr), {
                            "best_score": best_score,
                            "queries": query_count,
                            "epsilon": epsilon,
                            "best_text": best_text,
                        }

        if verbose:
            print(f"  Pass {pass_num + 1}: {improvements} improvements, "
                  f"best={best_score:.1f}, queries={query_count}")

        if improvements == 0:
            if verbose:
                print("  No improvements in this pass, stopping.")
            break

    linf = np.abs(adv_arr - clean_arr).max()
    if verbose:
        print(f"\n  Final: score={best_score:.1f} "
              f"queries={query_count}")
        print(f"  L-inf: {linf:.8f} (budget: {epsilon:.8f})")
        print(f"  Text: {best_text[:200]}")

    return pil_from_array(adv_arr), {
        "best_score": best_score,
        "queries": query_count,
        "epsilon": epsilon,
        "best_text": best_text,
        "linf": float(linf),
    }


def block_coordinate_descent(
    clean_pil, epsilon=8/255, max_queries=5000,
    block_size=20, hf_model=None,
    target_text="A cat sitting on a couch",
    host="http://127.0.0.1:11435",
    num_predict=20, seed=42, verbose=True,
):
    """
    Block coordinate descent: optimize blocks of pixels instead
    of individual pixels. More likely to cause measurable change.

    For each block:
      1. Try pushing all pixels to +epsilon
      2. Try pushing all pixels to -epsilon
      3. Try random pattern within block
      4. Keep best
    """
    rng = np.random.RandomState(seed)
    query_count = 0

    clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
    H, W, C = clean_arr.shape

    lower = np.clip(clean_arr - epsilon, 0, 1)
    upper = np.clip(clean_arr + epsilon, 0, 1)

    adv_arr = clean_arr.copy()

    # Initial score
    pil = pil_from_array(adv_arr)
    score, text, _ = ollama_score(
        pil, host=host, num_predict=num_predict
    )
    query_count += 1
    best_score = score
    best_text = text

    if verbose:
        print(f"  [init] score={best_score:.1f} | {best_text[:80]}")

    success_threshold = num_predict * 10.0 + 100.0

    # Generate blocks covering the image
    blocks = []
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            y_end = min(y + block_size, H)
            x_end = min(x + block_size, W)
            blocks.append((y, y_end, x, x_end))

    # Prioritize blocks by HF gradient magnitude
    if hf_model:
        device = hf_model.device
        tensor = torch.from_numpy(clean_arr).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        features = hf_model.get_all_features(tensor)
        target_ids = hf_model._tokenize(target_text).to(device)
        l_lang, _ = hf_model.get_multi_token_loss(tensor, target_ids)
        grad = torch.autograd.grad(l_lang, tensor)[0]
        grad_np = grad[0].permute(1, 2, 0).detach().cpu().numpy()
        grad_mag = np.abs(grad_np).mean(axis=2)

        block_scores = []
        for (y0, y1, x0, x1) in blocks:
            block_scores.append(grad_mag[y0:y1, x0:x1].mean())
        block_order = np.argsort(block_scores)[::-1]
        blocks = [blocks[i] for i in block_order]

        if verbose:
            print(f"  Blocks prioritized by HF gradient "
                  f"({len(blocks)} blocks)")

    # Multiple passes
    for pass_num in range(5):
        if query_count >= max_queries:
            break

        improvements = 0
        rng.shuffle(blocks)

        for (y0, y1, x0, x1) in blocks:
            if query_count >= max_queries:
                break

            bh = y1 - y0
            bw = x1 - x0

            # Try 3 variants per block
            candidates = [
                ("+eps", np.clip(adv_arr.copy().__setitem__(
                    slice(y0, y1), slice(x0, x1), 0
                ) or adv_arr.copy(), lower, upper) if False else None),
            ]

            # Variant 1: all +epsilon
            v1 = adv_arr.copy()
            v1[y0:y1, x0:x1] = upper[y0:y1, x0:x1]
            pil = pil_from_array(v1)
            s1, t1, _ = ollama_score(
                pil, host=host, num_predict=num_predict
            )
            query_count += 1

            best_variant = None
            best_v_score = best_score

            if s1 > best_score:
                best_variant = v1
                best_v_score = s1
                best_v_text = t1

            if query_count >= max_queries:
                break

            # Variant 2: all -epsilon
            v2 = adv_arr.copy()
            v2[y0:y1, x0:x1] = lower[y0:y1, x0:x1]
            pil = pil_from_array(v2)
            s2, t2, _ = ollama_score(
                pil, host=host, num_predict=num_predict
            )
            query_count += 1

            if s2 > best_v_score:
                best_variant = v2
                best_v_score = s2
                best_v_text = t2

            if query_count >= max_queries:
                break

            # Variant 3: random pattern
            v3 = adv_arr.copy()
            noise = rng.choice(
                [-1, 1], size=(bh, bw, C)
            ).astype(np.float32) * epsilon
            v3[y0:y1, x0:x1] = np.clip(
                clean_arr[y0:y1, x0:x1] + noise, 0, 1
            )
            v3 = np.clip(v3, lower, upper)
            pil = pil_from_array(v3)
            s3, t3, _ = ollama_score(
                pil, host=host, num_predict=num_predict
            )
            query_count += 1

            if s3 > best_v_score:
                best_variant = v3
                best_v_score = s3
                best_v_text = t3

            if best_variant is not None:
                adv_arr = best_variant
                best_score = best_v_score
                best_text = best_v_text
                improvements += 1

                if verbose:
                    print(f"  [q={query_count:5d} p={pass_num+1} "
                          f"block=({y0},{x0},{bw}x{bh})] "
                          f"score={best_score:.1f} "
                          f"| {best_text[:60]} ***")

                if best_score >= success_threshold:
                    if verbose:
                        print(f"  SUCCESS!")
                        print(f"  Text: {best_text[:200]}")
                    return pil_from_array(adv_arr), {
                        "best_score": best_score,
                        "queries": query_count,
                        "epsilon": epsilon,
                        "best_text": best_text,
                    }

        if verbose:
            print(f"  Pass {pass_num + 1}: {improvements} improvements, "
                  f"best={best_score:.1f}, queries={query_count}")

        if improvements == 0 and pass_num > 0:
            if verbose:
                print("  No improvements, stopping.")
            break

    linf = np.abs(adv_arr - clean_arr).max()
    if verbose:
        print(f"\n  Final: score={best_score:.1f} "
              f"queries={query_count}")
        print(f"  L-inf: {linf:.8f} (budget: {epsilon:.8f})")
        print(f"  Text: {best_text[:200]}")

    return pil_from_array(adv_arr), {
        "best_score": best_score,
        "queries": query_count,
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
        description="HF gradient-guided black-box attack on Ollama"
    )
    parser.add_argument("--image", default="data/vlm/dog07.jpg")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--output", default="outputs/adv_grad_guided.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=20)
    parser.add_argument("--attack-size", type=int, default=378)
    parser.add_argument("--method", choices=["pixel", "block"],
                        default="block")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--target", default="A cat sitting on a couch")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    pil = Image.open(args.image).convert("RGB")
    pil = pil.resize((args.attack_size, args.attack_size), Image.LANCZOS)

    print(f"Image: {args.image} ({args.attack_size}x{args.attack_size})")
    print(f"Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"Max queries: {args.queries}")
    print(f"Method: {args.method}")
    print()

    print("Loading HF moondream2 for gradient...")
    hf_model = MoondreamAdapter()

    if args.method == "pixel":
        adv_pil, info = gradient_guided_attack(
            pil, epsilon=args.epsilon, max_queries=args.queries,
            top_k=args.top_k, hf_model=hf_model,
            target_text=args.target, seed=args.seed,
        )
    else:
        adv_pil, info = block_coordinate_descent(
            pil, epsilon=args.epsilon, max_queries=args.queries,
            block_size=args.block_size, hf_model=hf_model,
            target_text=args.target, seed=args.seed,
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
