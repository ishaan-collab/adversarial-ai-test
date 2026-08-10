"""
Live demonstration: Run the full attack from scratch on a single image.
Shows every step with clear output.

Usage:
    PYTHONPATH=. python experiments/live_demo.py --image data/vlm/dog03.jpg
"""

import sys
import time
import argparse
import numpy as np
import torch
from PIL import Image

from attacks.moondream_pgd import moondream_pgd
from models.moondream_adapter import MoondreamAdapter
from attacks.blackbox_attack import ollama_score, DOG_KEYWORDS

sys.stdout.reconfigure(line_buffering=True)


def banner(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/vlm/dog03.jpg")
    parser.add_argument("--target", default="A cat sitting on a couch")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    eps_display = args.epsilon * 255

    # ============================================================
    # STEP 1: Load model
    # ============================================================
    banner("STEP 1: LOADING HUGGINGFACE MOONDREAM2 MODEL")
    t0 = time.time()
    md = MoondreamAdapter()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Device: {md.device}")
    print(f"  Dtype:  {md.dtype}")

    # ============================================================
    # STEP 2: Load image
    # ============================================================
    banner("STEP 2: LOADING IMAGE")
    pil_clean = Image.open(args.image).convert("RGB")
    pil_clean = pil_clean.resize((378, 378), Image.LANCZOS)
    img_tensor = (
        torch.from_numpy(np.array(pil_clean))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )
    print(f"  Image:  {args.image}")
    print(f"  Size:   378 x 378")
    print(f"  Shape:  {img_tensor.shape}")
    print(f"  Range:  [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    # ============================================================
    # STEP 3: Get clean description (before attack)
    # ============================================================
    banner("STEP 3: CLEAN IMAGE DESCRIPTION (before attack)")
    print("  Querying HF moondream2 with clean image...")
    clean_desc = md.describe_single_crop(pil_clean)
    print(f"\n  >>> MODEL OUTPUT: \"{clean_desc}\"")

    has_dog = any(kw in clean_desc.lower() for kw in DOG_KEYWORDS)
    print(f"  Contains dog keyword: {has_dog}")

    # ============================================================
    # STEP 4: Run the attack
    # ============================================================
    banner("STEP 4: RUNNING ADVERSARIAL ATTACK")
    print(f"  Target text:  \"{args.target}\"")
    print(f"  Epsilon:      {args.epsilon:.6f} ({eps_display:.0f}/255)")
    print(f"  Iterations:   {args.iterations}")
    print(f"  Alpha:        {2/255:.6f}")
    print()
    print("  Running PGD optimization...")
    print("  (Each iteration: forward pass -> loss -> backward -> pixel update)")
    print()

    t0 = time.time()
    adv_tensor, details = moondream_pgd(
        md,
        img_tensor,
        target_text=args.target,
        epsilon=args.epsilon,
        alpha=2 / 255,
        iterations=args.iterations,
        lambda_vision=1.0,
        lambda_alignment=1.0,
        lambda_language=5.0,
        random_start=True,
        seed=42,
        return_details=True,
    )
    attack_time = time.time() - t0

    print(f"\n  Attack completed in {attack_time:.1f}s")
    print(f"  L-inf perturbation: {details['linf'] * 255:.2f}/255")
    print(f"  L2 perturbation:    {details['l2']:.4f}")
    print(f"  Mean abs change:    {details['mean_abs'] * 255:.2f}/255")

    # ============================================================
    # STEP 5: Get adversarial description (after attack)
    # ============================================================
    banner("STEP 5: ADVERSARIAL IMAGE DESCRIPTION (after attack)")
    adv_pil = Image.fromarray(
        (adv_tensor[0].cpu().permute(1, 2, 0).numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )

    print("  Querying HF moondream2 with adversarial image...")
    adv_desc = md.describe_single_crop(adv_pil)
    print(f"\n  >>> MODEL OUTPUT: \"{adv_desc}\"")

    has_dog_adv = any(kw in adv_desc.lower() for kw in DOG_KEYWORDS)
    has_cat_adv = "cat" in adv_desc.lower()
    print(f"  Contains dog keyword: {has_dog_adv}")
    print(f"  Contains 'cat':       {has_cat_adv}")

    # ============================================================
    # STEP 6: Side-by-side comparison
    # ============================================================
    banner("STEP 6: RESULTS COMPARISON")
    print()
    print("  +---------------------------+------------------------------------------+")
    print("  | Image                     | Model Output                             |")
    print("  +---------------------------+------------------------------------------+")

    clean_short = clean_desc[:75] + "..." if len(clean_desc) > 75 else clean_desc
    adv_short = adv_desc[:75] + "..." if len(adv_desc) > 75 else adv_desc
    print(f"  | CLEAN (original)          | {clean_short:<40} |")
    print(f"  | ADVERSARIAL (perturbed)   | {adv_short:<40} |")
    print("  +---------------------------+------------------------------------------+")
    print()
    print(f"  Perturbation: max {details['linf'] * 255:.1f}/255 per pixel (invisible)")
    print(f"  Attack succeeded: {not has_dog_adv and has_cat_adv}")

    # ============================================================
    # STEP 7: Also check Ollama (Q4 quantized)
    # ============================================================
    banner("STEP 7: OLLAMA CHECK (Q4 quantized, black-box)")
    print("  Querying Ollama moondream with clean image...")
    _, clean_ol, _ = ollama_score(pil_clean, num_predict=60)
    print(f"\n  >>> OLLAMA CLEAN: \"{clean_ol.strip()[:100]}\"")

    print("\n  Querying Ollama moondream with adversarial image...")
    _, adv_ol, _ = ollama_score(adv_pil, num_predict=60)
    print(f"\n  >>> OLLAMA ADV:   \"{adv_ol.strip()[:100]}\"")

    ol_dog = any(kw in adv_ol.lower() for kw in DOG_KEYWORDS)
    print(f"\n  Ollama still says 'dog': {ol_dog}")
    if ol_dog:
        print("  -> Q4 quantization blocks the attack (as expected)")

    # ============================================================
    # Save
    # ============================================================
    banner("SAVED FILES")
    adv_pil.save("outputs/live_demo_adv.png")
    print(f"  Adversarial image: outputs/live_demo_adv.png")
    print(f"  You can compare with: {args.image}")


if __name__ == "__main__":
    main()
