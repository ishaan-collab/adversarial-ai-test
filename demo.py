#!/usr/bin/env python3
"""
========================================================================
  DRISHTI - Adversarial Robustness Demonstration
========================================================================

  A single entry point that demonstrates the full adversarial attack
  pipeline. Run this when someone asks "show me what this project does."

  THREE PARTS, one command:

    Part 1  Classical Classifier (ResNet-50)
            Clean baseline -> FGSM / BIM / PGD -> prediction changes

    Part 2  Vision-Language Model (Moondream2, white-box)
            "a golden retriever..." -> PGD -> "A cat sitting on a couch"

    Part 3  Cross-Model Transfer (Ollama, Q4 quantized, black-box)
            Same adversarial image -> different deployment -> does it
            still fool the model?

  USAGE:

    python demo.py                        # Full demo (all 3 parts)
    python demo.py --quick                # Fast demo (~2 min, 50 iters)
    python demo.py --classical-only       # Part 1 only (~1 second)
    python demo.py --skip-classical       # Parts 2 + 3 only
    python demo.py --skip-transfer        # Parts 1 + 2 only
    python demo.py --image data/vlm/dog03.jpg
    python demo.py --target "A photo of a cat"
    python demo.py --iterations 500       # More PGD iterations

  REQUIRES:
    - PyTorch with CUDA (for Part 2)
    - Ollama running on localhost:11435 with moondream model (Part 3)

  OUTPUTS:
    All images and JSON results saved to outputs/demo/
========================================================================
"""

import sys
import os
import time
import json
import argparse
import traceback

# Ensure project root is on sys.path so imports work from anywhere
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.chdir(_ROOT)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ============================================================
# CONSTANTS
# ============================================================

W = 72  # banner width

DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]


# ============================================================
# PRINTING UTILITIES
# ============================================================

def banner(text, char="="):
    """Print a centered banner."""
    print()
    print(char * W)
    pad = (W - len(text) - 2) // 2
    line = char * pad + f" {text} " + char * pad
    if len(line) < W:
        line += char
    print(line)
    print(char * W)


def step(num, total, text):
    """Print a numbered step."""
    print(f"\n  [{num}/{total}] {text}")


def trunc(text, length=80):
    """Truncate text to length with ellipsis."""
    text = str(text).strip().replace("\n", " ")
    return text[:length] + "..." if len(text) > length else text


def box(left, right, width=40):
    """Print a table row: | left | right |"""
    print(f"  | {trunc(left, 26):<26} | {trunc(right, width):<{width}} |")


def table_header(left_label, right_label, width=40):
    """Print a table header."""
    print(f"  +{'-'*28}+{'-'*(width+2)}+")
    print(f"  | {left_label:<26} | {right_label:<{width}} |")
    print(f"  +{'-'*28}+{'-'*(width+2)}+")


def table_footer(width=40):
    print(f"  +{'-'*28}+{'-'*(width+2)}+")


def save_pil(pil, path):
    """Save a PIL image, creating directories as needed."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    pil.save(path)
    print(f"  Saved: {path}")


def tensor_to_pil(tensor):
    """Convert [1,3,H,W] tensor in [0,1] to PIL Image."""
    import numpy as np
    t = tensor[0].detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_env():
    """Check and report the runtime environment."""
    banner("ENVIRONMENT CHECK")

    checks = {}

    # Python
    step(1, 5, "Python")
    print(f"        Version: {sys.version.split()[0]}")
    checks["python"] = True

    # PyTorch + CUDA
    step(2, 5, "PyTorch / CUDA")
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU only"
        print(f"        PyTorch:  {torch.__version__}")
        print(f"        CUDA:     {'YES - ' + device if cuda else 'NO'}")
        checks["torch"] = True
        checks["cuda"] = cuda
    except ImportError:
        print("        [FAIL] PyTorch not installed")
        checks["torch"] = False
        checks["cuda"] = False

    # Transformers
    step(3, 5, "Transformers (for VLM)")
    try:
        import transformers
        print(f"        Version:  {transformers.__version__}")
        checks["transformers"] = True
    except ImportError:
        print("        [WARN] transformers not installed (Part 2 will be skipped)")
        checks["transformers"] = False

    # Ollama
    step(4, 5, "Ollama (for transfer test)")
    try:
        import requests
        resp = requests.get("http://127.0.0.1:11435/api/tags", timeout=5)
        models = resp.json().get("models", [])
        names = [m["name"] for m in models]
        has_moondream = any("moondream" in n for n in names)
        print(f"        Host:     http://127.0.0.1:11435")
        print(f"        Models:   {names if names else 'none'}")
        print(f"        Moondream: {'YES' if has_moondream else 'NO'}")
        checks["ollama"] = has_moondream
    except Exception:
        print("        [WARN] Ollama not reachable (Part 3 will be skipped)")
        checks["ollama"] = False

    # Image
    step(5, 5, "Image files")
    import os as _os
    images = []
    for p in ["dog.jpg", "data/vlm/dog03.jpg", "data/vlm/dog01.jpg"]:
        if _os.path.exists(p):
            images.append(p)
    print(f"        Found:    {images}")
    checks["images"] = images

    return checks


# ============================================================
# PART 1: CLASSICAL CLASSIFIER (ResNet-50)
# ============================================================

def part1_classical(image_path, output_dir, epsilon=8 / 255):
    """
    Part 1: FGSM, BIM, PGD attacks against ResNet-50.

    Shows that imperceptible perturbations (8/255 per pixel) can
    completely change an ImageNet classifier's prediction.
    """
    banner("PART 1: CLASSICAL CLASSIFIER ATTACKS (ResNet-50)")

    import torch
    from models.registry import get_model
    from utils.image import load_image, save_tensor_as_image
    from attacks.fgsm import fgsm_attack
    from attacks.bim import bim_attack
    from attacks.pgd import pgd_attack
    from evaluation.predict import predict
    from evaluation.evaluator import evaluate_attack

    # --- Load model ---
    step(1, 5, "Loading ResNet-50 (ImageNet-pretrained)...")
    t0 = time.time()
    model_info = get_model("resnet50")
    print(f"        Model:   {model_info.name}")
    print(f"        Device:  {model_info.device}")
    print(f"        Loaded:  {time.time() - t0:.1f}s")

    # --- Load image ---
    step(2, 5, f"Loading image: {image_path}")
    _, image = load_image(image_path)
    image = image.to(model_info.device)
    print(f"        Shape:   {tuple(image.shape)}")
    print(f"        Range:   [{image.min():.4f}, {image.max():.4f}]")

    # --- Clean baseline ---
    step(3, 5, "Clean prediction (no perturbation)...")
    clean = predict(
        model=model_info.model,
        image=image,
        weights=model_info.weights,
        preprocess=model_info.preprocess,
    )
    print(f"        Prediction:  {clean['category']}")
    print(f"        Confidence:  {clean['confidence'] * 100:.2f}%")

    label = torch.tensor([clean["class_id"]], device=model_info.device)

    # --- Run attacks ---
    attacks = [
        ("fgsm", fgsm_attack, {"epsilon": epsilon}),
        ("bim",  bim_attack,  {"epsilon": epsilon, "alpha": 2/255, "iterations": 10}),
        ("pgd",  pgd_attack,  {"epsilon": epsilon, "alpha": 2/255, "iterations": 10,
                                "random_start": True}),
    ]

    results = []

    for name, fn, cfg in attacks:
        step(4, 5, f"Running {name.upper()} attack...")
        print(f"        Config: {cfg}")

        t0 = time.time()
        adv = fn(
            model=model_info.model, image=image, label=label,
            preprocess=model_info.preprocess, **cfg,
        )
        elapsed = time.time() - t0

        out = os.path.join(output_dir, f"classical_{name}.png")
        save_tensor_as_image(adv, out)

        ev = evaluate_attack(
            model=model_info.model, weights=model_info.weights,
            original_image=image, adversarial_image=adv,
            preprocess=model_info.preprocess,
        )
        ev["time"] = elapsed
        ev["name"] = name
        results.append((name, ev))

        print(f"        Time:      {elapsed:.2f}s")
        print(f"        Adv pred:  {ev['adversarial']['category']}")
        print(f"        Adv conf:  {ev['adversarial']['confidence'] * 100:.2f}%")
        print(f"        Changed:   {'YES' if ev['prediction_changed'] else 'NO'}")
        print(f"        L-inf:     {ev['linf']:.8f}")

    # --- Summary ---
    step(5, 5, "Summary")
    print()
    print(f"  {'Attack':<8} {'Changed':<9} {'Prediction':<28} "
          f"{'Conf':>8}  {'L-inf':>10}  {'Time':>6}")
    print(f"  {'-'*8} {'-'*9} {'-'*28} {'-'*8}  {'-'*10}  {'-'*6}")
    for name, r in results:
        print(f"  {name.upper():<8} "
              f"{'YES' if r['prediction_changed'] else 'NO':<9} "
              f"{trunc(r['adversarial']['category'], 28):<28} "
              f"{r['adversarial']['confidence'] * 100:>6.2f}%  "
              f"{r['linf']:>10.6f}  "
              f"{r['time']:>5.2f}s")

    ok = sum(1 for _, r in results if r["prediction_changed"])
    print(f"\n  Success: {ok}/{len(results)} ({ok/len(results)*100:.0f}%)")

    save_tensor_as_image(image, os.path.join(output_dir, "classical_clean.png"))
    return results


# ============================================================
# PART 2: VLM WHITE-BOX ATTACK (Moondream2)
# ============================================================

def part2_vlm(image_path, target_text, output_dir,
              epsilon=8 / 255, iterations=300):
    """
    Part 2: Multi-level PGD attack on Moondream2.

    Uses three loss surfaces (vision encoder, connector, language logits)
    to craft a perturbation that changes the model's description from
    "a golden retriever..." to the target text.
    """
    banner("PART 2: VLM WHITE-BOX ATTACK (Moondream2)")

    import numpy as np
    import torch
    from PIL import Image

    from models.moondream_adapter import MoondreamAdapter
    from attacks.moondream_pgd import moondream_pgd

    # --- Load model ---
    step(1, 6, "Loading HuggingFace Moondream2 model...")
    t0 = time.time()
    md = MoondreamAdapter()
    print(f"        Model:   {md.name}")
    print(f"        Device:  {md.device}")
    print(f"        Dtype:   {md.dtype}")
    print(f"        Loaded:  {time.time() - t0:.1f}s")

    # --- Load image ---
    step(2, 6, f"Loading image: {image_path}")
    pil_clean = Image.open(image_path).convert("RGB")
    pil_clean = pil_clean.resize((378, 378), Image.LANCZOS)
    img_tensor = (
        torch.from_numpy(np.array(pil_clean))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    )
    print(f"        Size:    378 x 378")
    print(f"        Shape:   {tuple(img_tensor.shape)}")

    # --- Clean baseline ---
    step(3, 6, 'Querying model with CLEAN image...')
    clean_desc = md.describe_single_crop(pil_clean)
    print(f'\n        >>> CLEAN OUTPUT: "{clean_desc}"')
    has_dog = any(kw in clean_desc.lower() for kw in DOG_KEYWORDS)
    print(f"        Dog keyword: {has_dog}")

    # --- Attack ---
    step(4, 6, "Running multi-level PGD attack...")
    print(f"        Target:    \"{target_text}\"")
    print(f"        Epsilon:   {epsilon:.6f} ({epsilon*255:.1f}/255)")
    print(f"        Itrs:      {iterations}")
    print(f"        Alpha:     {2/255:.6f}")
    print(f"        Loss:      vision + alignment + language")
    print()
    print("        [iter] forward -> loss -> backward -> pixel update")
    print()

    t0 = time.time()
    adv_tensor, details = moondream_pgd(
        md, img_tensor,
        target_text=target_text,
        epsilon=epsilon,
        alpha=2 / 255,
        iterations=iterations,
        lambda_vision=1.0,
        lambda_alignment=1.0,
        lambda_language=5.0,
        random_start=True,
        seed=42,
        return_details=True,
    )
    elapsed = time.time() - t0

    print(f"\n        Completed:  {elapsed:.1f}s")
    print(f"        L-inf:      {details['linf']*255:.2f}/255")
    print(f"        L2:         {details['l2']:.4f}")
    print(f"        Mean |dx|:  {details['mean_abs']*255:.2f}/255")

    # --- Adversarial result ---
    step(5, 6, 'Querying model with ADVERSARIAL image...')
    adv_pil = tensor_to_pil(adv_tensor)
    adv_desc = md.describe_single_crop(adv_pil)
    print(f'\n        >>> ADV OUTPUT: "{adv_desc}"')
    has_dog_adv = any(kw in adv_desc.lower() for kw in DOG_KEYWORDS)
    has_cat_adv = "cat" in adv_desc.lower()
    succeeded = not has_dog_adv and has_cat_adv
    print(f"        Dog keyword: {has_dog_adv}")
    print(f"        Cat keyword: {has_cat_adv}")
    print(f"        Succeeded:   {succeeded}")

    # --- Comparison ---
    step(6, 6, "Results")
    print()
    table_header("Image", "Moondream2 Output")
    box("CLEAN (original)", clean_desc)
    box("ADVERSARIAL", adv_desc)
    table_footer()
    print()
    print(f"  Perturbation:  L-inf = {details['linf']*255:.2f}/255 "
          f"(budget: {epsilon*255:.1f}/255)")
    print(f"  To human eye:  IDENTICAL (changes are < 3% per pixel)")
    print(f"  Attack time:   {elapsed:.1f}s")

    # --- Save ---
    save_pil(pil_clean, os.path.join(output_dir, "vlm_clean.png"))
    save_pil(adv_pil, os.path.join(output_dir, "vlm_adversarial.png"))

    # Perturbation amplified 10x
    pert = (adv_tensor - img_tensor.to(adv_tensor.device)).abs()
    pert_amp = (pert * 10).clamp(0, 1)
    save_pil(tensor_to_pil(pert_amp),
             os.path.join(output_dir, "vlm_perturbation_10x.png"))

    # JSON
    data = {
        "clean_description": clean_desc,
        "adversarial_description": adv_desc,
        "target_text": target_text,
        "epsilon": epsilon,
        "iterations": iterations,
        "attack_time_s": elapsed,
        "linf": details["linf"],
        "l2": details["l2"],
        "mean_abs": details["mean_abs"],
        "attack_succeeded": succeeded,
    }
    jpath = os.path.join(output_dir, "vlm_results.json")
    with open(jpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results: {jpath}")

    return {
        "clean_pil": pil_clean,
        "adv_pil": adv_pil,
        "clean_desc": clean_desc,
        "adv_desc": adv_desc,
        "details": details,
        "time": elapsed,
        "succeeded": succeeded,
    }


# ============================================================
# PART 3: CROSS-MODEL TRANSFER (Ollama Q4)
# ============================================================

def part3_transfer(clean_pil, adv_pil, output_dir,
                   host="http://127.0.0.1:11435"):
    """
    Part 3: Test whether the adversarial image from Part 2 transfers
    to Ollama's Q4-quantized deployment (black-box, no gradient access).
    """
    banner("PART 3: CROSS-MODEL TRANSFER (Ollama Q4 quantized)")

    import requests
    from attacks.blackbox_attack import ollama_score, DOG_KEYWORDS as KW

    # --- Check Ollama ---
    step(1, 4, "Checking Ollama...")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        names = [m["name"] for m in models]
        print(f"        Host:    {host}")
        print(f"        Models:  {names}")
        if not any("moondream" in n for n in names):
            print("        [SKIP] No moondream model in Ollama")
            return None
    except Exception as e:
        print(f"        [SKIP] Ollama not reachable: {e}")
        return None

    # --- Clean query ---
    step(2, 4, "Querying Ollama with CLEAN image...")
    _, clean_text, _ = ollama_score(clean_pil, host=host, num_predict=60)
    print(f'\n        >>> OLLAMA CLEAN: "{trunc(clean_text, 120)}"')
    clean_dog = any(kw in clean_text.lower() for kw in KW)
    print(f"        Dog keyword: {clean_dog}")

    # --- Adversarial query ---
    step(3, 4, "Querying Ollama with ADVERSARIAL image...")
    _, adv_text, _ = ollama_score(adv_pil, host=host, num_predict=60)
    print(f'\n        >>> OLLAMA ADV:   "{trunc(adv_text, 120)}"')
    adv_dog = any(kw in adv_text.lower() for kw in KW)
    adv_cat = "cat" in adv_text.lower()
    print(f"        Dog keyword: {adv_dog}")
    print(f"        Cat keyword: {adv_cat}")

    # --- Analysis ---
    step(4, 4, "Transfer analysis")
    print()
    table_header("Image", "Ollama Q4 Output")
    box("CLEAN", clean_text)
    box("ADVERSARIAL", adv_text)
    table_footer()
    print()
    print(f"  Clean says dog:     {clean_dog}")
    print(f"  Adv still says dog: {adv_dog}")
    print(f"  Adv says cat:       {adv_cat}")

    if adv_dog:
        print()
        print("  >> Q4 quantization BLOCKS the transfer.")
        print("     4-bit weights shift the decision boundary enough")
        print("     to break the full-precision perturbation.")
    elif adv_cat:
        print()
        print("  >> Transfer SUCCEEDED!")
        print("     The adversarial perturbation also fools the")
        print("     Q4 quantized model.")
    else:
        print()
        print("  >> Partial: dog keyword removed, but 'cat' not present.")

    data = {
        "ollama_clean": clean_text,
        "ollama_adv": adv_text,
        "clean_has_dog": clean_dog,
        "adv_has_dog": adv_dog,
        "adv_has_cat": adv_cat,
    }
    jpath = os.path.join(output_dir, "transfer_results.json")
    with open(jpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results: {jpath}")
    return data


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(classical, vlm, transfer, output_dir):
    banner("DEMONSTRATION COMPLETE")

    print()
    print("=" * W)
    print("  SUMMARY")
    print("=" * W)

    if classical:
        print()
        print("  PART 1 - ResNet-50 (Classical Classifier):")
        ok = sum(1 for _, r in classical if r["prediction_changed"])
        print(f"    Attacks:    FGSM, BIM, PGD")
        print(f"    Success:    {ok}/{len(classical)} changed prediction")
        print(f"    Budget:     L-inf = 8/255 (~3% per pixel)")

    if vlm:
        print()
        print("  PART 2 - Moondream2 (Vision-Language Model):")
        print(f'    Clean:      "{trunc(vlm["clean_desc"], 50)}"')
        print(f'    Adversarial: "{trunc(vlm["adv_desc"], 50)}"')
        print(f"    L-inf:      {vlm['details']['linf']*255:.2f}/255")
        print(f"    Succeeded:  {vlm['succeeded']}")

    if transfer:
        print()
        print("  PART 3 - Ollama Q4 (Cross-Model Transfer):")
        print(f"    Clean dog:  {transfer['clean_has_dog']}")
        print(f"    Adv dog:    {transfer['adv_has_dog']}")
        print(f"    Adv cat:    {transfer['adv_has_cat']}")

    print()
    print("=" * W)
    print("  KEY TAKEAWAYS")
    print("=" * W)
    print()
    print("  1. Imperceptible perturbations (8/255 per pixel) can")
    print("     completely change a model's output.")
    print()
    print("  2. White-box PGD is highly effective with gradient access.")
    print()
    print("  3. Cross-model transfer is NOT guaranteed - quantization")
    print("     and deployment differences can block attacks.")
    print()
    print("  4. Multi-level attacks (vision + alignment + language)")
    print("     are more effective than single-level on VLMs.")
    print()

    print("=" * W)
    print("  SAVED FILES")
    print("=" * W)
    print(f"  Directory: {output_dir}/")
    print()

    if classical:
        print("  Classical:")
        print(f"    classical_clean.png, classical_fgsm.png,")
        print(f"    classical_bim.png, classical_pgd.png")
        print()

    if vlm:
        print("  VLM:")
        print(f"    vlm_clean.png, vlm_adversarial.png,")
        print(f"    vlm_perturbation_10x.png, vlm_results.json")
        print()

    if transfer:
        print("  Transfer:")
        print(f"    transfer_results.json")
        print()

    print("=" * W)
    print("  Done.")
    print("=" * W)


# ============================================================
# MAIN
# ============================================================

def discover_images():
    """Find all usable images in the project."""
    import glob as _glob
    images = []
    if os.path.exists("dog.jpg"):
        images.append("dog.jpg")
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(sorted(_glob.glob(f"data/vlm/{ext}")))
    return images


def get_clean_description_ollama(image_path,
                                 host="http://127.0.0.1:11435"):
    """Query Ollama moondream for a description of the image."""
    import base64
    import requests

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "moondream",
        "prompt": "What do you see in this image?",
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(
        f"{host}/api/generate", json=payload, timeout=120
    )
    return resp.json().get("response", "").strip()


def generate_target_with_llm(clean_description,
                             llm_host="http://127.0.0.1:11471",
                             llm_model="vyas"):
    """
    Ask a text LLM to generate a target description that is
    completely different from the clean description.

    Returns a single-sentence target string.
    """
    import requests

    system_msg = (
        "You are a creative assistant. Given an image description, "
        "generate a completely DIFFERENT one-sentence image description "
        "that describes entirely unrelated subject matter. "
        "Do NOT mention any object, animal, or scene from the original. "
        "Reply with ONLY the description sentence, no preamble."
    )
    user_msg = f'Original description: "{clean_description}"'

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.8,
        "max_tokens": 60,
    }
    resp = requests.post(
        f"{llm_host}/v1/chat/completions",
        json=payload,
        timeout=30,
    )
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = text.strip('"').strip("'")
    if not text.endswith("."):
        text += "."
    return text


def auto_generate_target(image_path, ollama_host="http://127.0.0.1:11435",
                         llm_host="http://127.0.0.1:11471"):
    """
    Full pipeline:
      1. Get clean description from moondream (Ollama)
      2. Ask text LLM to generate a contrasting target
    """
    banner("AUTO-GENERATING TARGET TEXT")

    step(1, 2, f"Querying moondream for image description...")
    print(f"        Image:  {image_path}")
    clean_desc = get_clean_description_ollama(image_path, host=ollama_host)
    print(f'        Clean:  "{trunc(clean_desc, 100)}"')

    step(2, 2, "Asking LLM to generate a contrasting target...")
    target = generate_target_with_llm(clean_desc, llm_host=llm_host)
    print(f'        Target: "{target}"')

    return target


def select_image_interactive():
    """Let the user pick an image from the available list."""
    images = discover_images()
    if not images:
        print("  [ERROR] No images found in the project.")
        print("         Place a .jpg in the project root or data/vlm/")
        sys.exit(1)

    print()
    print("  Available images:")
    print()
    for i, path in enumerate(images, 1):
        print(f"    {i:>2}. {path}")
    print()
    print("    0. Type a custom path")
    print()

    while True:
        try:
            choice = input("  Select image [number]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

        if choice == "0":
            custom = input("  Enter image path: ").strip()
            if custom and os.path.exists(custom):
                return custom
            print(f"  [ERROR] File not found: {custom}")
            continue

        try:
            idx = int(choice)
            if 1 <= idx <= len(images):
                return images[idx - 1]
        except ValueError:
            pass

        print(f"  [ERROR] Invalid choice. Enter 0-{len(images)}.")


def main():
    p = argparse.ArgumentParser(
        description="Drishti - Adversarial Robustness Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                      # Interactive image pick, LLM auto-generates target
  python demo.py --quick              # Fast (~2 min, 50 iters)
  python demo.py --classical-only     # Part 1 only (~1 sec)
  python demo.py --skip-transfer      # Parts 1 + 2
  python demo.py --image data/vlm/dog03.jpg --target "A bird flying in the blue sky"
        """,
    )
    p.add_argument("--image", default=None,
                   help="Image path (if omitted, prompts interactively)")
    p.add_argument("--vlm-image", default=None,
                   help="Image for VLM attack (default: same as --image)")
    p.add_argument("--target", default=None,
                   help="Target text for VLM attack (if omitted, auto-generated by LLM)")
    p.add_argument("--epsilon", type=float, default=8/255,
                   help="L-inf budget (default: 8/255)")
    p.add_argument("--iterations", type=int, default=300,
                   help="PGD iterations (default: 300)")
    p.add_argument("--output-dir", default="outputs/demo",
                   help="Output directory")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11435",
                   help="Ollama host")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: 50 iterations, skip nothing")
    p.add_argument("--classical-only", action="store_true",
                   help="Only Part 1 (ResNet-50)")
    p.add_argument("--skip-classical", action="store_true",
                   help="Skip Part 1")
    p.add_argument("--skip-transfer", action="store_true",
                   help="Skip Part 3 (Ollama)")
    p.add_argument("--no-env-check", action="store_true",
                   help="Skip environment check")
    args = p.parse_args()

    # Interactive image selection if not provided
    if args.image is None:
        args.image = select_image_interactive()

    # Auto-generate target text with LLM if not provided
    if args.target is None:
        args.target = auto_generate_target(
            args.image, ollama_host=args.ollama_host
        )

    # Quick mode
    if args.quick:
        args.iterations = 50

    vlm_image = args.vlm_image or args.image
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Header
    print()
    print("=" * W)
    print("  DRISHTI - ADVERSARIAL ROBUSTNESS DEMONSTRATION")
    print("=" * W)
    print()
    print(f"  Image:       {args.image}")
    print(f"  Target:      \"{args.target}\"")
    print(f"  Epsilon:     {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Output:      {out}/")
    print()
    run_parts = []
    if not args.skip_classical:
        run_parts.append(1)
        print("  [x] Part 1: Classical (ResNet-50) - FGSM/BIM/PGD")
    if not args.classical_only:
        run_parts.append(2)
        print("  [x] Part 2: VLM (Moondream2) - multi-level PGD")
        if not args.skip_transfer:
            run_parts.append(3)
            print("  [x] Part 3: Transfer (Ollama Q4)")
        else:
            print("  [ ] Part 3: SKIPPED")
    else:
        print("  [ ] Part 2: SKIPPED")
        print("  [ ] Part 3: SKIPPED")

    # Environment check
    env = {}
    if not args.no_env_check:
        env = check_env()

    # ========================================================
    # RUN
    # ========================================================

    classical = None
    vlm = None
    transfer = None

    # --- Part 1 ---
    if 1 in run_parts:
        try:
            classical = part1_classical(args.image, out, args.epsilon)
        except Exception as e:
            print(f"\n  [ERROR] Part 1 failed: {e}")
            traceback.print_exc()

    # --- Part 2 ---
    if 2 in run_parts:
        if not env.get("cuda", True):
            print("\n  [SKIP] Part 2 requires CUDA")
        elif not env.get("transformers", True):
            print("\n  [SKIP] Part 2 requires transformers")
        else:
            try:
                vlm = part2_vlm(
                    vlm_image, args.target, out,
                    args.epsilon, args.iterations,
                )
            except Exception as e:
                print(f"\n  [ERROR] Part 2 failed: {e}")
                traceback.print_exc()

    # --- Part 3 ---
    if 3 in run_parts:
        if not env.get("ollama", False) and not args.no_env_check:
            print("\n  [SKIP] Part 3 requires Ollama with moondream model")
        elif vlm is None:
            print("\n  [SKIP] Part 3 requires Part 2 to succeed")
        else:
            try:
                transfer = part3_transfer(
                    vlm["clean_pil"], vlm["adv_pil"], out,
                    args.ollama_host,
                )
            except Exception as e:
                print(f"\n  [ERROR] Part 3 failed: {e}")
                traceback.print_exc()

    # ========================================================
    # SUMMARY
    # ========================================================

    final_summary(classical, vlm, transfer, out)


if __name__ == "__main__":
    main()
