"""
Architecture-Level Attack Evaluation: White-Box HF moondream2 -> Black-Box Ollama

This experiment creates adversarial perturbations using full white-box
access to the HuggingFace moondream2 model (all three pipeline stages:
vision encoder, connector, LLM) and evaluates:

  1. White-box success -- does HF moondream2 change its description?
  2. Black-box transfer -- does Ollama moondream (Q4 quantized) change?
  3. Keyword-based metrics -- source/target keyword presence
  4. Perturbation budget -- L-infinity, L2, mean absolute

Attack:
  - White-box model: HuggingFace moondream2 (bfloat16, full gradients)
  - Black-box target: Ollama moondream (Q4 quantized, API-only)
  - Attack: Multi-level PGD (L_vision + L_alignment + L_language)

Output directory structure:
  OUTPUT_DIR/
    adversarial_examples/   -- adv_{filename} PNG images
    results/                -- per-image JSON + aggregate JSON
    visualizations/         -- side-by-side original/adv/perturbation
    logs/                   -- attack loss history per image

Usage:
    PYTHONPATH=. python experiments/moondream_attack_eval.py
    PYTHONPATH=. python experiments/moondream_attack_eval.py --targeted "A cat"
    PYTHONPATH=. python experiments/moondream_attack_eval.py --untargeted
    PYTHONPATH=. python experiments/moondream_attack_eval.py --image dog.jpg
    PYTHONPATH=. python experiments/moondream_attack_eval.py --output-dir outputs/my_run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from attacks.moondream_pgd import moondream_pgd
from models.moondream_adapter import MoondreamAdapter
from models.ollama_adapter import OllamaVLMAdapter


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dog.jpg"
DATASET_DIR = "data/vlm"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

SOURCE_TEXT = "a photo of a dog"
TARGET_TEXT = "A cat sitting on a couch"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 300
SEED = 42

LAMBDA_VISION = 1.0
LAMBDA_ALIGNMENT = 1.0
LAMBDA_LANGUAGE = 5.0

OLLAMA_HOST = "http://127.0.0.1:11435"
OLLAMA_MODEL = "moondream"

OUTPUT_DIR = "outputs/moondream_attack_targeted"

ATTACK_SIZE = 378


# ============================================================
# HELPERS
# ============================================================

def load_image_raw(path, attack_size=ATTACK_SIZE):
    """Load an image as PIL + [1, 3, H, W] float tensor in [0, 1].

    If attack_size is set, resize to attack_size x attack_size for
    consistent single-crop processing across attack and evaluation.
    """
    pil_image = Image.open(path).convert("RGB")
    if attack_size:
        pil_image = pil_image.resize((attack_size, attack_size), Image.LANCZOS)
    tensor = (
        torch.from_numpy(np.array(pil_image))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    return pil_image, tensor.unsqueeze(0)


def tensor_to_pil(tensor):
    """Convert [1, 3, H, W] tensor to PIL."""
    tensor = tensor[0].detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def calculate_perturbation(original, adversarial):
    """Compute L-inf, L2, and mean absolute perturbation."""
    perturbation = adversarial - original
    linf = perturbation.abs().max().item()
    l2 = (
        torch.norm(
            perturbation.reshape(perturbation.shape[0], -1),
            p=2,
            dim=1,
        ).item()
    )
    mean_abs = perturbation.abs().mean().item()
    return {"linf": linf, "l2": l2, "mean_abs": mean_abs}


def get_dataset_paths():
    """Return sorted image paths from DATASET_DIR."""
    if not os.path.isdir(DATASET_DIR):
        return []
    paths = []
    for filename in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, filename)
        if (
            os.path.isfile(path)
            and filename.lower().endswith(IMAGE_EXTENSIONS)
        ):
            paths.append(path)
    return paths


def descriptions_match(desc1, desc2):
    """Check if two descriptions are effectively the same."""
    return desc1.strip().lower() == desc2.strip().lower()


def keyword_analysis(desc, source_kw, target_kw):
    """Check for source/target keywords in description."""
    desc_lower = desc.lower()
    return {
        "has_source": source_kw.lower() in desc_lower,
        "has_target": target_kw.lower() in desc_lower,
    }


def save_visualization(original_pil, adv_pil, metrics, filepath, filename):
    """Save a side-by-side visualization: original | adversarial | perturbation (amplified)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_arr = np.array(original_pil).astype(np.float32)
    adv_arr = np.array(adv_pil).astype(np.float32)
    perturbation = np.abs(adv_arr - orig_arr)

    # Amplify perturbation for visibility
    max_pert = perturbation.max()
    if max_pert > 0:
        pert_amp = np.clip(perturbation / max_pert * 255.0, 0, 255).astype(np.uint8)
    else:
        pert_amp = np.zeros_like(orig_arr, dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Adversarial Example: {filename}", fontsize=14, fontweight="bold")

    axes[0].imshow(orig_arr.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(adv_arr.astype(np.uint8))
    axes[1].set_title("Adversarial")
    axes[1].axis("off")

    axes[2].imshow(pert_amp)
    axes[2].set_title(
        f"Perturbation (amplified)\n"
        f"L-inf={metrics['linf']:.6f}  L2={metrics['l2']:.4f}  "
        f"mean={metrics['mean_abs']:.6f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(data, filepath):
    """Save data as JSON with indent."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# OUTPUT DIRECTORY SETUP
# ============================================================

def setup_output_dirs(base_dir):
    """Create the output directory structure."""
    dirs = {
        "base": base_dir,
        "adversarial": os.path.join(base_dir, "adversarial_examples"),
        "results": os.path.join(base_dir, "results"),
        "visualizations": os.path.join(base_dir, "visualizations"),
        "logs": os.path.join(base_dir, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# SINGLE IMAGE EVALUATION
# ============================================================

def evaluate_single_image(
    image_path,
    hf_model,
    ollama,
    output_dirs,
    target_text=None,
):
    """
    Full attack evaluation for one image:
      1. Clean -> HF moondream2 (white-box baseline)
      2. Clean -> Ollama moondream (black-box baseline)
      3. Generate adversarial via multi-level PGD on HF model
      4. Adversarial -> HF moondream2 (white-box success)
      5. Adversarial -> Ollama moondream (black-box transfer)
    """
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]
    targeted = target_text is not None

    print()
    print("=" * 70)
    print(f"IMAGE: {filename}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    pil_image, image_tensor = load_image_raw(image_path)
    print(f"  Image size: {pil_image.size}")

    source_kw = SOURCE_TEXT.replace("a photo of ", "").strip()
    target_kw = TARGET_TEXT.split()[0].lower()  # "cat" from "A cat sitting..."

    # --------------------------------------------------------
    # STEP 1: CLEAN -- HF MOONDREAM2 (WHITE-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 1: CLEAN EVALUATION -- HF MOONDREAM2 (WHITE-BOX)")
    print("-" * 70)

    t0 = time.time()
    clean_hf_desc = hf_model.describe(pil_image)
    hf_time = time.time() - t0
    print(f"  Description: {clean_hf_desc}")
    print(f"  Time: {hf_time:.2f}s")

    clean_hf_kw = keyword_analysis(clean_hf_desc, source_kw, target_kw)
    print(f"  Keywords: source={clean_hf_kw['has_source']}, "
          f"target={clean_hf_kw['has_target']}")

    # --------------------------------------------------------
    # STEP 2: CLEAN -- OLLAMA MOONDREAM (BLACK-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 2: CLEAN EVALUATION -- OLLAMA MOONDREAM (BLACK-BOX)")
    print("-" * 70)

    clean_ollama_desc = ollama.describe_image(pil_image)
    print(f"  Description: {clean_ollama_desc}")

    clean_ollama_kw = keyword_analysis(
        clean_ollama_desc, source_kw, target_kw
    )
    print(f"  Keywords: source={clean_ollama_kw['has_source']}, "
          f"target={clean_ollama_kw['has_target']}")

    # --------------------------------------------------------
    # STEP 3: GENERATE ADVERSARIAL IMAGE
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 3: GENERATING ADVERSARIAL IMAGE (MULTI-LEVEL PGD)")
    print("-" * 70)

    print(f"  Attack: {'targeted' if targeted else 'untargeted'}")
    if targeted:
        print(f"  Target: {target_text}")
    print(f"  Epsilon: {EPSILON:.6f}")
    print(f"  Alpha: {ALPHA:.6f}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Lambda: vision={LAMBDA_VISION}, "
          f"alignment={LAMBDA_ALIGNMENT}, "
          f"language={LAMBDA_LANGUAGE}")

    t0 = time.time()

    attack_result = moondream_pgd(
        model=hf_model,
        image=image_tensor.to(hf_model.device),
        target_text=target_text if targeted else None,
        epsilon=EPSILON,
        alpha=ALPHA,
        iterations=ITERATIONS,
        lambda_vision=LAMBDA_VISION,
        lambda_alignment=LAMBDA_ALIGNMENT,
        lambda_language=LAMBDA_LANGUAGE,
        random_start=True,
        seed=SEED,
        return_details=True,
    )

    adversarial, attack_details = attack_result
    attack_time = time.time() - t0
    print(f"  Attack time: {attack_time:.2f}s")

    # --------------------------------------------------------
    # PERTURBATION METRICS
    # --------------------------------------------------------

    metrics = calculate_perturbation(
        image_tensor.to(adversarial.device),
        adversarial,
    )

    print(
        f"  L-inf: {metrics['linf']:.8f} "
        f"(budget: {EPSILON:.8f}) "
        f"within: {'YES' if metrics['linf'] <= EPSILON + 1e-6 else 'NO'}"
    )
    print(f"  L2: {metrics['l2']:.8f}")
    print(f"  Mean |delta|: {metrics['mean_abs']:.8f}")

    # --------------------------------------------------------
    # SAVE ADVERSARIAL IMAGE
    # --------------------------------------------------------

    adv_pil = tensor_to_pil(adversarial)
    adv_img_path = os.path.join(
        output_dirs["adversarial"], f"adv_{stem}.png"
    )
    adv_pil.save(adv_img_path)
    print(f"  Saved adversarial image: {adv_img_path}")

    # Save visualization
    viz_path = os.path.join(
        output_dirs["visualizations"], f"viz_{stem}.png"
    )
    save_visualization(pil_image, adv_pil, metrics, viz_path, filename)
    print(f"  Saved visualization: {viz_path}")

    # Save attack log (loss history)
    log_path = os.path.join(output_dirs["logs"], f"attack_log_{stem}.json")
    attack_log = {
        "filename": filename,
        "attack_time": attack_time,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "iterations": ITERATIONS,
        "mode": "targeted" if targeted else "untargeted",
        "target_text": target_text,
        "seed": SEED,
        "lambdas": {
            "vision": LAMBDA_VISION,
            "alignment": LAMBDA_ALIGNMENT,
            "language": LAMBDA_LANGUAGE,
        },
        "perturbation_metrics": metrics,
        "clean_token": attack_details.get("clean_token"),
        "target_token": attack_details.get("target_token"),
        "history": attack_details.get("history", []),
    }
    save_json(attack_log, log_path)
    print(f"  Saved attack log: {log_path}")

    # --------------------------------------------------------
    # STEP 4: ADVERSARIAL -- HF MOONDREAM2 (WHITE-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 4: ADVERSARIAL EVALUATION -- HF MOONDREAM2 (WHITE-BOX)")
    print("-" * 70)

    adv_hf_desc = hf_model.describe(adv_pil)
    print(f"  Description: {adv_hf_desc}")

    adv_hf_kw = keyword_analysis(adv_hf_desc, source_kw, target_kw)
    print(f"  Keywords: source={adv_hf_kw['has_source']}, "
          f"target={adv_hf_kw['has_target']}")

    hf_desc_changed = not descriptions_match(
        clean_hf_desc, adv_hf_desc
    )
    hf_source_removed = (
        clean_hf_kw["has_source"] and not adv_hf_kw["has_source"]
    )
    hf_target_appeared = (
        not clean_hf_kw["has_target"] and adv_hf_kw["has_target"]
    )

    print(f"  Description changed: {'YES' if hf_desc_changed else 'NO'}")
    print(f"  Source removed: {'YES' if hf_source_removed else 'NO'}")
    print(f"  Target appeared: {'YES' if hf_target_appeared else 'NO'}")

    # --------------------------------------------------------
    # STEP 5: ADVERSARIAL -- OLLAMA MOONDREAM (BLACK-BOX)
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 5: ADVERSARIAL EVALUATION -- OLLAMA MOONDREAM (BLACK-BOX)")
    print("-" * 70)

    adv_ollama_desc = ollama.describe_image(adv_pil)
    print(f"  Description: {adv_ollama_desc}")

    adv_ollama_kw = keyword_analysis(
        adv_ollama_desc, source_kw, target_kw
    )
    print(f"  Keywords: source={adv_ollama_kw['has_source']}, "
          f"target={adv_ollama_kw['has_target']}")

    ollama_desc_changed = not descriptions_match(
        clean_ollama_desc, adv_ollama_desc
    )
    ollama_source_removed = (
        clean_ollama_kw["has_source"]
        and not adv_ollama_kw["has_source"]
    )
    ollama_target_appeared = (
        not clean_ollama_kw["has_target"]
        and adv_ollama_kw["has_target"]
    )

    print(f"  Description changed: {'YES' if ollama_desc_changed else 'NO'}")
    print(f"  Source removed: {'YES' if ollama_source_removed else 'NO'}")
    print(f"  Target appeared: {'YES' if ollama_target_appeared else 'NO'}")

    # --------------------------------------------------------
    # BUILD RESULT DICT
    # --------------------------------------------------------

    result = {
        "filename": filename,
        "image_path": image_path,
        "image_size": list(pil_image.size),
        # Clean descriptions
        "clean_hf_desc": clean_hf_desc,
        "clean_ollama_desc": clean_ollama_desc,
        # Adversarial descriptions
        "adv_hf_desc": adv_hf_desc,
        "adv_ollama_desc": adv_ollama_desc,
        # Keyword analysis
        "clean_hf_source": clean_hf_kw["has_source"],
        "clean_hf_target": clean_hf_kw["has_target"],
        "adv_hf_source": adv_hf_kw["has_source"],
        "adv_hf_target": adv_hf_kw["has_target"],
        "clean_ollama_source": clean_ollama_kw["has_source"],
        "clean_ollama_target": clean_ollama_kw["has_target"],
        "adv_ollama_source": adv_ollama_kw["has_source"],
        "adv_ollama_target": adv_ollama_kw["has_target"],
        # Success metrics
        "hf_desc_changed": hf_desc_changed,
        "hf_source_removed": hf_source_removed,
        "hf_target_appeared": hf_target_appeared,
        "ollama_desc_changed": ollama_desc_changed,
        "ollama_source_removed": ollama_source_removed,
        "ollama_target_appeared": ollama_target_appeared,
        # Perturbation
        "linf": metrics["linf"],
        "l2": metrics["l2"],
        "mean_abs": metrics["mean_abs"],
        "epsilon": EPSILON,
        "within_budget": metrics["linf"] <= EPSILON + 1e-6,
        # Attack config
        "attack_mode": "targeted" if targeted else "untargeted",
        "target_text": target_text,
        "iterations": ITERATIONS,
        "alpha": ALPHA,
        "seed": SEED,
        "lambda_vision": LAMBDA_VISION,
        "lambda_alignment": LAMBDA_ALIGNMENT,
        "lambda_language": LAMBDA_LANGUAGE,
        "attack_size": ATTACK_SIZE,
        "attack_time": attack_time,
        "hf_clean_time": hf_time,
        # Output paths
        "adv_image_path": adv_img_path,
        "visualization_path": viz_path,
        "attack_log_path": log_path,
    }

    # Save per-image result JSON
    result_path = os.path.join(
        output_dirs["results"], f"result_{stem}.json"
    )
    save_json(result, result_path)
    print(f"  Saved result: {result_path}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    global ITERATIONS, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Architecture-level attack on moondream2"
    )
    parser.add_argument(
        "--targeted",
        type=str,
        default=TARGET_TEXT,
        help="Target text for targeted attack (default: 'A cat sitting on a couch')",
    )
    parser.add_argument(
        "--untargeted",
        action="store_true",
        help="Run untargeted attack",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Number of PGD iterations",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image path to evaluate (skip dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Base output directory",
    )
    args = parser.parse_args()

    # Determine attack mode
    if args.targeted:
        target_text = args.targeted
        mode = "targeted"
    else:
        target_text = None
        mode = "untargeted"

    ITERATIONS = args.iterations
    OUTPUT_DIR = args.output_dir

    # Setup output directories
    output_dirs = setup_output_dirs(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("ARCHITECTURE-LEVEL ATTACK: MOONDREAM2")
    print("=" * 70)

    print()
    print("CONFIGURATION")
    print(f"  Attack mode      : {mode}")
    if target_text:
        print(f"  Target text      : {target_text}")
    print(f"  Source text      : {SOURCE_TEXT}")
    print(f"  Epsilon          : {EPSILON:.6f}")
    print(f"  Alpha            : {ALPHA:.6f}")
    print(f"  Iterations       : {ITERATIONS}")
    print(f"  Lambda vision    : {LAMBDA_VISION}")
    print(f"  Lambda alignment : {LAMBDA_ALIGNMENT}")
    print(f"  Lambda language  : {LAMBDA_LANGUAGE}")
    print(f"  Attack size      : {ATTACK_SIZE}x{ATTACK_SIZE}")
    print(f"  White-box model  : HuggingFace moondream2 (bfloat16)")
    print(f"  Black-box model  : Ollama moondream (Q4)")
    print(f"  Ollama host      : {OLLAMA_HOST}")
    print()
    print("OUTPUT DIRECTORIES")
    print(f"  Base            : {output_dirs['base']}")
    print(f"  Adversarial     : {output_dirs['adversarial']}")
    print(f"  Results         : {output_dirs['results']}")
    print(f"  Visualizations  : {output_dirs['visualizations']}")
    print(f"  Logs            : {output_dirs['logs']}")

    # ========================================================
    # LOAD MODELS
    # ========================================================

    print()
    print("-" * 70)
    print("LOADING MODELS")
    print("-" * 70)

    print("  Loading HuggingFace moondream2 (white-box)...")
    hf_model = MoondreamAdapter()
    print(f"  Device: {hf_model.device}")

    print("  Connecting to Ollama moondream (black-box)...")
    ollama = OllamaVLMAdapter(
        model_name=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        name="moondream-ollama",
        temperature=0.5,
        num_predict=200,
    )
    print(f"  Host: {OLLAMA_HOST}")

    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = [IMAGE_PATH]
        dataset_paths = get_dataset_paths()
        if dataset_paths:
            image_paths.extend(dataset_paths)

    seen = set()
    unique_paths = []
    for p in image_paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)
    image_paths = unique_paths

    print()
    print(f"  Images to evaluate: {len(image_paths)}")
    for p in image_paths:
        print(f"    - {p}")

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    results = []
    start_time = time.time()

    for image_path in image_paths:
        try:
            result = evaluate_single_image(
                image_path=image_path,
                hf_model=hf_model,
                ollama=ollama,
                output_dirs=output_dirs,
                target_text=target_text,
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if not results:
        print()
        print("No images were successfully processed.")
        return

    total = len(results)

    print()
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'Image':<20}"
        f"{'HF-Change':<12}"
        f"{'Ol-Change':<12}"
        f"{'HF-SrcRm':<10}"
        f"{'Ol-SrcRm':<10}"
        f"{'L-inf':<12}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['filename']:<20}"
            f"{'YES' if r['hf_desc_changed'] else 'NO':<12}"
            f"{'YES' if r['ollama_desc_changed'] else 'NO':<12}"
            f"{'YES' if r['hf_source_removed'] else 'NO':<10}"
            f"{'YES' if r['ollama_source_removed'] else 'NO':<10}"
            f"{r['linf']:<12.6f}"
        )

    # ========================================================
    # AGGREGATE
    # ========================================================

    hf_changed = sum(1 for r in results if r["hf_desc_changed"])
    ollama_changed = sum(1 for r in results if r["ollama_desc_changed"])
    hf_src_rm = sum(1 for r in results if r["hf_source_removed"])
    ollama_src_rm = sum(1 for r in results if r["ollama_source_removed"])
    hf_tgt_app = sum(1 for r in results if r["hf_target_appeared"])
    ollama_tgt_app = sum(1 for r in results if r["ollama_target_appeared"])

    avg_linf = np.mean([r["linf"] for r in results])
    avg_l2 = np.mean([r["l2"] for r in results])
    avg_time = np.mean([r["attack_time"] for r in results])

    print()
    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print()
    print(f"  Images evaluated           : {total}")
    print(f"  Attack mode                : {mode}")
    print(f"  Total time                 : {total_time:.1f}s")
    print()
    print(f"  White-box (HF) results:")
    print(f"    Description changed      : {hf_changed}/{total} ({hf_changed/total*100:.1f}%)")
    print(f"    Source keyword removed   : {hf_src_rm}/{total} ({hf_src_rm/total*100:.1f}%)")
    print(f"    Target keyword appeared  : {hf_tgt_app}/{total} ({hf_tgt_app/total*100:.1f}%)")
    print()
    print(f"  Black-box (Ollama) transfer:")
    print(f"    Description changed      : {ollama_changed}/{total} ({ollama_changed/total*100:.1f}%)")
    print(f"    Source keyword removed   : {ollama_src_rm}/{total} ({ollama_src_rm/total*100:.1f}%)")
    print(f"    Target keyword appeared  : {ollama_tgt_app}/{total} ({ollama_tgt_app/total*100:.1f}%)")
    print()
    print(f"  Average L-inf              : {avg_linf:.8f}")
    print(f"  Average L2                 : {avg_l2:.8f}")
    print(f"  Average attack time        : {avg_time:.2f}s")

    # ========================================================
    # DESCRIPTIONS COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("DESCRIPTIONS COMPARISON")
    print("=" * 70)

    for r in results:
        print()
        print(f"  {r['filename']}:")
        print(f"    Clean HF:     {r['clean_hf_desc']}")
        print(f"    Adv HF:       {r['adv_hf_desc']}")
        print(f"    Clean Ollama: {r['clean_ollama_desc']}")
        print(f"    Adv Ollama:   {r['adv_ollama_desc']}")

    # ========================================================
    # SAVE AGGREGATE RESULTS JSON
    # ========================================================

    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "total_time": total_time,
        "configuration": {
            "attack_mode": mode,
            "target_text": target_text,
            "source_text": SOURCE_TEXT,
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "iterations": ITERATIONS,
            "seed": SEED,
            "lambda_vision": LAMBDA_VISION,
            "lambda_alignment": LAMBDA_ALIGNMENT,
            "lambda_language": LAMBDA_LANGUAGE,
            "attack_size": ATTACK_SIZE,
            "white_box_model": "HuggingFace moondream2 (bfloat16)",
            "black_box_model": "Ollama moondream (Q4)",
        },
        "images_evaluated": total,
        "white_box_metrics": {
            "description_changed": hf_changed,
            "description_changed_pct": hf_changed / total * 100,
            "source_removed": hf_src_rm,
            "source_removed_pct": hf_src_rm / total * 100,
            "target_appeared": hf_tgt_app,
            "target_appeared_pct": hf_tgt_app / total * 100,
        },
        "black_box_metrics": {
            "description_changed": ollama_changed,
            "description_changed_pct": ollama_changed / total * 100,
            "source_removed": ollama_src_rm,
            "source_removed_pct": ollama_src_rm / total * 100,
            "target_appeared": ollama_tgt_app,
            "target_appeared_pct": ollama_tgt_app / total * 100,
        },
        "perturbation": {
            "avg_linf": avg_linf,
            "avg_l2": avg_l2,
            "avg_attack_time": avg_time,
        },
        "per_image_results": results,
    }

    aggregate_path = os.path.join(
        output_dirs["results"], "aggregate_results.json"
    )
    save_json(aggregate, aggregate_path)
    print()
    print(f"  Saved aggregate results: {aggregate_path}")

    # Save summary text report
    report_path = os.path.join(output_dirs["base"], "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MOONDREAM2 ADVERSARIAL ATTACK SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Images evaluated: {total}\n")
        f.write(f"Attack mode: {mode}\n\n")

        f.write("Configuration:\n")
        f.write(f"  epsilon={EPSILON:.6f}  alpha={ALPHA:.6f}  "
                f"iterations={ITERATIONS}\n")
        f.write(f"  lambda: vision={LAMBDA_VISION} "
                f"alignment={LAMBDA_ALIGNMENT} "
                f"language={LAMBDA_LANGUAGE}\n")
        f.write(f"  attack_size: {ATTACK_SIZE}x{ATTACK_SIZE}\n\n")

        f.write("White-box (HF moondream2) results:\n")
        f.write(f"  Description changed: {hf_changed}/{total} "
                f"({hf_changed/total*100:.1f}%)\n")
        f.write(f"  Source removed: {hf_src_rm}/{total} "
                f"({hf_src_rm/total*100:.1f}%)\n")
        f.write(f"  Target appeared: {hf_tgt_app}/{total} "
                f"({hf_tgt_app/total*100:.1f}%)\n\n")

        f.write("Black-box (Ollama moondream Q4) transfer:\n")
        f.write(f"  Description changed: {ollama_changed}/{total} "
                f"({ollama_changed/total*100:.1f}%)\n")
        f.write(f"  Source removed: {ollama_src_rm}/{total} "
                f"({ollama_src_rm/total*100:.1f}%)\n")
        f.write(f"  Target appeared: {ollama_tgt_app}/{total} "
                f"({ollama_tgt_app/total*100:.1f}%)\n\n")

        f.write(f"Perturbation:\n")
        f.write(f"  Average L-inf: {avg_linf:.8f}\n")
        f.write(f"  Average L2: {avg_l2:.8f}\n")
        f.write(f"  Average attack time: {avg_time:.2f}s\n\n")

        f.write("Per-image descriptions:\n")
        for r in results:
            f.write(f"\n  {r['filename']}:\n")
            f.write(f"    Clean HF:     {r['clean_hf_desc']}\n")
            f.write(f"    Adv HF:       {r['adv_hf_desc']}\n")
            f.write(f"    Clean Ollama: {r['clean_ollama_desc']}\n")
            f.write(f"    Adv Ollama:   {r['adv_ollama_desc']}\n")

    print(f"  Saved summary report: {report_path}")

    print()
    print("=" * 70)
    print("Attack evaluation complete.")
    print(f"  All outputs saved to: {output_dirs['base']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
