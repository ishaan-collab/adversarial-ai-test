"""
White-Box PGD Attack Evaluation: Qwen2-VL-2B

Creates adversarial perturbations using full white-box access to
HuggingFace Qwen2-VL-2B-Instruct (vision encoder + PatchMerger + LLM).

Attack:
  - White-box model: HuggingFace Qwen2-VL-2B-Instruct (bfloat16, full gradients)
  - Attack: Multi-level PGD (L_vision + L_alignment + L_language)
  - Epsilon: 8/255 (max ~3.1% per pixel -- visually imperceptible)
  - Iterations: 300, Momentum: 0.9 (MI-FGSM)

This is a surrogate for transfer attack to vision-vapt (Qwen2-VL MoE 100B+).
Same vision encoder family -> perturbations should transfer across scale.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/qwen_attack_eval.py
    PYTHONPATH=. .venv/bin/python experiments/qwen_attack_eval.py --image dog03.jpg
    PYTHONPATH=. .venv/bin/python experiments/qwen_attack_eval.py --iterations 500
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

from attacks.vlm_pgd_universal import MultiLevelPGDAttack
from models.qwen_vl_adapter import QwenVLAdapter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vlm",
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

SOURCE_TEXT = "dog"
TARGET_TEXT = "A cat sitting on a couch"

EPSILON = 8 / 255
ALPHA = 2 / 255
ITERATIONS = 300
SEED = 42
MOMENTUM = 0.9

LAMBDA_VISION = 1.0
LAMBDA_ALIGNMENT = 1.0
LAMBDA_LANGUAGE = 5.0

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "qwen_attack_targeted",
)


# ============================================================
# HELPERS
# ============================================================

def load_image_raw(path, attack_size):
    """Load an image as PIL + [1, 3, H, W] float tensor in [0, 1]."""
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
    model,
    output_dirs,
    target_text,
    epsilon,
    alpha,
    iterations,
    lambda_vision,
    lambda_alignment,
    lambda_language,
    momentum,
    seed,
):
    """Full attack evaluation for one image."""
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    print()
    print("=" * 70)
    print(f"IMAGE: {filename}")
    print("=" * 70)

    # --- Load image ---
    pil_image, image_tensor = load_image_raw(image_path, model.image_size)
    print(f"  Image size: {pil_image.size}")

    source_kw = SOURCE_TEXT
    target_kw = TARGET_TEXT.split()[1].lower()  # "cat"

    # --- Step 1: Clean baseline ---
    print()
    print("-" * 70)
    print("STEP 1: CLEAN EVALUATION -- Qwen2-VL-2B (WHITE-BOX)")
    print("-" * 70)

    t0 = time.time()
    clean_desc = model.describe(pil_image, max_tokens=100)
    clean_time = time.time() - t0
    print(f"  Description: {clean_desc}")
    print(f"  Time: {clean_time:.2f}s")

    clean_kw = keyword_analysis(clean_desc, source_kw, target_kw)
    print(f"  Keywords: source={clean_kw['has_source']}, "
          f"target={clean_kw['has_target']}")

    # --- Step 2: Generate adversarial ---
    print()
    print("-" * 70)
    print("STEP 2: GENERATING ADVERSARIAL IMAGE (MULTI-LEVEL PGD)")
    print("-" * 70)

    print(f"  Target: {target_text}")
    print(f"  Epsilon: {epsilon:.6f} ({epsilon*255:.1f}/255)")
    print(f"  Alpha: {alpha:.6f}")
    print(f"  Iterations: {iterations}")
    print(f"  Momentum: {momentum}")
    print(f"  Lambda: vision={lambda_vision}, "
          f"alignment={lambda_alignment}, "
          f"language={lambda_language}")

    attack = MultiLevelPGDAttack(
        adapter=model,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        momentum=momentum,
        random_start=True,
        seed=seed,
        lambda_vision=lambda_vision,
        lambda_alignment=lambda_alignment,
        lambda_language=lambda_language,
    )

    t0 = time.time()
    adv_tensor, attack_details = attack.attack(
        image_tensor.to(model.device),
        target_text=target_text,
        return_details=True,
        verbose=True,
    )
    attack_time = time.time() - t0
    print(f"  Attack time: {attack_time:.2f}s")

    # --- Perturbation metrics ---
    metrics = calculate_perturbation(
        image_tensor.to(adv_tensor.device),
        adv_tensor,
    )

    print(
        f"  L-inf: {metrics['linf']:.8f} "
        f"(budget: {epsilon:.8f}) "
        f"within: {'YES' if metrics['linf'] <= epsilon + 1e-6 else 'NO'}"
    )
    print(f"  L2: {metrics['l2']:.8f}")
    print(f"  Mean |delta|: {metrics['mean_abs']:.8f}")

    # --- Save adversarial image ---
    adv_pil = tensor_to_pil(adv_tensor)
    adv_img_path = os.path.join(
        output_dirs["adversarial"], f"adv_{stem}.png"
    )
    adv_pil.save(adv_img_path)
    print(f"  Saved adversarial image: {adv_img_path}")

    # --- Save visualization ---
    viz_path = os.path.join(
        output_dirs["visualizations"], f"viz_{stem}.png"
    )
    save_visualization(pil_image, adv_pil, metrics, viz_path, filename)
    print(f"  Saved visualization: {viz_path}")

    # --- Save attack log ---
    log_path = os.path.join(output_dirs["logs"], f"attack_log_{stem}.json")
    attack_log = {
        "filename": filename,
        "attack_time": attack_time,
        "epsilon": epsilon,
        "alpha": alpha,
        "iterations": iterations,
        "momentum": momentum,
        "mode": "targeted",
        "target_text": target_text,
        "seed": seed,
        "lambdas": {
            "vision": lambda_vision,
            "alignment": lambda_alignment,
            "language": lambda_language,
        },
        "perturbation_metrics": metrics,
        "history": attack_details.get("history", []),
    }
    save_json(attack_log, log_path)
    print(f"  Saved attack log: {log_path}")

    # --- Step 3: Adversarial evaluation ---
    print()
    print("-" * 70)
    print("STEP 3: ADVERSARIAL EVALUATION -- Qwen2-VL-2B (WHITE-BOX)")
    print("-" * 70)

    adv_desc = model.describe(adv_pil, max_tokens=100)
    print(f"  Description: {adv_desc}")

    adv_kw = keyword_analysis(adv_desc, source_kw, target_kw)
    print(f"  Keywords: source={adv_kw['has_source']}, "
          f"target={adv_kw['has_target']}")

    desc_changed = not descriptions_match(clean_desc, adv_desc)
    source_removed = (
        clean_kw["has_source"] and not adv_kw["has_source"]
    )
    target_appeared = (
        not clean_kw["has_target"] and adv_kw["has_target"]
    )

    print(f"  Description changed: {'YES' if desc_changed else 'NO'}")
    print(f"  Source removed: {'YES' if source_removed else 'NO'}")
    print(f"  Target appeared: {'YES' if target_appeared else 'NO'}")

    # --- Build result dict ---
    result = {
        "filename": filename,
        "image_path": image_path,
        "image_size": list(pil_image.size),
        "clean_desc": clean_desc,
        "adv_desc": adv_desc,
        "clean_source": clean_kw["has_source"],
        "clean_target": clean_kw["has_target"],
        "adv_source": adv_kw["has_source"],
        "adv_target": adv_kw["has_target"],
        "desc_changed": desc_changed,
        "source_removed": source_removed,
        "target_appeared": target_appeared,
        "linf": metrics["linf"],
        "l2": metrics["l2"],
        "mean_abs": metrics["mean_abs"],
        "epsilon": epsilon,
        "within_budget": metrics["linf"] <= epsilon + 1e-6,
        "attack_mode": "targeted",
        "target_text": target_text,
        "iterations": iterations,
        "alpha": alpha,
        "momentum": momentum,
        "seed": seed,
        "lambda_vision": lambda_vision,
        "lambda_alignment": lambda_alignment,
        "lambda_language": lambda_language,
        "attack_size": model.image_size,
        "attack_time": attack_time,
        "clean_time": clean_time,
        "adv_image_path": adv_img_path,
        "visualization_path": viz_path,
        "attack_log_path": log_path,
    }

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
    global ITERATIONS, OUTPUT_DIR, LAMBDA_VISION, LAMBDA_ALIGNMENT, LAMBDA_LANGUAGE

    parser = argparse.ArgumentParser(
        description="White-box PGD attack on Qwen2-VL-2B"
    )
    parser.add_argument(
        "--targeted",
        type=str,
        default=TARGET_TEXT,
        help="Target text for targeted attack",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Number of PGD iterations",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON,
        help="L-inf perturbation budget",
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
    parser.add_argument(
        "--lambda-vision",
        type=float,
        default=LAMBDA_VISION,
        help="Weight for vision feature loss",
    )
    parser.add_argument(
        "--lambda-alignment",
        type=float,
        default=LAMBDA_ALIGNMENT,
        help="Weight for connector alignment loss",
    )
    parser.add_argument(
        "--lambda-language",
        type=float,
        default=LAMBDA_LANGUAGE,
        help="Weight for language CE loss",
    )
    args = parser.parse_args()

    target_text = args.targeted
    ITERATIONS = args.iterations
    OUTPUT_DIR = args.output_dir
    epsilon = args.epsilon
    LAMBDA_VISION = args.lambda_vision
    LAMBDA_ALIGNMENT = args.lambda_alignment
    LAMBDA_LANGUAGE = args.lambda_language

    output_dirs = setup_output_dirs(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("WHITE-BOX PGD ATTACK: Qwen2-VL-2B")
    print("=" * 70)

    print()
    print("CONFIGURATION")
    print(f"  Target text      : {target_text}")
    print(f"  Source keyword   : {SOURCE_TEXT}")
    print(f"  Epsilon          : {epsilon:.6f} ({epsilon*255:.1f}/255)")
    print(f"  Alpha            : {ALPHA:.6f}")
    print(f"  Iterations       : {ITERATIONS}")
    print(f"  Momentum         : {MOMENTUM}")
    print(f"  Lambda vision    : {LAMBDA_VISION}")
    print(f"  Lambda alignment : {LAMBDA_ALIGNMENT}")
    print(f"  Lambda language  : {LAMBDA_LANGUAGE}")
    print()

    # --- Load model ---
    print("-" * 70)
    print("LOADING MODEL")
    print("-" * 70)

    print(f"  Loading HuggingFace Qwen2-VL-2B-Instruct (white-box)...")
    model = QwenVLAdapter(model_name="qwen2-vl-2b")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    print(f"  Image size: {model.image_size}")
    print(f"  Vision dim: {model.model.config.vision_config.embed_dim}")
    print(f"  LLM hidden: {model.model.config.text_config.hidden_size}")
    print(f"  Vocab size: {model.model.config.text_config.vocab_size}")

    # --- Collect images ---
    if args.image:
        if not os.path.isabs(args.image):
            args.image = os.path.join(DATASET_DIR, args.image)
        image_paths = [args.image]
    else:
        image_paths = get_dataset_paths()

    print()
    print(f"  Images to evaluate: {len(image_paths)}")
    for p in image_paths:
        print(f"    - {p}")

    # --- Run evaluation ---
    results = []
    start_time = time.time()

    for image_path in image_paths:
        try:
            result = evaluate_single_image(
                image_path=image_path,
                model=model,
                output_dirs=output_dirs,
                target_text=target_text,
                epsilon=epsilon,
                alpha=ALPHA,
                iterations=ITERATIONS,
                lambda_vision=LAMBDA_VISION,
                lambda_alignment=LAMBDA_ALIGNMENT,
                lambda_language=LAMBDA_LANGUAGE,
                momentum=MOMENTUM,
                seed=SEED,
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time

    # --- Final summary ---
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
        f"{'Changed':<10}"
        f"{'SrcRm':<8}"
        f"{'TgtApp':<8}"
        f"{'L-inf':<12}"
        f"{'Time':<8}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['filename']:<20}"
            f"{'YES' if r['desc_changed'] else 'NO':<10}"
            f"{'YES' if r['source_removed'] else 'NO':<8}"
            f"{'YES' if r['target_appeared'] else 'NO':<8}"
            f"{r['linf']:<12.6f}"
            f"{r['attack_time']:<8.1f}"
        )

    # --- Aggregate ---
    changed = sum(1 for r in results if r["desc_changed"])
    src_rm = sum(1 for r in results if r["source_removed"])
    tgt_app = sum(1 for r in results if r["target_appeared"])

    avg_linf = np.mean([r["linf"] for r in results])
    avg_l2 = np.mean([r["l2"] for r in results])
    avg_time = np.mean([r["attack_time"] for r in results])

    print()
    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print()
    print(f"  Images evaluated           : {total}")
    print(f"  Total time                 : {total_time:.1f}s")
    print()
    print(f"  White-box (Qwen2-VL-2B) results:")
    print(f"    Description changed      : {changed}/{total} ({changed/total*100:.1f}%)")
    print(f"    Source keyword removed   : {src_rm}/{total} ({src_rm/total*100:.1f}%)")
    print(f"    Target keyword appeared  : {tgt_app}/{total} ({tgt_app/total*100:.1f}%)")
    print()
    print(f"  Average L-inf              : {avg_linf:.8f}")
    print(f"  Average L2                 : {avg_l2:.8f}")
    print(f"  Average attack time        : {avg_time:.2f}s")

    # --- Descriptions comparison ---
    print()
    print("=" * 70)
    print("DESCRIPTIONS COMPARISON")
    print("=" * 70)

    for r in results:
        print()
        print(f"  {r['filename']}:")
        print(f"    Clean: {r['clean_desc']}")
        print(f"    Adv:   {r['adv_desc']}")

    # --- Save aggregate JSON ---
    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "total_time": total_time,
        "configuration": {
            "attack_mode": "targeted",
            "target_text": target_text,
            "source_text": SOURCE_TEXT,
            "epsilon": epsilon,
            "alpha": ALPHA,
            "iterations": ITERATIONS,
            "momentum": MOMENTUM,
            "seed": SEED,
            "lambda_vision": LAMBDA_VISION,
            "lambda_alignment": LAMBDA_ALIGNMENT,
            "lambda_language": LAMBDA_LANGUAGE,
            "attack_size": model.image_size,
            "white_box_model": "HuggingFace Qwen2-VL-2B-Instruct (bfloat16)",
        },
        "images_evaluated": total,
        "white_box_metrics": {
            "description_changed": changed,
            "description_changed_pct": changed / total * 100,
            "source_removed": src_rm,
            "source_removed_pct": src_rm / total * 100,
            "target_appeared": tgt_app,
            "target_appeared_pct": tgt_app / total * 100,
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

    # --- Save summary text report ---
    report_path = os.path.join(output_dirs["base"], "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Qwen2-VL-2B ADVERSARIAL ATTACK SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Images evaluated: {total}\n\n")

        f.write("Configuration:\n")
        f.write(f"  epsilon={epsilon:.6f}  alpha={ALPHA:.6f}  "
                f"iterations={ITERATIONS}\n")
        f.write(f"  momentum={MOMENTUM}\n")
        f.write(f"  lambda: vision={LAMBDA_VISION} "
                f"alignment={LAMBDA_ALIGNMENT} "
                f"language={LAMBDA_LANGUAGE}\n")
        f.write(f"  attack_size: {model.image_size}x{model.image_size}\n\n")

        f.write("White-box (Qwen2-VL-2B) results:\n")
        f.write(f"  Description changed: {changed}/{total} "
                f"({changed/total*100:.1f}%)\n")
        f.write(f"  Source removed: {src_rm}/{total} "
                f"({src_rm/total*100:.1f}%)\n")
        f.write(f"  Target appeared: {tgt_app}/{total} "
                f"({tgt_app/total*100:.1f}%)\n\n")

        f.write(f"Perturbation:\n")
        f.write(f"  Average L-inf: {avg_linf:.8f}\n")
        f.write(f"  Average L2: {avg_l2:.8f}\n")
        f.write(f"  Average attack time: {avg_time:.2f}s\n\n")

        f.write("Per-image descriptions:\n")
        for r in results:
            f.write(f"\n  {r['filename']}:\n")
            f.write(f"    Clean: {r['clean_desc']}\n")
            f.write(f"    Adv:   {r['adv_desc']}\n")

    print(f"  Saved summary report: {report_path}")

    print()
    print("=" * 70)
    print("Attack evaluation complete.")
    print(f"  All outputs saved to: {output_dirs['base']}")
    print("=" * 70)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
