"""
Frontier-LLM transfer test.

For each (clean, adversarial) pair found in ``outputs/``,
query GLM-4.5V (Zhipu), GLM-5V-Turbo (Zhipu), GPT-4o (OpenAI),
and Gemini (Google AI Studio) and record whether each model
still says ``dog``.

Goal
----
Show whether perturbations crafted against small surrogate VLMs
(moondream, LLaVA, Qwen-7B) survive transfer to frontier-scale
vision-language models that humans can still trivially parse.

Usage
-----
    export ZHIPU_API_KEY=...
    export OPENAI_API_KEY=...
    export GOOGLE_API_KEY=...
    PYTHONPATH=. python experiments/frontier_transfer_test.py
    PYTHONPATH=. python experiments/frontier_transfer_test.py --models glm gemini
    PYTHONPATH=. python experiments/frontier_transfer_test.py --models glm5
    PYTHONPATH=. python experiments/frontier_transfer_test.py --adv-dir outputs/moondream_attack_targeted/adversarial_examples
    PYTHONPATH=. python experiments/frontier_transfer_test.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# IMAGE PAIR DISCOVERY
# ============================================================

DEFAULT_ADV_DIRS = [
    "outputs/moondream_attack_targeted/adversarial_examples",
    "outputs/moondream_attack/adversarial_examples",
    "outputs/moondream_targeted/adversarial_examples",
    "outputs/ollama_transfer",
    "outputs/vlm_dataset",
]

CLEAN_DIR = "data/vlm"
CLEAN_FALLBACK = "dog.jpg"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_clean_image(name: str) -> Optional[str]:
    """Locate the clean counterpart for an adversarial image."""
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    for stem_try in (stem, stem.replace("adv_", "")):
        for directory in (CLEAN_DIR, "."):
            for try_ext in (ext, ".jpg", ".png"):
                candidate = os.path.join(directory, stem_try + try_ext)
                if os.path.isfile(candidate):
                    return candidate
    if os.path.isfile(CLEAN_FALLBACK):
        return CLEAN_FALLBACK
    return None


def discover_image_pairs(adv_dirs: List[str]) -> List[Dict[str, str]]:
    """Walk ``adv_dirs`` and pair every ``adv_*`` image with its clean counterpart."""
    seen: set = set()
    pairs: List[Dict[str, str]] = []

    for adv_dir in adv_dirs:
        if not os.path.isdir(adv_dir):
            print(f"  [skip] missing directory: {adv_dir}")
            continue

        for fname in sorted(os.listdir(adv_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            if not fname.lower().startswith("adv_"):
                continue

            adv_path = os.path.abspath(os.path.join(adv_dir, fname))
            if adv_path in seen:
                continue
            seen.add(adv_path)

            clean_path = find_clean_image(fname)
            if clean_path is None:
                print(f"  [skip] no clean counterpart for {adv_path}")
                continue

            clean_path = os.path.abspath(clean_path)
            if clean_path == adv_path:
                continue

            pairs.append({
                "name": os.path.splitext(fname)[0],
                "adv_dir": adv_dir,
                "adv_path": adv_path,
                "clean_path": clean_path,
            })

    return pairs


# ============================================================
# DOG-DETECTION SCORING
# ============================================================

DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]


def contains_dog(text: str) -> Tuple[bool, Optional[str]]:
    """Return (mentions_dog, first_matched_keyword)."""
    if not text:
        return False, None
    text_lower = text.lower()
    for kw in DOG_KEYWORDS:
        if kw in text_lower:
            return True, kw
    return False, None


def load_image(path: str, size: int = 378) -> Image.Image:
    pil = Image.open(path).convert("RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return pil


def linf_distance(pil_a: Image.Image, pil_b: Image.Image) -> float:
    a = np.asarray(pil_a, dtype=np.float32) / 255.0
    b = np.asarray(pil_b, dtype=np.float32) / 255.0
    return float(np.abs(a - b).max())


# ============================================================
# MODEL REGISTRY
# ============================================================

def build_adapter(name: str, host: Optional[str] = None,
                   api_key: Optional[str] = None):
    """Construct a frontier adapter by short name. Raises if key missing."""
    from models.frontier_adapter import (
        GLMAdapter, GLM5Adapter, OpenAIAdapter, GeminiAdapter,
    )

    if name == "glm":
        return GLMAdapter(name="glm-4.5v", host=host, api_key=api_key)
    if name == "glm5":
        return GLM5Adapter(name="glm-5v-turbo", host=host, api_key=api_key)
    if name == "openai" or name == "gpt4o":
        return OpenAIAdapter(name="gpt-4o", host=host, api_key=api_key)
    if name == "gemini":
        return GeminiAdapter(name="gemini-2.5-pro", host=host, api_key=api_key)
    raise ValueError(f"Unknown model '{name}'")


def list_available_models() -> List[str]:
    """Return the subset of {glm, glm5, openai, gemini} whose env var is set.

    ``glm`` and ``glm5`` both read ``ZHIPU_API_KEY`` (same endpoint/auth).

    When ``FRONTIER_MOCK_HOST`` is set, all four are considered available
    regardless of API keys (the mock ignores them).
    """
    available: List[str] = []
    mock = bool(os.environ.get("FRONTIER_MOCK_HOST"))
    if os.environ.get("ZHIPU_API_KEY") or mock:
        available.append("glm")
        available.append("glm5")
    if os.environ.get("OPENAI_API_KEY") or mock:
        available.append("openai")
    if (os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY") or mock):
        available.append("gemini")
    return available


# ============================================================
# SINGLE-IMAGE EVALUATION
# ============================================================

def query_adapter(adapter, pil_image: Image.Image, question: str,
                  max_tokens: int, dry_run: bool = False) -> Tuple[str, float]:
    """Query the adapter, returning (text, elapsed_seconds)."""
    if dry_run:
        return "<dry-run>", 0.0
    t0 = time.time()
    text = adapter.query(pil_image, question=question, max_tokens=max_tokens)
    return text, time.time() - t0


def evaluate_pair(pair: Dict[str, str],
                  adapters: Dict[str, object],
                  question: str,
                  max_tokens: int,
                  dry_run: bool) -> Dict:
    """Evaluate one (clean, adversarial) pair across all adapters."""
    name = pair["name"]
    clean_pil = load_image(pair["clean_path"])
    adv_pil = load_image(pair["adv_path"])
    distance = linf_distance(clean_pil, adv_pil)

    record: Dict = {
        "name": name,
        "adv_path": pair["adv_path"],
        "clean_path": pair["clean_path"],
        "linf": distance,
        "models": {},
    }

    for model_name, adapter in adapters.items():
        print(f"\n    [{model_name}] clean ->", end=" ", flush=True)
        clean_text, clean_t = query_adapter(
            adapter, clean_pil, question, max_tokens, dry_run,
        )
        clean_dog, clean_kw = contains_dog(clean_text)
        print(f"{clean_t:.1f}s | dog={clean_dog} | {clean_text[:80]!r}")

        print(f"    [{model_name}] adv   ->", end=" ", flush=True)
        adv_text, adv_t = query_adapter(
            adapter, adv_pil, question, max_tokens, dry_run,
        )
        adv_dog, adv_kw = contains_dog(adv_text)
        print(f"{adv_t:.1f}s | dog={adv_dog} | {adv_text[:80]!r}")

        transfer_success = clean_dog and not adv_dog
        print(f"    [{model_name}] transfer_success={transfer_success}")

        record["models"][model_name] = {
            "clean_text": clean_text,
            "clean_has_dog": clean_dog,
            "clean_keyword": clean_kw,
            "adv_text": adv_text,
            "adv_has_dog": adv_dog,
            "adv_keyword": adv_kw,
            "transfer_success": transfer_success,
            "clean_time_s": clean_t,
            "adv_time_s": adv_t,
        }

    return record


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transfer-test existing adversarial images "
                    "against frontier VLMs (GLM-4.5V, GLM-5V-Turbo, "
                    "GPT-4o, Gemini).",
    )
    parser.add_argument(
        "--adv-dir", action="append", default=None,
        help="Directory containing adv_*.png adversarial images. "
             "Repeat to add multiple. Defaults to all known locations.",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["glm", "glm5", "openai", "gemini"], default=None,
        help="Frontier models to test (default: all with API keys set).",
    )
    parser.add_argument(
        "--question", default="What do you see in this image?",
        help="Prompt sent with every image.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150,
        help="Max tokens per model response.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of image pairs evaluated.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/frontier_transfer",
        help="Where to save JSON + Markdown reports.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the pipeline without calling any API (useful for testing).",
    )
    parser.add_argument(
        "--mock-host", default=os.environ.get("FRONTIER_MOCK_HOST"),
        help="Redirect all frontier API calls to this host "
             "(e.g. http://127.0.0.1:11880 for the local mock server). "
             "Also reads FRONTIER_MOCK_HOST env var.",
    )
    args = parser.parse_args()

    adv_dirs = args.adv_dir or DEFAULT_ADV_DIRS

    print("=" * 78)
    print("FRONTIER-LLM TRANSFER TEST")
    print("=" * 78)
    print(f"  Question : {args.question!r}")
    print(f"  Max tok  : {args.max_tokens}")
    print(f"  Output   : {args.output_dir}")
    print(f"  Dry run  : {args.dry_run}")
    if args.mock_host:
        print(f"  Mock host: {args.mock_host}  (using local mock server)")

    available = list_available_models()
    chosen = args.models or available
    if args.dry_run:
        chosen = chosen or ["glm", "glm5", "openai", "gemini"]
    missing = [m for m in chosen
               if m not in available and not args.dry_run and not args.mock_host]
    if missing:
        print(f"  WARNING : no API key for {missing}; those will be skipped.")
    chosen = [m for m in chosen
              if m in available or args.dry_run or args.mock_host]

    if not chosen:
        print("No frontier models selected. Set ZHIPU_API_KEY / "
              "OPENAI_API_KEY / GOOGLE_API_KEY and retry.")
        sys.exit(1)

    print(f"  Models   : {', '.join(chosen)}")

    print()
    print("Discovering image pairs...")
    pairs = discover_image_pairs(adv_dirs)
    print(f"  Found {len(pairs)} adversarial image(s) with clean counterparts.")
    for p in pairs:
        print(f"    - {p['name']}  ({p['adv_dir']})")
    if not pairs:
        print("No adversarial images found. Run a local attack first.")
        sys.exit(1)

    if args.limit is not None:
        pairs = pairs[: args.limit]
        print(f"  Limited to first {len(pairs)} pair(s).")

    print()
    print("Initialising adapters...")
    adapters = {}
    for model_name in chosen:
        if args.dry_run:
            adapters[model_name] = None
            print(f"  [dry-run] skipping {model_name}")
            continue
        try:
            adapters[model_name] = build_adapter(
                model_name,
                host=args.mock_host,
                api_key="mock-key" if args.mock_host else None,
            )
            print(f"  [ok] {model_name}: {adapters[model_name].model_name} "
                  f"-> {adapters[model_name].host}")
        except Exception as exc:
            print(f"  [fail] {model_name}: {exc}")

    if not adapters:
        print("No adapters initialised. Aborting.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[Dict] = []
    started = time.time()

    for pair in pairs:
        print()
        print("-" * 78)
        print(f"  PAIR: {pair['name']}")
        print(f"    adv   : {pair['adv_path']}")
        print(f"    clean : {pair['clean_path']}")
        print("-" * 78)
        try:
            record = evaluate_pair(
                pair, adapters, args.question, args.max_tokens,
                dry_run=args.dry_run,
            )
            results.append(record)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"name": pair["name"], "error": str(exc)})

        partial = os.path.join(args.output_dir, "results.json")
        with open(partial, "w") as fh:
            json.dump(results, fh, indent=2)

    elapsed = time.time() - started
    summary = build_summary(results, chosen, elapsed)
    print()
    print(summary)

    md_path = os.path.join(args.output_dir, "report.md")
    with open(md_path, "w") as fh:
        fh.write(summary)
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump({
            "question": args.question,
            "max_tokens": args.max_tokens,
            "dry_run": args.dry_run,
            "models": chosen,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, fh, indent=2)

    print()
    print(f"Report : {md_path}")
    print(f"Data   : {json_path}")


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results: List[Dict],
                  models: List[str],
                  elapsed: float) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("FRONTIER-LLM TRANSFER TEST - SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Images tested  : {len(results)}")
    lines.append(f"Models         : {', '.join(models)}")
    lines.append(f"Wall-clock (s) : {elapsed:.1f}")
    lines.append("")

    header = f"{'Image':<28}" + "".join(f"{m:>14}" for m in models)
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if "error" in r:
            lines.append(f"{r['name']:<28} ERROR: {r['error']}")
            continue
        row = f"{r['name']:<28}"
        for m in models:
            entry = r["models"].get(m, {})
            if "transfer_success" not in entry:
                row += f"{'n/a':>14}"
            elif entry["transfer_success"]:
                row += f"{'TRANSFER!':>14}"
            else:
                row += f"{'no transfer':>14}"
        lines.append(row)

    lines.append("")
    lines.append("Per-model transfer rate (clean->no-dog):")
    for m in models:
        n_total = sum(
            1 for r in results
            if r.get("models", {}).get(m, {}).get("transfer_success") is not None
        )
        n_success = sum(
            1 for r in results
            if r.get("models", {}).get(m, {}).get("transfer_success") is True
        )
        n_clean = sum(
            1 for r in results
            if r.get("models", {}).get(m, {}).get("clean_has_dog") is True
        )
        n_adv_dog = sum(
            1 for r in results
            if r.get("models", {}).get(m, {}).get("adv_has_dog") is True
        )
        if n_total:
            rate = 100.0 * n_success / n_total
            lines.append(
                f"  {m:<10} transfer={n_success}/{n_total} ({rate:5.1f}%)  "
                f"clean_dog={n_clean}  adv_dog={n_adv_dog}"
            )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  transfer success  -> frontier model now fails to say 'dog' on the")
    lines.append("                       adversarial image but would have said 'dog' on")
    lines.append("                       the clean original (humans still see a dog).")
    lines.append("  no transfer       -> frontier model still says 'dog'; perturbation")
    lines.append("                       did not survive the transfer.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
