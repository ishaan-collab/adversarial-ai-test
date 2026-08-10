"""
Phase 1: Transfer test — send existing moondream adversarial images
to Qwen 35B via OpenAI-compatible API (port 11471).
"""

import base64
import io
import json
import time
import os
from PIL import Image
import requests

QWEN_HOST = "http://127.0.0.1:11471"
QWEN_MODEL = "vyas"
ADV_DIR = "outputs/moondream_attack_targeted/adversarial_examples"
CLEAN_IMAGES = ["dog.jpg"] + [f"data/vlm/dog{i:02d}.jpg" for i in range(1, 10)]

DOG_KEYWORDS = [
    "dog", "puppy", "canine", "pup", "hound", "beagle", "retriever",
    "labrador", "husky", "dalmatian", "chihuahua", "pug", "shepherd",
    "terrier", "great dane", "corgi", "spaniel", "collie", "mastiff",
    "bulldog", "boxer", "rottweiler", "doberman", "shiba", "akita",
    "malamute", "schnauzer", "dachshund", "bichon", "sheltie",
]


def query_qwen(pil_img, question="What do you see in this image?", max_tokens=100):
    """Query Qwen 35B via OpenAI-compatible API with an image."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{QWEN_HOST}/v1/chat/completions",
                json=payload, timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 503:
                time.sleep(5)
                continue
            else:
                return f"<error {resp.status_code}: {resp.text[:200]}>"
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                return f"<error: {e}>"
    return "<error: max retries>"


def main():
    print("=" * 80)
    print("PHASE 1: TRANSFER TEST — Moondream adversarial images → Qwen 35B")
    print("=" * 80)
    print(f"  Model:  Qwen3.6-35B-A3B (Q8_0)")
    print(f"  API:    {QWEN_HOST}/v1/chat/completions")
    print(f"  Images: {len(CLEAN_IMAGES)} clean + adversarial pairs")
    print()

    results = []

    for img_path in CLEAN_IMAGES:
        name = os.path.basename(img_path)
        base = os.path.splitext(name)[0]
        adv_path = os.path.join(ADV_DIR, f"adv_{base}.png")

        print(f"\n{'─' * 70}")
        print(f"  Image: {name}")
        print(f"{'─' * 70}")

        # Load clean
        clean_pil = Image.open(img_path).convert("RGB")
        clean_pil = clean_pil.resize((378, 378), Image.LANCZOS)

        print(f"  [CLEAN] Querying Qwen 35B...", end=" ", flush=True)
        t0 = time.time()
        clean_desc = query_qwen(clean_pil)
        clean_time = time.time() - t0
        clean_dog = any(kw in clean_desc.lower() for kw in DOG_KEYWORDS)
        print(f"({clean_time:.1f}s)")
        print(f"  [CLEAN] {clean_desc[:150]}")
        print(f"  [CLEAN] Dog keyword: {'YES' if clean_dog else 'NO'}")

        # Load adversarial
        adv_desc = "<no adversarial image found>"
        adv_dog = True
        if os.path.exists(adv_path):
            adv_pil = Image.open(adv_path).convert("RGB")
            print(f"  [ADV]   Querying Qwen 35B...", end=" ", flush=True)
            t0 = time.time()
            adv_desc = query_qwen(adv_pil)
            adv_time = time.time() - t0
            adv_dog = any(kw in adv_desc.lower() for kw in DOG_KEYWORDS)
            print(f"({adv_time:.1f}s)")
            print(f"  [ADV]   {adv_desc[:150]}")
            print(f"  [ADV]   Dog keyword: {'YES' if adv_dog else 'NO'}")
            print(f"  [ADV]   Transfer success: {'NO (still says dog)' if adv_dog else 'YES (no dog keyword!)'}")
        else:
            print(f"  [ADV]   {adv_path} not found, skipping")

        results.append({
            "image": name,
            "clean_desc": clean_desc,
            "adv_desc": adv_desc,
            "clean_has_dog": clean_dog,
            "adv_has_dog": adv_dog,
            "transfer_success": not adv_dog if os.path.exists(adv_path) else None,
        })

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Image':<14} {'Clean Dog?':<12} {'Adv Dog?':<12} {'Transfer?':<12}")
    print("-" * 50)
    for r in results:
        ts = "YES!" if r["transfer_success"] else "NO"
        print(f"{r['image']:<14} {'YES' if r['clean_has_dog'] else 'NO':<12} "
              f"{'YES' if r['adv_has_dog'] else 'NO':<12} {ts:<12}")

    n_success = sum(1 for r in results if r["transfer_success"])
    n_total = sum(1 for r in results if r["transfer_success"] is not None)
    print(f"\nTransfer success: {n_success}/{n_total} ({100*n_success/max(n_total,1):.0f}%)")

    # Save results
    os.makedirs("outputs/qwen_transfer", exist_ok=True)
    with open("outputs/qwen_transfer/transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: outputs/qwen_transfer/transfer_results.json")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    main()
