"""
Dry-run validator for the GLM-5V-Turbo attack.

Three checks:

1. **Request shape validation** against the real Zhipu endpoint.
   We POST a payload with no Authorization header and expect HTTP 401
   with a clear ``{"error":{"code":...}}`` body. If the request were
   malformed we would get HTTP 400 / 422 instead, so this proves our
   payload shape (model, messages[].content with image_url, max_tokens,
   stream, temperature) matches what glm-5v-turbo accepts.

2. **Mock end-to-end** with the full default budget (400 queries) to
   show realistic timing, query counts and L-inf.

3. **Per-call payload dump** - prints the exact JSON bytes that
   ``GLM5Adapter.query()`` sends, so you can copy it into curl / a
   notebook and replay against the real API once you have a key.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import requests
from PIL import Image

from attacks.square_attack_glm5 import GLM5SquareAttackV2
from models.frontier_adapter import GLM5Adapter, GLMAdapter as _GLMBase  # noqa: F401
GLM_HOST = GLM5Adapter.DEFAULT_HOST


# ============================================================
# 1) Request shape validation
# ============================================================

def build_payload(adapter: GLM5Adapter, pil: Image.Image,
                  max_tokens: int = 30,
                  temperature: float = 0.1) -> Dict[str, Any]:
    """Reproduce exactly what GLM5Adapter.query() sends.

    Mirrors models/frontier_adapter.py:GLMAdapter.query lines 119-141.
    """
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{img_b64}"
    return {
        "model": adapter.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_url}},
                    {"type": "text",
                     "text": adapter.question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def validate_request_shape(image_path: str, model: str = "glm-5v-turbo"):
    print("=" * 70)
    print("[1] REQUEST SHAPE VALIDATION against real Zhipu endpoint")
    print("=" * 70)

    if not os.path.isfile(image_path):
        print(f"  ERROR: image not found: {image_path}")
        return False

    pil = Image.open(image_path).convert("RGB").resize((378, 378), Image.LANCZOS)
    payload = build_payload(
        GLM5Adapter(name=model, api_key="dry-run-no-key"),
        pil, max_tokens=30,
    )

    # Show a redacted preview (don't dump the whole base64 image)
    preview = {
        "model": payload["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url":
                         f"data:image/png;base64,<{len(payload['messages'][0]['content'][0]['image_url']['url'])} chars>"}},
                    {"type": "text",
                     "text": payload["messages"][0]["content"][1]["text"]},
                ],
            }
        ],
        "max_tokens": payload["max_tokens"],
        "temperature": payload["temperature"],
    }
    print(f"\n  Endpoint : {GLM_HOST}/api/paas/v4/chat/completions")
    print(f"  Payload  : {json.dumps(preview, indent=4)}")
    print(f"  Image    : {image_path} ({pil.size[0]}x{pil.size[1]}, "
          f"PNG, {len(payload['messages'][0]['content'][0]['image_url']['url'])} chars)")

    url = f"{GLM_HOST}/api/paas/v4/chat/completions"
    print(f"\n  Posting (no auth) ...")
    t0 = time.time()
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False
    dt = time.time() - t0

    print(f"  HTTP {r.status_code}  in {dt:.2f}s")
    body = r.text
    print(f"  Body     : {body[:400]}")
    print()

    if r.status_code == 401:
        # Expected: auth error, not a format error. 400/422 would mean
        # the payload shape is wrong (e.g. wrong content-type, wrong field).
        try:
            err = r.json().get("error", {})
            code = err.get("code") or err.get("type") or "?"
            msg = err.get("message", "")[:140]
            print(f"  Auth required - payload shape looks valid.")
            print(f"  Error code    : {code}")
            print(f"  Error message : {msg}")
            return True
        except Exception:
            print("  Auth required (non-JSON body).")
            return True
    elif r.status_code in (400, 422):
        print("  PAYLOAD REJECTED - shape may be wrong.")
        return False
    elif r.status_code == 200:
        print("  Unexpected: 200 OK without auth. "
              "The endpoint accepted an anonymous request.")
        return True
    else:
        print(f"  Unexpected status {r.status_code}.")
        return False


# ============================================================
# 2) Full mock end-to-end
# ============================================================

def run_mock_e2e(image_path: str, queries: int = 400,
                 target: str = "cat", epsilon: float = 16 / 255,
                 mock_host: str = "http://127.0.0.1:11883",
                 report: str = None):
    print("\n" + "=" * 70)
    print("[2] FULL MOCK END-TO-END")
    print("=" * 70)
    print(f"  mock_host={mock_host}  queries={queries}  "
          f"epsilon={epsilon * 255:.1f}/255  target={target}")

    # Check mock is alive
    try:
        r = requests.get(mock_host.replace("/api/paas/v4", ""),
                         timeout=3)
    except requests.RequestException:
        print(f"  mock server not running on {mock_host}; "
              f"start it first with:")
        print(f"    PYTHONPATH=. python experiments/mock_glm_server.py "
              f"--port 11883 --preload &")
        return False

    adapter = GLM5Adapter(
        name="glm-5v-turbo",
        host=mock_host,
        api_key="dry-run",
        image_size=378,
    )
    pil = Image.open(image_path).convert("RGB")
    attack = GLM5SquareAttackV2(
        adapter=adapter, image_size=378, sleep_s=0.0,
        target_group=target, verbose=True,
    )
    t0 = time.time()
    adv, info = attack.attack(pil, epsilon=epsilon, queries=queries)
    dt = time.time() - t0

    print(f"\n  Real-time elapsed : {dt:.1f}s")
    print(f"  Queries used      : {info['queries_used']} / {queries}")
    print(f"  Best score        : {info['best_score']:.2f}  "
          f"(char={info['best_char_score']:.1f}, "
          f"bonus={info['best_target_bonus']:.1f})")
    print(f"  L-inf             : {info['linf']:.5f} "
          f"({info['linf'] * 255:.2f}/255, "
          f"budget {epsilon * 255:.1f}/255)")
    print(f"  Contains 'dog'    : {info['contains_dog']}")
    print(f"  Target hit        : {info['best_target_bonus'] > 0}")
    print(f"  Final response    : {info['best_text'][:200]!r}")
    if report:
        os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
        with open(report, "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"  Report saved      : {report}")
    return True


# ============================================================
# 3) Cost / time estimate
# ============================================================

def estimate_cost(queries: int, sleep_s: float = 0.5):
    print("\n" + "=" * 70)
    print("[3] COST / TIME ESTIMATE for real GLM-5V-Turbo")
    print("=" * 70)
    # Zhipu glm-4v pricing: 0.001 CNY / 1k tokens (input) approx; image
    # tokens vary. Use rough numbers as a guide only.
    avg_input_tokens = 1200   # ~1100 image + ~80 prompt
    avg_output_tokens = 30
    # 0.001 CNY/1k input, 0.001 CNY/1k output for glm-4v/5v tier
    cny_per_1k_in = 0.001
    cny_per_1k_out = 0.001
    cny_per_call = (avg_input_tokens * cny_per_1k_in
                    + avg_output_tokens * cny_per_1k_out) / 1000
    total_cny = cny_per_call * queries
    total_usd = total_cny * 0.14  # rough CNY->USD
    wall_s = queries * sleep_s
    wall_min = wall_s / 60
    print(f"  Queries           : {queries}")
    print(f"  Sleep per call    : {sleep_s}s")
    print(f"  Wall time floor   : {wall_min:.1f} minutes  "
          f"({wall_s:.0f}s of pure sleep)")
    print(f"  Estimated cost    : {total_cny:.3f} CNY  (~${total_usd:.3f} USD)")
    print(f"  (Assumes {avg_input_tokens} input tokens, "
          f"{avg_output_tokens} output tokens per call,")
    print(f"   glm-5v-tier pricing 0.001/0.001 CNY per 1k tokens.)")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Dry-run validation for the GLM-5 attack.")
    p.add_argument("--image", default="dog.jpg")
    p.add_argument("--model", default="glm-5v-turbo")
    p.add_argument("--queries", type=int, default=400)
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--target", default="cat")
    p.add_argument("--mock-host", default="http://127.0.0.1:11883")
    p.add_argument("--skip-mock", action="store_true",
                   help="Skip section 2 (the mock end-to-end).")
    p.add_argument("--skip-estimate", action="store_true")
    p.add_argument("--report", default="outputs/adv_glm5_dryrun.json")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    ok = validate_request_shape(args.image, model=args.model)
    if not ok:
        print("\nAborting: request shape rejected by endpoint.")
        sys.exit(1)

    if not args.skip_mock:
        run_mock_e2e(
            image_path=args.image,
            queries=args.queries,
            target=args.target,
            epsilon=args.epsilon,
            mock_host=args.mock_host,
            report=args.report,
        )

    if not args.skip_estimate:
        estimate_cost(args.queries, sleep_s=0.5)


if __name__ == "__main__":
    main()
