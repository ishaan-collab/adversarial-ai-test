"""
Test script for QwenVLAdapter.

Verifies all adapter methods work correctly:
  1. Model loads
  2. describe() generates text
  3. tokenize_prompt() produces correct number of image pad tokens
  4. get_vision_features() returns [1024, 1280]
  5. get_connector_tokens() returns [256, 3584]
  6. compute_loss() produces scalar loss with gradient
  7. get_llm_logits() returns logits
  8. Full PGD attack on one image
"""

import sys
import os
import time
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen_vl_adapter import QwenVLAdapter
from attacks.vlm_pgd_universal import MultiLevelPGDAttack


def main():
    print("=" * 60)
    print("QwenVLAdapter Test")
    print("=" * 60)

    # --- 1. Load model ---
    print("\n[1] Loading Qwen2-VL-2B...")
    t0 = time.time()
    adapter = QwenVLAdapter(model_name="qwen2-vl-2b", device="cuda")
    print(f"    Loaded in {time.time()-t0:.1f}s")
    print(f"    image_size={adapter.image_size}")
    print(f"    patch_size={adapter.patch_size}")
    print(f"    grid_h={adapter.grid_h}, grid_w={adapter.grid_w}")
    print(f"    num_patches={adapter.num_patches}")
    print(f"    num_image_tokens={adapter.num_image_tokens}")
    print(f"    image_token_id={adapter.image_token_id}")
    print(f"    dtype={adapter.dtype}")

    # --- 2. Test describe() ---
    print("\n[2] Testing describe()...")
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vlm", "dog03.jpg")
    pil = Image.open(img_path).convert("RGB")
    pil_resized = pil.resize((adapter.image_size, adapter.image_size), Image.LANCZOS)
    desc = adapter.describe(pil_resized, max_tokens=50)
    print(f"    Image: {img_path}")
    print(f"    Description: {desc}")

    # --- 3. Test tokenize_prompt() ---
    print("\n[3] Testing tokenize_prompt()...")
    prompt_ids = adapter.tokenize_prompt("What do you see in this image?")
    print(f"    prompt_ids shape: {prompt_ids.shape}")
    n_image_tokens = (prompt_ids == adapter.image_token_id).sum().item()
    print(f"    Image pad tokens in prompt: {n_image_tokens}")
    assert n_image_tokens == adapter.num_image_tokens, \
        f"Expected {adapter.num_image_tokens} image tokens, got {n_image_tokens}"
    print(f"    OK: {n_image_tokens} == {adapter.num_image_tokens}")

    # --- 4. Test get_vision_features() ---
    print("\n[4] Testing get_vision_features()...")
    img_tensor = adapter.pil_to_tensor(pil_resized)
    print(f"    Input tensor shape: {img_tensor.shape}")
    vision_feats = adapter.get_vision_features(img_tensor)
    print(f"    Vision features shape: {vision_feats.shape}")
    expected_vision = (adapter.num_patches, adapter.model.config.vision_config.embed_dim)
    print(f"    Expected: {expected_vision}")
    assert vision_feats.shape == expected_vision, \
        f"Expected {expected_vision}, got {vision_feats.shape}"
    print(f"    OK")

    # --- 5. Test get_connector_tokens() ---
    print("\n[5] Testing get_connector_tokens()...")
    connector = adapter.get_connector_tokens(img_tensor)
    print(f"    Connector tokens shape: {connector.shape}")
    expected_conn = (adapter.num_image_tokens, adapter.model.config.text_config.hidden_size)
    print(f"    Expected: {expected_conn}")
    assert connector.shape == expected_conn, \
        f"Expected {expected_conn}, got {connector.shape}"
    print(f"    OK")

    # --- 6. Test compute_loss() with gradient ---
    print("\n[6] Testing compute_loss() with gradient...")
    target_text = "A cat sitting on a couch"
    target_ids = adapter.tokenize(target_text)
    print(f"    Target: '{target_text}'")
    print(f"    Target tokens: {target_ids.shape}")

    img_grad = img_tensor.clone().detach().requires_grad_(True)
    loss = adapter.compute_loss(img_grad, target_ids, prompt_ids)
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Loss requires_grad: {loss.requires_grad}")

    grad = torch.autograd.grad(loss, img_grad)[0]
    print(f"    Gradient shape: {grad.shape}")
    print(f"    Gradient norm: {grad.norm().item():.4f}")
    print(f"    Gradient has NaN: {torch.isnan(grad).any().item()}")
    assert not torch.isnan(grad).any(), "Gradient contains NaN!"
    print(f"    OK")

    # --- 7. Test get_llm_logits() ---
    print("\n[7] Testing get_llm_logits()...")
    logits = adapter.get_llm_logits(img_tensor, prompt_ids)
    print(f"    Logits shape: {logits.shape}")
    expected_logits = (1, prompt_ids.shape[1], adapter.model.config.text_config.vocab_size)
    print(f"    Expected: {expected_logits}")
    assert logits.shape == expected_logits, \
        f"Expected {expected_logits}, got {logits.shape}"
    print(f"    OK")

    # --- 8. Quick PGD attack (5 iters) ---
    print("\n[8] Testing MultiLevelPGDAttack (5 iters)...")
    attack = MultiLevelPGDAttack(
        adapter=adapter,
        epsilon=8/255,
        alpha=2/255,
        iterations=5,
        momentum=0.9,
        lambda_vision=1.0,
        lambda_alignment=1.0,
        lambda_language=5.0,
        random_start=True,
        seed=42,
    )
    t0 = time.time()
    adv_tensor, details = attack.attack(
        img_tensor, target_text,
        return_details=True, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n    Attack completed in {elapsed:.1f}s")
    print(f"    L-inf: {details['linf']:.6f} (budget: {details['epsilon']:.6f})")
    print(f"    L2: {details['l2']:.4f}")
    print(f"    mean_abs: {details['mean_abs']:.6f}")

    # Check budget
    perturbation = adv_tensor - img_tensor
    linf = perturbation.abs().max().item()
    assert linf <= 8/255 + 1e-5, f"Perturbation exceeds budget: {linf} > {8/255}"
    print(f"    Budget check: PASS (linf={linf:.6f} <= {8/255:.6f})")

    # --- 9. Verify adversarial description ---
    print("\n[9] Testing adversarial description...")
    adv_pil = adapter.tensor_to_pil(adv_tensor)
    adv_desc = adapter.describe(adv_pil, max_tokens=50)
    print(f"    Clean desc: {desc}")
    print(f"    Adv desc:   {adv_desc}")
    print(f"    Changed: {desc != adv_desc}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
