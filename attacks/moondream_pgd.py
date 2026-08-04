"""
Multi-level PGD attack against moondream2.

Optimizes a single L-infinity-bounded perturbation using three
complementary loss surfaces:

  L_vision     — feature distance in the vision encoder
  L_alignment  — token distance at the connector (vision projection)
  L_language   — logit-space cross-entropy on the first output token

Both targeted and untargeted modes are supported.

Usage:
    from models.moondream_adapter import MoondreamAdapter
    from attacks.moondream_pgd import moondream_pgd

    model = MoondreamAdapter()
    adv = moondream_pgd(model, image, target_text="A cat")
"""

import random

import random

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MULTI-LEVEL PGD
# ============================================================

def moondream_pgd(
    model,
    image,
    target_text=None,
    epsilon=8 / 255,
    alpha=2 / 255,
    iterations=100,
    lambda_vision=0.5,
    lambda_alignment=0.5,
    lambda_language=1.0,
    random_start=False,
    seed=42,
    return_details=False,
):
    """
    Generate an adversarial image against moondream2.

    The attack minimises a combined loss:

        L = lambda_language * L_lang
            - lambda_vision    * MSE(adv_vision, clean_vision)
            - lambda_alignment * MSE(adv_connector, clean_connector)

    where L_lang is:
        - targeted:   CE(logits, target_token)
        - untargeted: -CE(logits, clean_token)

    so that the PGD step (gradient descent) pushes the output
    toward the target (or away from the clean prediction) while
    simultaneously maximising feature disruption.

    Args:
        model:
            MoondreamAdapter instance (white-box).

        image:
            Tensor [1, 3, H, W] in [0, 1].

        target_text:
            If provided, run a targeted attack pushing the model
            toward this text.  If None, run an untargeted attack.

        epsilon:
            L-infinity perturbation budget.

        alpha:
            PGD step size.

        iterations:
            Number of PGD iterations.

        lambda_vision:
            Weight for the vision-feature disruption loss.

        lambda_alignment:
            Weight for the connector-token disruption loss.

        lambda_language:
            Weight for the language logit loss.

        random_start:
            Whether to initialise randomly inside the epsilon-ball.

        seed:
            Random seed for reproducibility.

        return_details:
            If True, return (adversarial, details_dict).

    Returns:
        adversarial tensor [1, 3, H, W] in [0, 1],
        optionally with a details dict.
    """
    # ========================================================
    # VALIDATION
    # ========================================================

    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            f"Expected [1, 3, H, W], got {tuple(image.shape)}"
        )

    if epsilon <= 0 or alpha <= 0 or iterations <= 0:
        raise ValueError("epsilon, alpha, iterations must be positive")

    # ========================================================
    # SETUP
    # ========================================================

    set_seed(seed)
    device = model.device
    targeted = target_text is not None

    original = (
        image.detach()
        .clone()
        .to(device=device, dtype=torch.float32)
    )
    original = torch.clamp(original, 0.0, 1.0)

    eps_t = torch.tensor(float(epsilon), device=device)
    lower = torch.clamp(original - eps_t, 0.0, 1.0)
    upper = torch.clamp(original + eps_t, 0.0, 1.0)

    # ========================================================
    # CLEAN REFERENCE FEATURES (no gradient)
    # ========================================================

    with torch.no_grad():
        clean_out = model.get_all_features(original)
        clean_vision = clean_out["vision_features"].detach()
        clean_connector = clean_out["connector_tokens"].detach()
        clean_logits = clean_out["logits"].detach()
        clean_token = clean_logits.argmax(dim=-1).item()

    # ========================================================
    # TARGET TOKENS
    # ========================================================

    if targeted:
        target_ids = model.encode_text(target_text)
        target_token = target_ids[0, 0].item()
        target_label = torch.tensor(
            [target_token], device=device, dtype=torch.long
        )
        print(
            f"  Target text: {target_text!r}"
        )
        print(
            f"  Target tokens: {target_ids.shape[1]} "
            f"({model.decode_tokens(target_ids[0].tolist())!r})"
        )
    else:
        target_ids = None
        target_label = torch.tensor(
            [clean_token], device=device, dtype=torch.long
        )

    print(
        f"  Clean token: {clean_token} "
        f"({model.decode_tokens([clean_token])!r})"
    )
    print(
        f"  Mode: {'targeted (multi-token)' if targeted else 'untargeted'}  "
        f"eps={epsilon:.6f}  alpha={alpha:.6f}  "
        f"iters={iterations}"
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    if random_start:
        noise = torch.empty_like(original).uniform_(
            -float(epsilon), float(epsilon)
        )
        adversarial = torch.clamp(original + noise, 0.0, 1.0)
        adversarial = torch.maximum(
            torch.minimum(adversarial, upper), lower
        )
    else:
        adversarial = original.clone()

    # ========================================================
    # PGD LOOP (with momentum + input diversity for transfer)
    # ========================================================

    history = []
    momentum = torch.zeros_like(original)  # MI-FGSM momentum buffer

    for iteration in range(iterations):
        adversarial.requires_grad_(True)

        # ------------------------------------------------
        # Input diversity: random resize + pad (DI-FGSM)
        # Stabilizes perturbation across preprocessing variants
        # ------------------------------------------------

        if targeted:
            # For multi-crop pipeline, random resize helps because
            # the tiling changes with image size
            scale = random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
            if scale != 1.0:
                _, _, H, W = adversarial.shape
                new_h, new_w = int(H * scale), int(W * scale)
                diverse_input = F.interpolate(
                    adversarial, size=(new_h, new_w),
                    mode="bilinear", align_corners=False, antialias=True,
                )
            else:
                diverse_input = adversarial
        else:
            diverse_input = adversarial

        # ------------------------------------------------
        # Forward pass — all three feature levels (single-crop)
        # ------------------------------------------------

        features = model.get_all_features(diverse_input)

        # ------------------------------------------------
        # L_vision: maximise distance from clean features
        # ------------------------------------------------

        l_vision = F.mse_loss(
            features["vision_features"].float(),
            clean_vision.float(),
        )

        # ------------------------------------------------
        # L_alignment: maximise distance from clean tokens
        # ------------------------------------------------

        l_alignment = F.mse_loss(
            features["connector_tokens"].float(),
            clean_connector.float(),
        )

        # ------------------------------------------------
        # L_language: dual-pipeline multi-token CE
        # (both single-crop and multi-crop for transfer)
        # ------------------------------------------------

        if targeted:
            # Multi-crop CE (matches HF describe() and Ollama)
            l_mc, _ = model.get_multicrop_multi_token_loss(
                diverse_input, target_ids
            )
            # Single-crop CE (helps transfer to quantized model)
            l_sc, _ = model.get_multi_token_loss(
                diverse_input, target_ids
            )
            # Combined dual-pipeline loss
            l_language = 0.7 * l_mc + 0.3 * l_sc
            # Also get single-token logits for logging
            logits = features["logits"].float()
        else:
            logits = features["logits"].float()
            l_language = -F.cross_entropy(logits, target_label)

        # ------------------------------------------------
        # Total loss (minimised by PGD)
        # ------------------------------------------------

        loss = (
            lambda_language * l_language
            - lambda_vision * l_vision
            - lambda_alignment * l_alignment
        )

        # ------------------------------------------------
        # Gradient with momentum (MI-FGSM)
        # ------------------------------------------------

        grad = torch.autograd.grad(
            loss,
            diverse_input,
            retain_graph=False,
            create_graph=False,
        )[0]

        # If we used input diversity, resize gradient back to original size
        if grad.shape != adversarial.shape:
            grad = F.interpolate(
                grad, size=adversarial.shape[2:],
                mode="bilinear", align_corners=False, antialias=True,
            )

        # MI-FGSM: accumulate momentum
        grad_l1 = grad.abs().mean()
        if grad_l1 > 0:
            grad = grad / grad_l1
        momentum = 0.9 * momentum + grad  # mu=0.9
        step_grad = momentum.sign()
        # Fallback: if momentum is zero, use raw grad sign
        if step_grad.abs().sum() == 0:
            step_grad = grad.sign()

        # ------------------------------------------------
        # PGD step (gradient descent on loss)
        # ------------------------------------------------

        with torch.no_grad():
            adversarial = adversarial - float(alpha) * step_grad
            adversarial = torch.maximum(
                torch.minimum(adversarial, upper), lower
            )
            adversarial = torch.clamp(adversarial, 0.0, 1.0)

        # ------------------------------------------------
        # Logging
        # ------------------------------------------------

        should_log = (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration == iterations - 1
        )

        if should_log:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                clean_prob = probs[0, clean_token].item()
                if targeted:
                    target_prob = probs[0, target_token].item()
                    # Show multi-token CE progress
                    mt_loss_val = l_language.item()
                else:
                    target_prob = 1.0 - clean_prob
                    mt_loss_val = None

                pred_token = probs.argmax(dim=-1).item()

            entry = {
                "iteration": iteration + 1,
                "loss": loss.item(),
                "l_vision": l_vision.item(),
                "l_alignment": l_alignment.item(),
                "l_language": l_language.item(),
                "clean_prob": clean_prob,
                "target_prob": target_prob,
                "pred_token": pred_token,
                "pred_text": model.decode_tokens([pred_token]),
            }
            history.append(entry)

            if targeted:
                print(
                    f"  Iter {iteration + 1:3d}/{iterations} | "
                    f"loss={loss.item():.4f} "
                    f"L_vis={l_vision.item():.4f} "
                    f"L_align={l_alignment.item():.4f} "
                    f"L_lang={l_language.item():.4f} | "
                    f"clean_p={clean_prob:.4f} "
                    f"target_p={target_prob:.4f} "
                    f"mt_ce={mt_loss_val:.4f} "
                    f"pred={entry['pred_text']!r}"
                )
            else:
                print(
                    f"  Iter {iteration + 1:3d}/{iterations} | "
                    f"loss={loss.item():.4f} "
                    f"L_vis={l_vision.item():.4f} "
                    f"L_align={l_alignment.item():.4f} "
                    f"L_lang={l_language.item():.4f} | "
                    f"clean_p={clean_prob:.4f} "
                    f"target_p={target_prob:.4f} "
                    f"pred={entry['pred_text']!r}"
                )

    # ========================================================
    # FINAL PROJECTION
    # ========================================================

    with torch.no_grad():
        adversarial = torch.maximum(
            torch.minimum(adversarial, upper), lower
        )
        adversarial = torch.clamp(adversarial, 0.0, 1.0)

    adversarial = adversarial.detach()

    # ========================================================
    # RETURN
    # ========================================================

    if not return_details:
        return adversarial

    with torch.no_grad():
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

    details = {
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "iterations": int(iterations),
        "seed": seed,
        "random_start": bool(random_start),
        "targeted": targeted,
        "target_text": target_text,
        "clean_token": clean_token,
        "target_token": target_token if targeted else None,
        "lambda_vision": lambda_vision,
        "lambda_alignment": lambda_alignment,
        "lambda_language": lambda_language,
        "linf": linf,
        "l2": l2,
        "mean_abs": mean_abs,
        "history": history,
    }

    return adversarial, details
