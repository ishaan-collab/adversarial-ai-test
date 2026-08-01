from evaluation.predict import predict
from evaluation.metrics import (
    compute_perturbation_metrics,
)


def evaluate_attack(
    model,
    weights,
    original_image,
    adversarial_image,
    preprocess=None,
):
    """
    Evaluate an adversarial example against
    a specific target model.

    Images must be raw tensors in [0, 1].

    Preprocessing is applied only when
    performing model inference.

    Perturbation metrics are measured
    in raw pixel space.
    """

    # ============================================================
    # Clean prediction
    # ============================================================

    clean = predict(
        model=model,
        image=original_image,
        weights=weights,
        preprocess=preprocess,
    )

    # ============================================================
    # Adversarial prediction
    # ============================================================

    adversarial = predict(
        model=model,
        image=adversarial_image,
        weights=weights,
        preprocess=preprocess,
    )

    # ============================================================
    # Attack success
    # ============================================================

    prediction_changed = (
        clean["class_id"]
        != adversarial["class_id"]
    )

    # ============================================================
    # Confidence change
    # ============================================================

    confidence_change = (
        adversarial["confidence"]
        - clean["confidence"]
    )

    # ============================================================
    # Perturbation metrics
    # ============================================================

    perturbation = (
        compute_perturbation_metrics(
            original_image,
            adversarial_image,
        )
    )

    # ============================================================
    # Return unified result
    # ============================================================

    return {
        "clean": clean,
        "adversarial": adversarial,

        "prediction_changed":
            prediction_changed,

        "confidence_change":
            confidence_change,

        **perturbation,
    }


def print_evaluation(
    results,
    attack_name,
):
    """
    Pretty-print standardized attack results.
    """

    print()
    print("=" * 60)
    print(
        f"{attack_name.upper()} ATTACK RESULTS"
    )
    print("=" * 60)

    # ------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------

    print()

    print(
        "Clean prediction:"
    )

    print(
        f"  {results['clean']['category']}"
    )

    print(
        "Clean confidence:",
        f"{results['clean']['confidence'] * 100:.2f}%",
    )

    # ------------------------------------------------------------
    # Adversarial
    # ------------------------------------------------------------

    print()

    print(
        "Adversarial prediction:"
    )

    print(
        f"  {results['adversarial']['category']}"
    )

    print(
        "Adversarial confidence:",
        f"{results['adversarial']['confidence'] * 100:.2f}%",
    )

    # ------------------------------------------------------------
    # Success
    # ------------------------------------------------------------

    print()

    print(
        "Prediction changed:",
        "YES"
        if results["prediction_changed"]
        else "NO",
    )

    print(
        "Confidence change:",
        f"{results['confidence_change'] * 100:+.2f}"
        " percentage points",
    )

    # ------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------

    print()

    print(
        "Perturbation metrics:"
    )

    print(
        "  L∞:",
        f"{results['linf']:.8f}",
    )

    print(
        "  L2:",
        f"{results['l2']:.8f}",
    )

    print(
        "  Mean |perturbation|:",
        f"{results['mean_perturbation']:.8f}",
    )

    print("=" * 60)