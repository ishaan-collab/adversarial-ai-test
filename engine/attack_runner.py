from engine.attack_registry import get_attack


def run_attack(
    attack_name,
    model,
    image,
    label,
    preprocess=None,
    **kwargs,
):
    """
    Unified attack execution interface.

    The attack-specific parameters are passed
    through kwargs.
    """

    attack = get_attack(
        attack_name
    )

    return attack(
        model=model,
        image=image,
        label=label,
        preprocess=preprocess,
        **kwargs,
    )