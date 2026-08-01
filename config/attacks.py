"""
Centralized configuration for adversarial attacks.

All attack hyperparameters used by the experiment
runner should be defined here.
"""

ATTACK_CONFIG = {

    # --------------------------------------------------
    # FGSM
    # --------------------------------------------------

    "fgsm": {
        "epsilon": 8 / 255,
    },

    # --------------------------------------------------
    # BIM
    # --------------------------------------------------

    "bim": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "iterations": 10,
    },

    # --------------------------------------------------
    # PGD
    # --------------------------------------------------

    "pgd": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "iterations": 10,
        "random_start": True,
    },

    # --------------------------------------------------
    # DeepFool
    # --------------------------------------------------

    "deepfool": {
        "max_iterations": 20,
        "overshoot": 0.02,
    },

    "autoattack": {
        "epsilon": 8 / 255,
        "version": "standard",
    },
}


def get_attack_config(name):
    """
    Return configuration for an attack.
    """

    name = name.lower()

    if name not in ATTACK_CONFIG:

        available = ", ".join(
            ATTACK_CONFIG.keys()
        )

        raise ValueError(
            f"Unknown attack '{name}'. "
            f"Available attacks: {available}"
        )

    return ATTACK_CONFIG[name].copy()
