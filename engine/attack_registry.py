from attacks.fgsm import fgsm_attack
from attacks.bim import bim_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.autoattack import autoattack_attack

ATTACK_REGISTRY = {
    "fgsm": fgsm_attack,
    "bim": bim_attack,
    "pgd": pgd_attack,
    "deepfool": deepfool_attack,
    "autoattack": autoattack_attack
}


def get_attack(name):
    """
    Retrieve an attack implementation
    from the centralized attack registry.
    """

    name = name.lower()

    if name not in ATTACK_REGISTRY:

        available = ", ".join(
            ATTACK_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown attack '{name}'. "
            f"Available attacks: {available}"
        )

    return ATTACK_REGISTRY[name]