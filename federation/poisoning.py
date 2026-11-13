"""Attaques simulées contre les mises à jour d'un client clinique (A1 suite).

Les attaques sont appliquées dans le processus client avant l'écriture du fichier
de poids et avant le crochet d'ancrage du serveur. Le fichier reçu représente
donc bien la contribution envoyée par le client, attaque comprise.
"""
import re

import numpy as np


ATTACK_TYPES = {"none", "sign_inversion", "gaussian_noise", "label_flip"}


def validate_attack(attack: dict | None) -> dict:
    """Normalise et valide la configuration d'une attaque."""
    attack = dict(attack or {})
    kind = attack.get("type", "none")
    clinic_id = attack.get("clinic_id", "clinic_3")
    start_round = attack.get("start_round", 1)
    strength = attack.get("strength", 10.0)

    if kind not in ATTACK_TYPES:
        raise ValueError(f"Attaque inconnue : {kind}. Valeurs : {sorted(ATTACK_TYPES)}")
    if not isinstance(clinic_id, str) or not re.fullmatch(r"clinic_[1-9][0-9]*", clinic_id):
        raise ValueError("clinic_id doit avoir la forme clinic_1, clinic_2, ...")
    if not isinstance(start_round, int) or start_round < 1:
        raise ValueError("start_round doit être un entier supérieur ou égal à 1.")
    if not isinstance(strength, (int, float)) or not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength doit être un nombre fini strictement positif.")
    if kind == "label_flip" and strength > 1:
        raise ValueError("Pour label_flip, strength est une proportion comprise entre 0 et 1.")

    return {
        "type": kind,
        "clinic_id": clinic_id,
        "start_round": start_round,
        "strength": float(strength),
    }


def attack_is_active(attack: dict, clinic_id: str, round_idx: int) -> bool:
    """Vrai si cette clinique est ciblée à partir du round demandé."""
    return (
        attack["type"] != "none"
        and clinic_id == attack["clinic_id"]
        and round_idx >= attack["start_round"]
    )


def flip_labels(y: np.ndarray, fraction: float, n_classes: int, seed: int) -> np.ndarray:
    """Décale d'une classe les étiquettes d'une fraction déterministe des exemples.

    Pour les classes indexées 0..n-1, l'étiquette choisie c devient (c + 1) % n.
    Le même seed sélectionne les mêmes lignes à chaque round.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction doit être dans l'intervalle (0, 1].")
    if n_classes < 2:
        raise ValueError("Il faut au moins deux classes pour inverser des étiquettes.")
    rng = np.random.default_rng(seed)
    mask = rng.random(len(y)) < fraction
    # Garantit au moins une étiquette modifiée pour une fraction positive.
    if not mask.any() and len(y):
        mask[rng.integers(0, len(y))] = True
    poisoned = np.asarray(y, dtype=np.int64).copy()
    poisoned[mask] = (poisoned[mask] + 1) % n_classes
    return poisoned


def sign_invert(global_params: dict, local_params: dict, strength: float) -> dict:
    """Inverse/amplifie le déplacement local : global - k * (local - global)."""
    return {
        name: global_params[name] - strength * (local_params[name] - global_params[name])
        for name in ("W", "b")
    }


def add_gaussian_noise(
    global_params: dict,
    local_params: dict,
    strength: float,
    seed: int,
) -> dict:
    """Ajoute un bruit à l'update, avec sigma = strength × RMS de l'update local."""
    delta = {name: local_params[name] - global_params[name] for name in ("W", "b")}
    values = np.concatenate([delta["W"].ravel(), delta["b"].ravel()])
    rms = float(np.sqrt(np.mean(values ** 2)))
    sigma = strength * rms
    rng = np.random.default_rng(seed)
    return {
        name: local_params[name] + rng.normal(0.0, sigma, size=local_params[name].shape)
        for name in ("W", "b")
    }


def apply_parameter_attack(
    attack_type: str,
    global_params: dict,
    local_params: dict,
    strength: float,
    seed: int,
) -> dict:
    """Applique une attaque sur les poids locaux après l'entraînement."""
    if attack_type == "sign_inversion":
        return sign_invert(global_params, local_params, strength)
    if attack_type == "gaussian_noise":
        return add_gaussian_noise(global_params, local_params, strength, seed)
    if attack_type == "none" or attack_type == "label_flip":
        return local_params
    raise ValueError(f"Attaque sur poids non prise en charge : {attack_type}")
