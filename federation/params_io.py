"""Lecture / ecriture des parametres du modele dans un fichier .npz.

ATTENTION : ne jamais comparer ni hacher les OCTETS d'un .npz. Le format zip y
inscrit une date : deux fichiers aux valeurs identiques peuvent differer.
On compare les VALEURS (c'est aussi ce que fera l'empreinte de A2).
"""
from pathlib import Path

import numpy as np


def save_params(path, params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=params["W"], b=params["b"])


def load_params(path):
    with np.load(path) as f:
        return {"W": f["W"].astype(np.float64, copy=True),
                "b": f["b"].astype(np.float64, copy=True)}