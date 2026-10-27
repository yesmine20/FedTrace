"""Modele softmax (regression logistique multinomiale) de FedTrace.

Parametres : un dict {"W": (61, 6), "b": (6,)} en float64.
Aucune fonction ici ne modifie ses arguments : on renvoie toujours de
nouveaux objets (utile pour les attaques, qui ont besoin de w_global ET w_local).
"""
import numpy as np

N_FEATURES = 61
N_CLASSES = 6


# ---------------------------------------------------------------- parametres
def init_params():
    """Initialisation a zero : le probleme est convexe, donc aucun aleatoire
    a gerer et les 3 cliniques partent du meme point."""
    return {"W": np.zeros((N_FEATURES, N_CLASSES), dtype=np.float64),
            "b": np.zeros(N_CLASSES, dtype=np.float64)}


def copy_params(params):
    """Copie profonde (evite qu'un client ecrase le modele global)."""
    return {k: v.copy() for k, v in params.items()}


# ------------------------------------------------------------------ softmax
def softmax(z):
    """Softmax ligne par ligne. On soustrait le max de chaque ligne :
    le resultat est mathematiquement identique mais exp() ne deborde plus."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    """log(softmax(z)) calcule directement, sans passer par une proba qui
    pourrait valoir 0 (log(0) = -inf)."""
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


# --------------------------------------------------------------- prediction
def logits(params, x):
    return x @ params["W"] + params["b"]


def predict_proba(params, x):
    return softmax(logits(params, x))


def predict(params, x):
    """Classe la plus probable. argmax(logits) suffit : softmax est croissant."""
    return np.argmax(logits(params, x), axis=1)


# -------------------------------------------------------------------- perte
def loss(params, x, y, l2=0.0):
    """Entropie croisee moyenne + (l2/2)*||W||^2 (le biais n'est pas regularise)."""
    n = x.shape[0]
    logp = log_softmax(logits(params, x))
    ce = -np.mean(logp[np.arange(n), y])
    return float(ce + 0.5 * l2 * np.sum(params["W"] ** 2))


# ---------------------------------------------------------------- gradient
def gradient(params, x, y, l2=0.0):
    """Gradient de loss() par rapport a W et b.

    dZ = (P - Y) / n   avec Y one-hot ; dW = X^T dZ + l2*W ; db = somme de dZ.
    """
    n = x.shape[0]
    dz = predict_proba(params, x)
    dz[np.arange(n), y] -= 1.0          # P - Y sans construire Y explicitement
    dz /= n
    return {"W": x.T @ dz + l2 * params["W"],
            "b": dz.sum(axis=0)}


# --------------------------------------------------------------- entrainement
def train_local(params, x, y, lr=0.1, epochs=5, l2=0.0):
    """Descente de gradient sur TOUT le jeu de donnees (full-batch).

    Deterministe : pas de melange ni de tirage aleatoire. Renvoie de nouveaux
    parametres ; `params` n'est pas modifie.
    """
    p = copy_params(params)
    for _ in range(epochs):
        g = gradient(p, x, y, l2)
        p["W"] -= lr * g["W"]
        p["b"] -= lr * g["b"]
    return p