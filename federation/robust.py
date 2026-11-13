"""Agregation robuste (B3) : mediane + distance a la mediane + ecart des aberrantes.

Une mise a jour = un dict de parametres {"W": (61, 6), "b": (6,)}, comme dans federation/model.py.
"""
import numpy as np


def flatten(params):
    """Met W et b bout a bout dans un seul vecteur."""
    return np.concatenate([np.asarray(params["W"], dtype=np.float64).ravel(),
                           np.asarray(params["b"], dtype=np.float64).ravel()])


def unflatten(vec, like):
    """Inverse de flatten : redonne un dict {"W", "b"} de meme forme que `like`."""
    nw = like["W"].size
    return {"W": vec[:nw].reshape(like["W"].shape).copy(), "b": vec[nw:].reshape(like["b"].shape).copy()}


def robust_aggregate(updates, n_samples, tau):
    """Agrege les mises a jour des cliniques en ecartant celles qui sont trop loin de la mediane.

    updates   : {clinic_id: params dict}
    n_samples : {clinic_id: nombre de patients}
    tau       : seuil ; on ecarte la clinique k si d_k > tau * mediane des distances
    Retourne  : {"params", "retained", "discarded", "distances"}
    """
    ids = sorted(updates)                               # ordre fixe : le resultat ne depend pas de l'ordre d'arrivee
    like = updates[ids[0]]
    vecs = {k: flatten(updates[k]) for k in ids}

    discarded = []
    valid = []
    for k in ids:                                       # NaN ou infini : ecarte sans planter
        if np.isfinite(vecs[k]).all():
            valid.append(k)
        else:
            discarded.append({"clinic_id": k, "reason": "non_finite"})
    if not valid:
        raise ValueError("aucune mise a jour exploitable")

    stack = np.stack([vecs[k] for k in valid])
    median = np.median(stack, axis=0)                   # mediane coordonnee par coordonnee
    dist = {k: float(np.linalg.norm(vecs[k] - median)) for k in valid}
    ref = float(np.median(list(dist.values())))         # distance "habituelle"

    retained = []
    for k in valid:
        if ref > 0 and dist[k] > tau * ref:
            discarded.append({"clinic_id": k, "reason": f"distance {dist[k]:.3g} > {tau} x {ref:.3g}"})
        else:
            retained.append(k)

    if not retained:
        return {"params": None, "retained": [],
                "discarded": sorted(discarded, key=lambda d: d["clinic_id"]), "distances": dist}

    w = np.array([n_samples[k] for k in retained], dtype=np.float64)
    avg = (w[:, None] * np.stack([vecs[k] for k in retained])).sum(axis=0) / w.sum()
    return {"params": unflatten(avg, like), "retained": retained,
            "discarded": sorted(discarded, key=lambda d: d["clinic_id"]), "distances": dist}