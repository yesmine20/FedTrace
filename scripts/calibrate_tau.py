"""Calibration de tau (B3) sur une federation SANS attaque deja lancee."""
from pathlib import Path

import numpy as np

from federation.ids import round_id
from federation.params_io import load_params

ROOT = Path("offchain")
K_ATTAQUE = 10  # inversion de signe de reference


def flat(p):
    return np.concatenate([p["W"].ravel(), p["b"]])


def ratios(vecs):
    """d_k / mediane(d) : meme calcul que robust_aggregate."""
    mat = np.stack(vecs)
    med = np.median(mat, axis=0)
    d = np.linalg.norm(mat - med, axis=1)
    return d / np.median(d)


def main(n_rounds=20):
    honnete, attaque = [], []
    for r in range(1, n_rounds + 1):
        rd = ROOT / round_id(r)
        files = sorted(rd.glob("clinic_*.npz"))
        if len(files) < 3:
            raise SystemExit(f"round {r} : {len(files)} fichiers clinic_*.npz dans {rd}")
        vecs = [flat(load_params(str(f))) for f in files]
        honnete.append(ratios(vecs).max())
        # attaque simulee sur la derniere clinique : w_global - k (w_local - w_global)
        w_global = flat(load_params(str(ROOT / round_id(r - 1) / "global_model.npz")))
        vecs[-1] = w_global - K_ATTAQUE * (vecs[-1] - w_global)
        attaque.append(ratios(vecs).max())
        print(f"{round_id(r)}  honnete: {honnete[-1]:6.2f}   inversion x{K_ATTAQUE}: {attaque[-1]:6.2f}")

    h, a = max(honnete), min(attaque)
    print(f"\nPlus grand rapport sans attaque : {h:.2f}")
    print(f"Plus petit rapport avec attaque : {a:.2f}")
    if h < a:
        print(f"tau doit etre entre {h:.2f} et {a:.2f} ; suggestion (marge x1.5 sur le cas honnete) : {1.5 * h:.2f}")
    else:
        print("Les deux plages se chevauchent : aucun tau ne separe proprement les cas.")


if __name__ == "__main__":
    main()