"""Balayage de tau (B3) : fausses alertes et detection, sur les 20 rounds deja produits."""
from pathlib import Path

from federation.ids import round_id
from federation.params_io import load_params
from federation.robust import robust_aggregate

ROOT = Path("offchain")
N = {"clinic_1": 4390, "clinic_2": 2991, "clinic_3": 3651}
CIBLE = "clinic_3"
ROUNDS = 20
TAUS = (1.3, 1.5, 1.82, 2.0, 2.5, 3.0, 4.0)
FORCES = (1, 2, 3, 5, 10)


def charge(r):
    rd = ROOT / round_id(r)
    return {c: load_params(str(rd / f"{c}.npz")) for c in N}


def inversion(local, glob, k):
    return {n: glob[n] - k * (local[n] - glob[n]) for n in ("W", "b")}


def ecartees(u, tau):
    return [d["clinic_id"] for d in robust_aggregate(u, N, tau)["discarded"]]


def main():
    honnetes = [charge(r) for r in range(1, ROUNDS + 1)]
    globaux = [load_params(str(ROOT / round_id(r - 1) / "global_model.npz")) for r in range(1, ROUNDS + 1)]

    print(f"{'tau':>5} | {'fausses alertes':>15} | detection de {CIBLE} (rounds sur {ROUNDS}) par force k")
    print(f"{'':>5} | {'(sur ' + str(ROUNDS * 3) + ')':>15} | " + "  ".join(f"k={k:<3}" for k in FORCES))
    for tau in TAUS:
        faux = sum(len(ecartees(u, tau)) for u in honnetes)
        det = []
        for k in FORCES:
            ok = 0
            for u, g in zip(honnetes, globaux):
                v = dict(u)
                v[CIBLE] = inversion(u[CIBLE], g, k)
                ok += CIBLE in ecartees(v, tau)
            det.append(ok)
        print(f"{tau:>5} | {faux:>15} | " + "  ".join(f"{d:>5}" for d in det))


if __name__ == "__main__":
    main()