"""Rejoue robust_aggregate (B3) sur les 20 rounds deja produits, sans et avec attaque."""
from pathlib import Path

from federation.ids import round_id
from federation.params_io import load_params
from federation.robust import robust_aggregate

ROOT = Path("offchain")
N = {"clinic_1": 4390, "clinic_2": 2991, "clinic_3": 3651}
TAU = 1.5
CIBLE = "clinic_3"


def charge(r):
    rd = ROOT / round_id(r)
    return {c: load_params(str(rd / f"{c}.npz")) for c in N}


def inversion(local, glob, k):
    """w_global - k * (w_local - w_global)"""
    return {n: glob[n] - k * (local[n] - glob[n]) for n in ("W", "b")}


def ecartees(u):
    return [d["clinic_id"] for d in robust_aggregate(u, N, TAU)["discarded"]]


def main(n_rounds=20):
    faux = 0
    for r in range(1, n_rounds + 1):
        faux += len(ecartees(charge(r)))
    print(f"Sans attaque : {faux} contribution(s) ecartee(s) a tort sur {n_rounds} rounds (attendu : 0)")

    print(f"\nInversion de signe sur {CIBLE} (tau = {TAU}) :")
    print(" force k | rounds ou la cible est ecartee | autres ecartees a tort")
    for k in (0.5, 1, 2, 3, 5, 10, 50):
        ok, collateral = 0, 0
        for r in range(1, n_rounds + 1):
            u = charge(r)
            glob = load_params(str(ROOT / round_id(r - 1) / "global_model.npz"))
            u[CIBLE] = inversion(u[CIBLE], glob, k)
            e = ecartees(u)
            ok += CIBLE in e
            collateral += len([c for c in e if c != CIBLE])
        print(f" {k:>7} | {ok:>2} / {n_rounds}                          | {collateral}")


if __name__ == "__main__":
    main()