"""Fin de B1 : compare 4 agregations sur 20 rounds HONNETES (sans attaque)."""
import csv
from pathlib import Path

import numpy as np

from federation.metrics import class_names, evaluate, feature_columns, load_split, to_xy
from federation.model import init_params, predict, train_local
from federation.robust import robust_aggregate

LR, EPOCHS, L2, ROUNDS, TAU = 0.1, 5, 0.1, 20, 1.5  # memes reglages que le serveur de A


def prepare():
    clinics, test_df = load_split(".")
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    raw = {f"clinic_{k}": to_xy(d, classes, cols) for k, d in clinics.items()}
    x_all = np.concatenate([x for x, _ in raw.values()])
    mu = x_all.mean(axis=0)
    sd = np.sqrt(np.maximum(x_all.var(axis=0), 1e-12))
    sd[sd == 0] = 1.0
    data = {c: ((x - mu) / sd, y) for c, (x, y) in raw.items()}
    xt, yt = to_xy(test_df, classes, cols)
    return data, ((xt - mu) / sd, yt), classes


def stack(u, key):
    return np.stack([u[c][key] for c in sorted(u)])


def fedavg(u, n):
    w = np.array([n[c] for c in sorted(u)], dtype=np.float64)
    w /= w.sum()
    return {k: np.tensordot(w, stack(u, k), axes=1) for k in ("W", "b")}


def mediane(u, n):
    return {k: np.median(stack(u, k), axis=0) for k in ("W", "b")}


def tronquee(u, n):
    out = {}
    for k in ("W", "b"):
        s = np.sort(stack(u, k), axis=0)
        out[k] = s[1:-1].mean(axis=0)  # retire 1 valeur de chaque cote
    return out


def robuste(u, n):
    return robust_aggregate(u, n, TAU)["params"]


STRATEGIES = {"fedavg": fedavg, "mediane": mediane, "tronquee": tronquee, "robuste": robuste}


def run(agg, data, test, classes):
    xt, yt = test
    n = {c: len(y) for c, (_, y) in data.items()}
    g = init_params()
    hist = []
    for _ in range(ROUNDS):
        u = {c: train_local(g, x, y, lr=LR, epochs=EPOCHS, l2=L2) for c, (x, y) in data.items()}
        g = agg(u, n)
        hist.append(evaluate(yt, predict(g, xt), classes))
    return hist


def main():
    data, test, classes = prepare()
    results = {name: run(f, data, test, classes) for name, f in STRATEGIES.items()}

    for metric in ("accuracy_6", "accuracy_normal"):
        print(f"\n{metric}")
        print(f"{'round':>6} " + " ".join(f"{n:>10}" for n in results))
        for r in (1, 5, 10, 15, 20):
            print(f"{r:>6} " + " ".join(f"{results[n][r - 1][metric]:>10.3f}" for n in results))

    Path("measures").mkdir(exist_ok=True)
    with open("measures/aggregation_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "strategy", "accuracy_6", "accuracy_normal"])
        for name, hist in results.items():
            for r, m in enumerate(hist, start=1):
                w.writerow([r, name, m["accuracy_6"], m["accuracy_normal"]])
    print("\nCourbes ecrites dans measures/aggregation_comparison.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("accuracy_6", "accuracy_normal")):
            for name, hist in results.items():
                ax.plot(range(1, ROUNDS + 1), [m[metric] for m in hist], label=name,
                        linestyle="--" if name == "tronquee" else "-")
            ax.set_title(metric)
            ax.set_xlabel("round")
            ax.legend()
        fig.tight_layout()
        fig.savefig("measures/aggregation_comparison.png", dpi=120)
        print("Graphique : measures/aggregation_comparison.png")
    except ImportError:
        print("matplotlib absent : pip install matplotlib pour le graphique")


if __name__ == "__main__":
    main()