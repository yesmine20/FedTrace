"""Controle A1 : chaque clinique entraine SEULE le modele softmax de A.

Pour chaque clinique (et pour le centralise, en comparaison) :
  - entrainement full-batch avec les hyperparametres de central_reference.json
    (aucun reglage sur le test : il n'est lu qu'une fois par modele) ;
  - metriques sur le test commun + rappel par maladie ;
  - sauvegarde dans measures/local_reference.json.
Standardisation : moyenne / ecart-type des 3 jeux d'entrainement reunis, exactement
ce que le round 0 de la federation calcule -> comparaison directe avec la federation.

Lancer depuis la racine du depot :  python measures/check_local.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federation.metrics import (class_names, evaluate, feature_columns,  # noqa: E402
                                load_split, standardizer, to_xy)
from federation.model import init_params, loss, predict, train_local  # noqa: E402

HERE = Path(__file__).resolve().parent
REF = HERE / "central_reference.json"
OUT = HERE / "local_reference.json"


def hyperparameters():
    if REF.exists():
        hp = json.loads(REF.read_text(encoding="utf-8"))["hyperparameters"]
        return hp["lr"], hp["epochs"], hp["l2"]
    return 0.1, 300, 0.1


def recall_per_class(y, pred, n_classes):
    """Part des patients d'une maladie que le modele classe correctement."""
    return [float(np.mean(pred[y == c] == c)) if (y == c).any() else float("nan")
            for c in range(n_classes)]


def main():
    lr, epochs, l2 = hyperparameters()
    clinics, test = load_split()
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    train_all = pd.concat(clinics.values(), ignore_index=True)

    x_all, _ = to_xy(train_all, classes, cols)
    mu, sd = standardizer(x_all)
    xt, yt = to_xy(test, classes, cols)
    xt = (xt - mu) / sd

    datasets = {f"clinic_{k}": d for k, d in clinics.items()}
    datasets["centralise"] = train_all

    results = {}
    print(f"lr={lr}  epochs={epochs}  l2={l2}   (perte de depart = ln 6 = 1.792)\n")
    print(f"{'modele':<11}{'n_train':>8}{'perte':>8}{'acc6 train':>12}{'acc6 test':>11}{'accN test':>11}")
    for name, df in datasets.items():
        x, y = to_xy(df, classes, cols)
        x = (x - mu) / sd
        p = train_local(init_params(), x, y, lr=lr, epochs=epochs, l2=l2)
        m = evaluate(yt, predict(p, xt), classes)
        acc_train = float(np.mean(predict(p, x) == y))
        results[name] = {"n_train": int(len(y)), "final_train_loss": loss(p, x, y, l2),
                         "accuracy_6_train": acc_train, **m,
                         "recall_per_class_test": dict(zip(classes, recall_per_class(yt, predict(p, xt), len(classes))))}
        print(f"{name:<11}{len(y):>8}{results[name]['final_train_loss']:>8.3f}{acc_train:>12.3f}"
              f"{m['accuracy_6']:>11.3f}{m['accuracy_normal']:>11.3f}")

    print("\nRappel par maladie sur le test (part des cas bien classes) :")
    short = [c[:12] for c in classes]
    print(f"{'modele':<11}" + "".join(f"{s:>14}" for s in short))
    for name, r in results.items():
        print(f"{name:<11}" + "".join(f"{v:>14.2f}" for v in r["recall_per_class_test"].values()))

    se = (0.3 * 0.7 / len(yt)) ** 0.5
    print(f"\nMarge d'incertitude du test : ~ +/-{se:.3f} sur acc6 -> un ecart < 0.02 n'est pas significatif.")
    print("Reference B (sklearn, seul) : clinic_1 0.343 | clinic_2 0.254 | clinic_3 0.345 | centralise 0.296")

    OUT.write_text(json.dumps({"hyperparameters": {"lr": lr, "epochs": epochs, "l2": l2},
                               "standardization": "moyenne/ecart-type des 3 jeux d'entrainement reunis",
                               "n_test": int(len(yt)), "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultats enregistres dans {OUT}")


if __name__ == "__main__":
    main()