"""Controle A1 : entrainement CENTRALISE avec le modele softmax de A.

But : verifier que model.py apprend sur les vraies donnees avant de federer.
Les hyperparametres sont choisis sur une validation tiree des donnees
d'entrainement (par groupe de patients) ; le test n'est lu qu'une fois, a la fin.
Lancer depuis la racine du depot :  python measures/check_central.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federation.metrics import (GROUP, class_names, evaluate, feature_columns,  # noqa: E402
                                load_split, standardizer, to_xy)
from federation.model import init_params, loss, predict, train_local  # noqa: E402

LR, EPOCHS = 0.1, 300
L2_GRID = [0.0, 0.001, 0.01, 0.1]
OUT_FILE = Path(__file__).resolve().parent / "central_reference.json"


def fit(x, y, l2, log_every=0):
    """Entraine epoch par epoch (meme resultat que train_local(epochs=N))."""
    p = init_params()
    for e in range(1, EPOCHS + 1):
        p = train_local(p, x, y, lr=LR, epochs=1, l2=l2)
        if log_every and e % log_every == 0:
            print(f"   epoque {e:4d}  perte = {loss(p, x, y, l2):.4f}")
    return p


def main():
    clinics, test = load_split()
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    train = pd.concat(clinics.values(), ignore_index=True)

    # validation = 20 % des GROUPES de patients (pas des lignes : pas de fuite)
    groups = np.sort(train[GROUP].unique())
    val_groups = set(np.random.default_rng(0).choice(groups, size=len(groups) // 5, replace=False))
    is_val = train[GROUP].isin(val_groups)
    sub, val = train[~is_val], train[is_val]

    xs, ys = to_xy(sub, classes, cols)
    xv, yv = to_xy(val, classes, cols)
    mu, sd = standardizer(xs)                       # statistiques du train seulement
    xs, xv = (xs - mu) / sd, (xv - mu) / sd

    print("== Choix de l2 sur la validation ==")
    best_l2, best_score = None, -1.0
    val_results = []
    for l2 in L2_GRID:
        p = fit(xs, ys, l2)
        m = evaluate(yv, predict(p, xv), classes)
        val_results.append({"l2": l2, **m})
        print(f"l2={l2:<6} val accuracy_6={m['accuracy_6']:.3f}  accuracy_normal={m['accuracy_normal']:.3f}")
        if m["accuracy_6"] > best_score:
            best_l2, best_score = l2, m["accuracy_6"]
    print(f"-> l2 retenu : {best_l2}\n")

    # reentrainement sur tout le train, puis UNE seule lecture du test
    xa, ya = to_xy(train, classes, cols)
    mu, sd = standardizer(xa)
    xa = (xa - mu) / sd
    xt, yt = to_xy(test, classes, cols)
    xt = (xt - mu) / sd

    print("== Courbe de perte (train complet) ==")
    p = fit(xa, ya, best_l2, log_every=50)
    m = evaluate(yt, predict(p, xt), classes)
    print(f"\nTEST  accuracy_6={m['accuracy_6']:.3f}  accuracy_normal={m['accuracy_normal']:.3f}")
    print("Reference B (sklearn, centralise) : 0.296 / 1.000 ; seuils : > 0.212 et >= 0.95")

    # sauvegarde : hyperparametres retenus + metriques, pour A, pour B et pour le rapport
    result = {
        "model": "softmax (federation/model.py)",
        "hyperparameters": {"lr": LR, "epochs": EPOCHS, "l2": best_l2, "batch": "full-batch"},
        "standardization": "moyenne/ecart-type du jeu d'entrainement (3 cliniques reunies)",
        "validation": {"split": "20 % des groupes source_group_id, graine 0",
                       "criterion": "accuracy_6", "results_per_l2": val_results},
        "test": {"accuracy_6": m["accuracy_6"], "accuracy_normal": m["accuracy_normal"],
                 "n_test": int(len(test))},
        "n_train": int(len(train)),
        "final_train_loss": loss(p, xa, ya, best_l2),
    }
    OUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultats enregistres dans {OUT_FILE}")


if __name__ == "__main__":
    main()