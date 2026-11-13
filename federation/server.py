"""Serveur et orchestrateur d'apprentissage fédéré (FedTrace - Étape A1).

Orchestre 3 clients (chacun dans un processus dédié via multiprocessing.Process
et multiprocessing.Pipe). Exécute l'agrégation FedAvg pondérée par n_samples.
Comporte les crochets on_local_update et on_round_end réservés pour l'ancrage
blockchain de la Personne B.
"""
import json
import multiprocessing as mp
from pathlib import Path
import sys
import numpy as np

# Permet d'importer federation même si le script est lancé depuis n'importe où
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federation.client import client_main
from federation.ids import round_id, clinic_id
from federation.metrics import (
    load_split, class_names, feature_columns, to_xy, evaluate
)
from federation.model import init_params, loss, predict
from federation.params_io import save_params, load_params


# ---------------------------------------------------------------------------
# Crochets d'ancrage (Hooks pour la Personne B)
# ---------------------------------------------------------------------------
def on_local_update(cid: str, round_idx: int, update_path: str, n_samples: int) -> None:
    """Crochet appelé dès réception de la mise à jour d'une clinique (avant agrégation).
    À remplir ultérieurement par B (B2/B4) pour l'ancrage de la contribution.
    """
    pass


def on_round_end(round_idx: int, global_path: str, metrics: dict, retained: list, discarded: list) -> None:
    """Crochet appelé à la fin de chaque round (après agrégation et évaluation).
    À remplir ultérieurement par B (B2/B4) pour l'ancrage du modèle global.
    """
    pass


# ---------------------------------------------------------------------------
# Agrégation FedAvg (Moyenne pondérée par n_samples)
# ---------------------------------------------------------------------------
def aggregate_fedavg(updates: list[dict]) -> dict:
    """Calcule la moyenne pondérée des paramètres reçus des cliniques.

    updates : liste de dicts contenant {'clinic_id': str, 'path': str, 'n_samples': int}
    Retourne : dict {"W": ..., "b": ...} en float64.
    """
    total_samples = sum(u["n_samples"] for u in updates)
    if total_samples <= 0:
        raise ValueError("Le nombre total d'échantillons doit être strictement positif.")

    first_params = load_params(updates[0]["path"])
    w_shape = first_params["W"].shape
    b_shape = first_params["b"].shape

    w_acc = np.zeros(w_shape, dtype=np.float64)
    b_acc = np.zeros(b_shape, dtype=np.float64)

    for u in updates:
        p = load_params(u["path"])
        weight = u["n_samples"] / total_samples
        w_acc += weight * p["W"]
        b_acc += weight * p["b"]

    return {"W": w_acc, "b": b_acc}


# ---------------------------------------------------------------------------
# Orchestration de la Fédération
# ---------------------------------------------------------------------------
def run_federation(n_rounds: int = 20, lr: float = 0.1, local_epochs: int = 5, l2: float = 0.1, seed: int = 42):
    """Lance la fédération sur n_rounds avec les 3 cliniques."""
    print("=" * 70)
    print(f"Démarrage FedTrace — Serveur Fédéré (FedAvg)")
    print(f"Rounds: {n_rounds} | LR: {lr} | Époques locales: {local_epochs} | L2: {l2} | Graine: {seed}")
    print("=" * 70)

    # 1. Chargement des données et métadonnées
    clinics, test_df = load_split(str(PROJECT_ROOT))
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    n_clinics = len(clinics)

    cfg = {
        "data_root": str(PROJECT_ROOT),
        "classes": classes,
        "cols": cols,
        "lr": lr,
        "local_epochs": local_epochs,
        "l2": l2,
    }

    # 2. Démarrage des 3 processus clients
    pipes = {}
    processes = []
    for k in range(1, n_clinics + 1):
        cid = clinic_id(k)
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(target=client_main, args=(cid, child_conn, cfg), name=f"Proc_{cid}")
        proc.start()
        pipes[cid] = parent_conn
        processes.append(proc)

    try:
        # 3. Synchronisation Round 0 : Normalisation commune (moyenne et écart-type)
        print("\n[Round 000] Calcul du centrage-réduction global...")
        total_n = 0
        sum_x = None
        sum_sq_x = None

        for cid, conn in pipes.items():
            conn.send(("stats",))
            resp = conn.recv()
            if resp[0] != "stats":
                raise RuntimeError(f"Erreur reçue de {cid} lors de 'stats' : {resp}")
            _, _, n_k, sum_k, sum_sq_k = resp
            total_n += n_k
            if sum_x is None:
                sum_x = np.copy(sum_k)
                sum_sq_x = np.copy(sum_sq_k)
            else:
                sum_x += sum_k
                sum_sq_x += sum_sq_k

        global_mu = sum_x / total_n
        var_x = (sum_sq_x / total_n) - (global_mu ** 2)
        var_x = np.maximum(var_x, 1e-12)
        global_sd = np.sqrt(var_x)
        global_sd[global_sd == 0] = 1.0

        # Envoi du scaler calculé à chaque client
        for cid, conn in pipes.items():
            conn.send(("scaler", global_mu, global_sd))
            resp = conn.recv()
            if resp != ("ok", cid):
                raise RuntimeError(f"Erreur synchronisation scaler avec {cid} : {resp}")

        # Sauvegarde du fichier preprocess.json (exigé par l'étape 0)
        preprocess_data = {
            "means": dict(zip(cols, global_mu.tolist())),
            "stds": dict(zip(cols, global_sd.tolist())),
            "n_samples": int(total_n),
        }
        (PROJECT_ROOT / "data" / "preprocess.json").write_text(
            json.dumps(preprocess_data, indent=2), encoding="utf-8"
        )
        # Sauvegarde du dictionnaire des labels (exigé par l'étape 0)
        labels_data = {c: i for i, c in enumerate(classes)}
        (PROJECT_ROOT / "data" / "labels.json").write_text(
            json.dumps(labels_data, indent=2), encoding="utf-8"
        )

        # Préparation du test (centré-réduit avec les mêmes constantes d'entraînement)
        xt, yt = to_xy(test_df, classes, cols)
        xt = (xt - global_mu) / global_sd

        # Initialisation du modèle global (Round 0)
        global_params = init_params()
        r0_dir = PROJECT_ROOT / "offchain" / round_id(0)
        r0_dir.mkdir(parents=True, exist_ok=True)
        r0_path = r0_dir / "global_model.npz"
        save_params(r0_path, global_params)

        m0 = evaluate(yt, predict(global_params, xt), classes)
        loss0 = loss(global_params, xt, yt, l2=0.0)
        print(f"[Round 000] Modèle initialisé | Perte test: {loss0:.4f} | acc6: {m0['accuracy_6']:.3f} | accN: {m0['accuracy_normal']:.3f}")
        on_round_end(0, str(r0_path), m0, retained=[clinic_id(k) for k in range(1, n_clinics + 1)], discarded=[])

        # 4. Boucle des rounds de fédération
        history = [{"round": 0, "loss": loss0, **m0}]
        current_global_path = r0_path

        print("\n" + f"{'Round':<8}{'Perte test':>12}{'accuracy_6':>14}{'accuracy_normal':>18}")
        print("-" * 54)

        for r in range(1, n_rounds + 1):
            rid = round_id(r)
            round_dir = PROJECT_ROOT / "offchain" / rid
            round_dir.mkdir(parents=True, exist_ok=True)

            # a) Lancement de l'entraînement local pour les 3 cliniques
            for k in range(1, n_clinics + 1):
                cid = clinic_id(k)
                out_path = round_dir / f"{cid}.npz"
                pipes[cid].send(("train", r, str(current_global_path), str(out_path)))

            # b) Réception des résultats
            updates = []
            for k in range(1, n_clinics + 1):
                cid = clinic_id(k)
                resp = pipes[cid].recv()
                if resp[0] == "error":
                    raise RuntimeError(f"Erreur dans {cid} : {resp[2]}")
                _, _, out_path, n_samples = resp

                # Crochet avant agrégation
                on_local_update(cid, r, out_path, n_samples)
                updates.append({"clinic_id": cid, "path": out_path, "n_samples": n_samples})

            # c) Agrégation FedAvg
            new_params = aggregate_fedavg(updates)
            next_global_path = round_dir / "global_model.npz"
            save_params(next_global_path, new_params)
            current_global_path = next_global_path

            # d) Évaluation sur test.csv
            preds = predict(new_params, xt)
            m = evaluate(yt, preds, classes)
            round_loss = loss(new_params, xt, yt, l2=0.0)

            history.append({"round": r, "loss": round_loss, **m})
            print(f"{rid:<8}{round_loss:>12.4f}{m['accuracy_6']:>14.3f}{m['accuracy_normal']:>18.3f}")

            # Crochet après agrégation
            retained = [u["clinic_id"] for u in updates]
            on_round_end(r, str(current_global_path), m, retained=retained, discarded=[])

        # Sauvegarde de l'historique complet pour les analyses ultérieures
        history_path = PROJECT_ROOT / "measures" / "federation_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("-" * 54)
        print(f"\nFédération terminée avec succès. Historique enregistré dans : {history_path}")

    finally:
        # 5. Arrêt propre des processus clients
        print("\nArrêt des processus clients...")
        for cid, conn in pipes.items():
            try:
                conn.send(("stop",))
            except Exception:
                pass
        for proc in processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        print("Tous les processus ont été arrêtés proprement.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serveur d'apprentissage fédéré (FedTrace)")
    parser.add_argument("--rounds", type=int, default=20, help="Nombre de rounds de fédération (défaut: 20)")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques locales par round (défaut: 5)")
    parser.add_argument("--lr", type=float, default=0.1, help="Taux d'apprentissage (défaut: 0.1)")
    parser.add_argument("--l2", type=float, default=0.1, help="Coefficient de régularisation L2 (défaut: 0.1)")
    args = parser.parse_args()

    run_federation(n_rounds=args.rounds, lr=args.lr, local_epochs=args.epochs, l2=args.l2)

