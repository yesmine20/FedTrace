"""Serveur fédéré FedTrace — Étape A1.

Chaque run est horodaté et isolé dans offchain/runs/<run_id>/ :
  offchain/runs/
  └── 20261113_152301_fedavg/          # un run = un dossier unique
      ├── run_meta.json                # stratégie, hyperparamètres, seed, ...
      ├── r000/global_model.npz        # modèle initial W=0, b=0
      ├── r001/
      │   ├── clinic_1.npz            # poids locaux après entraînement local
      │   ├── clinic_2.npz
      │   ├── clinic_3.npz
      │   └── global_model.npz        # poids globaux après agrégation FedAvg
      └── ...r020/

Un nouveau run n'écrase jamais un run précédent.
Les crochets on_local_update / on_round_end sont réservés à la Personne B
(agent d'ancrage blockchain, étapes B2/B4).
"""
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federation.client import client_main
from federation.ids import round_id, clinic_id
from federation.metrics import (
    CLINIC_FILES, load_split, class_names, feature_columns, to_xy, evaluate
)
from federation.model import init_params, loss, predict
from federation.params_io import save_params, load_params
from federation.robust import robust_aggregate
from federation.poisoning import ATTACK_TYPES, validate_attack


# ---------------------------------------------------------------------------
# Crochets d'ancrage — Hooks pour la Personne B (B2/B4)
# NE PAS modifier ces signatures : B les remplace par son agent.
# ---------------------------------------------------------------------------
def on_local_update(
    run_id: str,
    cid: str,
    round_idx: int,
    update_path: str,
    n_samples: int,
) -> None:
    """Appelé dès réception de la mise à jour locale d'une clinique.
    Paramètres transmis à B pour construire l'enregistrement 'contribution'.
    """
    pass


def on_round_end(
    run_id: str,
    round_idx: int,
    parent_round_id: str,
    global_path: str,
    metrics: dict,
    strategy: str,
    retained: list,
    discarded: list,
) -> None:
    """Appelé après agrégation et évaluation du modèle global.
    Paramètres transmis à B pour construire l'enregistrement 'model'.
    """
    pass


# ---------------------------------------------------------------------------
# Agrégation FedAvg — Moyenne pondérée par n_samples
# ---------------------------------------------------------------------------
def aggregate_fedavg(updates: list[dict]) -> dict:
    """Moyenne pondérée des paramètres locaux.

    updates : liste de dicts {'clinic_id', 'path', 'n_samples'}
    Retourne : {'W': ndarray, 'b': ndarray}
    """
    total_samples = sum(u["n_samples"] for u in updates)
    if total_samples <= 0:
        raise ValueError("Le nombre total d'échantillons doit être > 0.")

    first = load_params(updates[0]["path"])
    w_acc = np.zeros_like(first["W"])
    b_acc = np.zeros_like(first["b"])

    for u in updates:
        p = load_params(u["path"])
        w = u["n_samples"] / total_samples
        w_acc += w * p["W"]
        b_acc += w * p["b"]

    return {"W": w_acc, "b": b_acc}


# ---------------------------------------------------------------------------
# Agrégation robuste — Médiane + filtrage L2 (implémentation B3)
# ---------------------------------------------------------------------------
def aggregate_robust_median(updates: list[dict], tau: float) -> dict:
    """Charge les poids et délègue à federation/robust.py (étape B3).

    Retourne : {'params', 'retained', 'discarded', 'distances'}
    """
    params_by_clinic = {u["clinic_id"]: load_params(u["path"]) for u in updates}
    samples_by_clinic = {u["clinic_id"]: u["n_samples"] for u in updates}
    return robust_aggregate(params_by_clinic, samples_by_clinic, tau)


# ---------------------------------------------------------------------------
# Création du dossier de run et écriture du manifeste
# ---------------------------------------------------------------------------
def _make_run_dir(strategy: str, tau: float | None, n_rounds: int,
                  lr: float, local_epochs: int, l2: float, seed: int,
                  attack: dict) -> tuple[str, Path]:
    """Crée offchain/runs/<timestamp>_<strategy>/ et écrit run_meta.json.

    Retourne (run_id, run_dir).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"{strategy}_tau{tau}" if (strategy == "robust_median" and tau is not None) else strategy
    if attack["type"] != "none":
        suffix += f"_{attack['type']}_{attack['clinic_id']}_r{attack['start_round']}"
    run_id = f"{ts}_{suffix}"
    run_dir = PROJECT_ROOT / "offchain" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "tau": tau,
        "attack": attack,
        "hyperparameters": {
            "n_rounds": n_rounds,
            "lr": lr,
            "local_epochs": local_epochs,
            "l2": l2,
            "seed": seed,
        },
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_id, run_dir


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------
def run_federation(
    n_rounds: int = 20,
    lr: float = 0.1,
    local_epochs: int = 5,
    l2: float = 0.1,
    seed: int = 42,
    strategy: str = "fedavg",
    tau: float | None = None,
    attack: dict | None = None,
) -> str:
    """Lance une fédération complète et retourne le run_id créé.

    strategy : 'fedavg' ou 'robust_median' (aligné sur model.schema.json)
    tau      : seuil de rejet pour robust_median (ignoré pour fedavg)
    """
    # --- Validation des arguments ---
    VALID_STRATEGIES = {"fedavg", "robust_median"}
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Stratégie invalide '{strategy}'. Valeurs acceptées : {VALID_STRATEGIES}")
    if strategy == "robust_median":
        if tau is None or not np.isfinite(tau) or tau <= 0:
            raise ValueError("robust_median exige un tau fini et strictement positif (ex: --tau 1.3).")

    attack_cfg = validate_attack(attack)
    if attack_cfg["type"] != "none":
        available = {clinic_id(k) for k in CLINIC_FILES}
        if attack_cfg["clinic_id"] not in available:
            raise ValueError(f"Clinique ciblée absente des données : {attack_cfg['clinic_id']}")

    # --- Création du dossier de run isolé ---
    run_id, run_dir = _make_run_dir(strategy, tau, n_rounds, lr, local_epochs, l2, seed, attack_cfg)

    print("=" * 72)
    print(f"  FedTrace — Run : {run_id}")
    print(f"  Stratégie : {strategy}" + (f" | τ = {tau}" if strategy == "robust_median" else ""))
    print(f"  Attaque : {attack_cfg['type']}" + (
        f" sur {attack_cfg['clinic_id']} dès r{attack_cfg['start_round']:03d}, force={attack_cfg['strength']}"
        if attack_cfg["type"] != "none" else " (aucune)"
    ))
    print(f"  Rounds : {n_rounds} | LR : {lr} | Époques : {local_epochs} | L2 : {l2} | Seed : {seed}")
    print(f"  Dossier de run : offchain/runs/{run_id}/")
    print("=" * 72)

    # Fixer la graine aléatoire globale
    np.random.seed(seed)

    # --- Chargement des données ---
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
        "attack": attack_cfg,
    }

    # --- Démarrage des processus clients ---
    pipes: dict[str, mp.connection.Connection] = {}
    processes: list[mp.Process] = []
    for k in range(1, n_clinics + 1):
        cid = clinic_id(k)
        client_cfg = {**cfg, "seed": seed + k}
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(
            target=client_main,
            args=(cid, child_conn, client_cfg),
            name=f"FedTrace_{cid}",
            daemon=True,
        )
        proc.start()
        pipes[cid] = parent_conn
        processes.append(proc)

    try:
        # ----------------------------------------------------------------
        # Round 0 : Normalisation globale + modèle initial
        # ----------------------------------------------------------------
        print(f"\n[r000] Collecte des statistiques pour la normalisation...")
        total_n = 0
        sum_x: np.ndarray | None = None
        sum_sq_x: np.ndarray | None = None

        for cid, conn in pipes.items():
            conn.send(("stats",))
            resp = conn.recv()
            if resp[0] != "stats":
                raise RuntimeError(f"{cid} → réponse inattendue à 'stats' : {resp}")
            _, _, n_k, sum_k, sum_sq_k = resp
            total_n += n_k
            if sum_x is None:
                sum_x = sum_k.copy()
                sum_sq_x = sum_sq_k.copy()
            else:
                sum_x += sum_k
                sum_sq_x += sum_sq_k

        # Moyenne et écart-type globaux (calculés sur les données d'entraînement uniquement)
        global_mu = sum_x / total_n
        var_x = (sum_sq_x / total_n) - global_mu ** 2
        global_sd = np.sqrt(np.maximum(var_x, 0.0))
        global_sd[var_x < 1e-10] = 1.0
        print(f"[r000] Normalisation : {total_n} exemples d'entraînement, {len(cols)} features.")

        # Envoi du scaler à chaque client
        for cid, conn in pipes.items():
            conn.send(("scaler", global_mu, global_sd))
            resp = conn.recv()
            if resp != ("ok", cid):
                raise RuntimeError(f"Échec synchronisation scaler avec {cid} : {resp}")

        # Persistance des constantes de normalisation (étape 0 du sujet)
        preprocess_data = {
            "means": dict(zip(cols, global_mu.tolist())),
            "stds": dict(zip(cols, global_sd.tolist())),
            "n_samples_train": int(total_n),
        }
        (PROJECT_ROOT / "data" / "preprocess.json").write_text(
            json.dumps(preprocess_data, indent=2), encoding="utf-8"
        )
        (PROJECT_ROOT / "data" / "labels.json").write_text(
            json.dumps({c: i for i, c in enumerate(classes)}, indent=2), encoding="utf-8"
        )

        # Jeu de test normalisé (jamais utilisé pour calculer mu/sd)
        xt, yt = to_xy(test_df, classes, cols)
        xt = (xt - global_mu) / global_sd

        # Modèle initial W=0, b=0 — enregistré dans r000/ du run courant
        global_params = init_params()
        r0_dir = run_dir / round_id(0)
        r0_dir.mkdir(exist_ok=True)
        r0_path = r0_dir / "global_model.npz"
        save_params(r0_path, global_params)

        m0 = evaluate(yt, predict(global_params, xt), classes)
        loss0 = loss(global_params, xt, yt, l2=0.0)
        print(f"[r000] Modèle initial  | Perte : {loss0:.4f} | acc6 : {m0['accuracy_6']:.3f} | accN : {m0['accuracy_normal']:.3f}")
        on_round_end(run_id, 0, "", str(r0_path), m0, strategy="init", retained=[], discarded=[])

        # ----------------------------------------------------------------
        # Boucle des rounds
        # ----------------------------------------------------------------
        history = [{"round": 0, "strategy": "init", "loss": loss0,
                    "retained": [], "discarded": {}, **m0}]
        current_global_path = r0_path

        col_w = 72
        print("\n" + f"{'Round':<8}{'Perte test':>12}{'accuracy_6':>14}{'accuracy_normal':>18}{'Rejetées':>12}")
        print("-" * col_w)

        for r in range(1, n_rounds + 1):
            rid = round_id(r)
            round_dir = run_dir / rid
            round_dir.mkdir(exist_ok=True)

            # a) Ordre d'entraînement local à toutes les cliniques
            for k in range(1, n_clinics + 1):
                cid = clinic_id(k)
                out_path = round_dir / f"{cid}.npz"
                pipes[cid].send(("train", r, str(current_global_path), str(out_path)))

            # b) Réception des mises à jour locales
            updates: list[dict] = []
            for k in range(1, n_clinics + 1):
                cid = clinic_id(k)
                resp = pipes[cid].recv()
                if resp[0] == "error":
                    raise RuntimeError(f"Erreur dans le processus {cid} : {resp[2]}")
                _, _, out_path, n_samples = resp
                # Crochet B : ancrage de la contribution
                on_local_update(run_id, cid, r, out_path, n_samples)
                updates.append({"clinic_id": cid, "path": out_path, "n_samples": n_samples})

            # c) Agrégation selon la stratégie choisie
            distances: dict = {}
            if strategy == "fedavg":
                new_params = aggregate_fedavg(updates)
                retained = [u["clinic_id"] for u in updates]
                discarded = []
            else:  # robust_median
                result = aggregate_robust_median(updates, tau)
                retained = result["retained"]
                discarded = result["discarded"]
                distances = result["distances"]
                if not retained or result["params"] is None:
                    print(f"  [ALERTE {rid}] Toutes les cliniques ont été rejetées ! Modèle précédent conservé.")
                    new_params = load_params(current_global_path)
                else:
                    new_params = result["params"]

            # d) Sauvegarde du modèle global de ce round
            next_global_path = round_dir / "global_model.npz"
            save_params(next_global_path, new_params)
            current_global_path = next_global_path

            # e) Évaluation sur le jeu de test commun
            m = evaluate(yt, predict(new_params, xt), classes)
            round_loss = loss(new_params, xt, yt, l2=0.0)

            discarded_ids = [d["clinic_id"] for d in discarded] if discarded else []
            history.append({
                "round": r,
                "strategy": strategy,
                "loss": round_loss,
                "retained": retained,
                "discarded": discarded,
                "distances": distances,
                **m,
            })
            print(
                f"{rid:<8}{round_loss:>12.4f}{m['accuracy_6']:>14.3f}"
                f"{m['accuracy_normal']:>18.3f}{str(discarded_ids):>12}"
            )

            # Crochet B : ancrage du modèle global
            parent_rid = round_id(r - 1)
            on_round_end(run_id, r, parent_rid, str(current_global_path), m,
                         strategy=strategy, retained=retained, discarded=discarded)

        # ----------------------------------------------------------------
        # Fin du run : sauvegarde de l'historique dans le dossier du run
        # ----------------------------------------------------------------
        print("-" * col_w)
        history_path = run_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        # Lien symbolique measures/latest_history.json -> run courant (optionnel, Windows)
        latest = PROJECT_ROOT / "measures" / "federation_history.json"
        latest.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        m_final = history[-1]
        print(f"\n✓ Run '{run_id}' terminé.")
        print(f"  Round final : acc6 = {m_final['accuracy_6']:.3f} | accN = {m_final['accuracy_normal']:.3f}")
        print(f"  Historique  : offchain/runs/{run_id}/history.json")
        return run_id

    finally:
        # Arrêt propre de tous les processus clients
        for cid, conn in pipes.items():
            try:
                conn.send(("stop",))
            except Exception:
                pass
        for proc in processes:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="Serveur fédéré FedTrace — chaque exécution crée un run isolé.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds",   type=int,   default=config.get("rounds", 20),     help="Nombre de rounds")
    parser.add_argument("--epochs",   type=int,   default=5,                             help="Époques locales par round")
    parser.add_argument("--lr",       type=float, default=0.1,                           help="Taux d'apprentissage")
    parser.add_argument("--l2",       type=float, default=0.1,                           help="Régularisation L2")
    parser.add_argument("--seed",     type=int,   default=config.get("seed", 42),        help="Graine aléatoire")
    parser.add_argument(
        "--strategy",
        choices=("fedavg", "robust_median"),
        default=config.get("strategy", "fedavg"),
        help="Stratégie d'agrégation (aligné sur model.schema.json)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=config.get("tau"),
        help="Seuil τ pour robust_median (ex: 1.3)",
    )
    parser.add_argument(
        "--attack-type",
        choices=tuple(sorted(ATTACK_TYPES)),
        default="none",
        help="Attaque simulée à appliquer côté client (défaut: aucune)",
    )
    parser.add_argument(
        "--attack-clinic", default="clinic_3",
        help="Clinique ciblée (défaut: clinic_3)",
    )
    parser.add_argument(
        "--attack-start-round", type=int, default=1,
        help="Premier round attaqué (défaut: 1)",
    )
    parser.add_argument(
        "--attack-strength", type=float, default=10.0,
        help="Force : k pour inversion, multiplicateur RMS pour bruit, fraction pour label_flip",
    )
    args = parser.parse_args()

    run_federation(
        n_rounds=args.rounds,
        lr=args.lr,
        local_epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
        strategy=args.strategy,
        tau=args.tau,
        attack={
            "type": args.attack_type,
            "clinic_id": args.attack_clinic,
            "start_round": args.attack_start_round,
            "strength": args.attack_strength,
        },
    )
