"""Un client clinique = un PROCESSUS. Il ne lit que SON csv et ne parle au
serveur que par un tube (Pipe) : messages courts, poids echanges par fichiers.
"""
from pathlib import Path

import pandas as pd

from federation.ids import clinic_num
from federation.metrics import CLINIC_FILES, to_xy
from federation.model import train_local
from federation.params_io import load_params, save_params


def client_main(clinic_id, conn, cfg):
    try:
        df = pd.read_csv(Path(cfg["data_root"]) / CLINIC_FILES[clinic_num(clinic_id)])
        x_raw, y = to_xy(df, cfg["classes"], cfg["cols"])
        x = None                                   # donnees standardisees (apres "scaler")
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "stats":                     # round 0 : n, somme, somme des carres
                conn.send(("stats", clinic_id, len(y), x_raw.sum(axis=0), (x_raw ** 2).sum(axis=0)))
            elif cmd == "scaler":                  # moyenne / ecart-type GLOBAUX
                _, mu, sd = msg
                x = (x_raw - mu) / sd
                conn.send(("ok", clinic_id))
            elif cmd == "train":
                _, _rnd, global_path, out_path = msg
                new = train_local(load_params(global_path), x, y,
                                  lr=cfg["lr"], epochs=cfg["local_epochs"], l2=cfg["l2"])
                save_params(out_path, new)
                conn.send(("update", clinic_id, out_path, int(len(y))))
            elif cmd == "stop":
                break
    except Exception as e:                         # le serveur doit le savoir, pas rester bloque
        conn.send(("error", clinic_id, repr(e)))