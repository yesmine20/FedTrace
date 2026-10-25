"""Metriques et chargement des donnees """
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Diagnosis" # la colonne cible diagnostic
GROUP = "source_group_id" # l’identifiant du groupe de patients, utilisé pour séparer les données sans fuite.
NORMAL = "Normal" # la classe normale, utilisée pour calculer la précision sur les patients normaux.
NON_FEATURES = ["case_pseudo", GROUP, TARGET, "clinic"] # les colonnes qui ne sont pas des variables pour le modèle.
CLINIC_FILES = {k: f"data/clinic_{k}/clinic_{k}.csv" for k in (1, 2, 3)}
TEST_FILE = "data/test/test.csv"

##### charger les données #####
def load_split(root="."):
    """Retourne ({1: df, 2: df, 3: df}, test_df)."""
    root = Path(root)
    clinics = {k: pd.read_csv(root / p) for k, p in CLINIC_FILES.items()}
    return clinics, pd.read_csv(root / TEST_FILE)


def class_names(clinics):
    """Cette fon trie les 6 noms de maladies par ordre alpha (ordre stable pour les indices 0 a 5)."""
    return sorted(set().union(*[set(d[TARGET]) for d in clinics.values()]))


def feature_columns(df):
    """garde toutes les colonnes sauf NON_FEATURES"""
    return [c for c in df.columns if c not in NON_FEATURES]

# preparation pour le modele
def to_xy(df, classes, cols):
    x = df[cols].to_numpy(dtype=np.float64) 
    y = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return x, y

# mettre les variables à la même échelle
def standardizer(x_train):
    """Moyenne et ecart-type du jeu d'entrainement (jamais ceux du test)."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def accuracy_6(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def accuracy_normal(y_true, y_pred, normal_index):
    t = np.asarray(y_true) == normal_index
    p = np.asarray(y_pred) == normal_index
    return float(np.mean(t == p))


def evaluate(y_true, y_pred, classes):
    n = classes.index(NORMAL)
    return {"accuracy_6": accuracy_6(y_true, y_pred),
            "accuracy_normal": accuracy_normal(y_true, y_pred, n)}


def mean_std(rows):
    """rows : liste de dicts de metriques (une par graine) -> {metrique: (moyenne, ecart-type)}."""
    keys = rows[0].keys()
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}