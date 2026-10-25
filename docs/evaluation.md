# Protocole d'évaluation (B1)

## 1. But
Fixer des règles de mesure communes à A et B, et calculer des notes de référence avant d'avoir le système fédéré. Ces notes servent à juger plus tard le système fédéré, les attaques et la défense robuste.

## 2. Données
| Fichier | Rôle |
|---|---|
| `data/clinic_1/clinic_1.csv` | entraînement de la clinique 1 |
| `data/clinic_2/clinic_2.csv` | entraînement de la clinique 2 |
| `data/clinic_3/clinic_3.csv` | entraînement de la clinique 3 |
| `data/test/test.csv` | test commun (2 780 patients) |

- Les groupes de patients (`source_group_id`) du test sont absents des 3 cliniques (vérifié par `tests/test_data_split.py`).
- 61 variables : toutes les colonnes sauf `case_pseudo`, `source_group_id`, `Diagnosis` et `clinic`.
- Le test n'est jamais utilisé pour entraîner, ni pour calculer la moyenne et l'écart-type de standardisation.

## 3. Mesures (`federation/metrics.py`)
- `accuracy_6` : part de bons diagnostics sur les 6 classes.
- `accuracy_normal` : « Normal contre maladie ». Une confusion entre deux maladies ne compte pas comme erreur ; confondre un sain avec un malade, si.
- Chaque expérience est répétée avec 5 graines (0 à 4). Résultat = moyenne ± écart-type.

Repères sur le test :
| Repère | accuracy_6 |
|---|---|
| Hasard sur 6 classes | 0,167 |
| Toujours répondre la classe majoritaire du test | 0,212 |

## 4. Références
Modèle utilisé : régression logistique de scikit-learn (famille softmax, comme le modèle de A, mais avec régularisation et optimiseur différents). À chaque graine, 80 % des lignes d'entraînement sont tirées au hasard.

| Modèle de référence entraîné sur | accuracy_6 | accuracy_normal |
|---|---|---|
| 3 cliniques réunies (centralisé) | 0,296 ± 0,010 | 1,000 ± 0,000 |
| Clinique 1 seule | 0,343 ± 0,008 | 0,999 ± 0,000 |
| Clinique 2 seule | 0,254 ± 0,008 | 0,985 ± 0,002 |
| Clinique 3 seule | 0,345 ± 0,008 | 0,980 ± 0,001 |

## 5. Analyse non-IID
Part de chaque diagnostic dans chaque clinique :

| Diagnostic | Clinique 1 | Clinique 2 | Clinique 3 | Moyenne |
|---|---|---|---|---|
| Autoimmune orchitis | 0,263 | 0,193 | 0,074 | 0,177 |
| Graves' disease | 0,104 | 0,327 | 0,046 | 0,159 |
| Normal | 0,148 | 0,220 | 0,151 | 0,173 |
| Rheumatoid arthritis | 0,195 | 0,099 | 0,207 | 0,167 |
| Sjögren syndrome | 0,144 | 0,073 | 0,222 | 0,146 |
| Systemic lupus erythematosus (SLE) | 0,146 | 0,089 | 0,301 | 0,178 |

Distance de variation totale à la moyenne (0 = identique à la moyenne) : clinique 1 = 0,115 ; clinique 2 = 0,231 ; clinique 3 = 0,238.

Conséquence pour la défense robuste (B3) : une clinique honnête peut s'éloigner beaucoup de la moyenne. Le seuil τ doit être calibré sur des rounds sans attaque, pour ne pas écarter une clinique honnête.

## 6. Seuils acceptables (proposition, à valider avec A)
| Situation | Mesure | Seuil |
|---|---|---|
| Fédération sans attaque | `accuracy_normal` | au moins 0,95 |
| Fédération sans attaque | `accuracy_6` | au-dessus de 0,212 |
| Agrégation robuste contre moyenne pondérée, sans attaque | les deux mesures | écart d'au plus 0,02 |
| Avec attaque, moyenne pondérée | `accuracy_6` | baisse nette (l'attaque fonctionne) |
| Avec attaque, agrégation robuste | les deux mesures | écart d'au plus 0,02 par rapport au cas sans attaque |

Le seuil de 0,02 est du même ordre que l'écart entre graines (environ 0,01) : en dessous, on ne peut pas affirmer qu'il y a une différence.

## 7. Lecture et limites
- Normal contre maladie est presque parfait partout (0,98 à 1,00). La classe Normal est répartie de façon assez homogène entre les cliniques (15 % à 22 %), ce qui peut l'expliquer, sans que ce soit démontré.
- Les 5 maladies se distinguent mal sur ce jeu de données synthétique : `accuracy_6` reste entre 0,25 et 0,35. Ce n'est pas une performance clinique.
- Observation non expliquée : la clinique 1 seule (0,343) fait mieux que le modèle sur les 3 cliniques réunies (0,296).
- Les références utilisent la régression logistique de scikit-learn, pas le modèle du projet. Elles donnent un ordre de grandeur, pas la valeur exacte.

## 8. À faire quand A livre son modèle et son serveur (A1)
1. Recalculer la référence centralisée et les références locales avec **le modèle de A**, pour comparer le même modèle des deux côtés.
2. Lancer 20 rounds avec 3 cliniques honnêtes et comparer moyenne pondérée, médiane et moyenne tronquée à ces références.
3. Tracer chaque mesure par round, avec 5 graines.
4. Mettre à jour les tableaux 4 et 6 de ce document avec les nouvelles valeurs.

## 9. Reproduire les résultats
```
pip install -r requirements.txt
python measures/baselines.py
python -m pytest -q tests
```
Résultat attendu des tests : tous passent (`tests/test_data_split.py` et `tests/test_evaluation.py`).