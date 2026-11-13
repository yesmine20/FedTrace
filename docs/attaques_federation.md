# Attaques de fédération (étape A1)

Les attaques sont simulées dans le processus de la clinique ciblée. Elles agissent
sur sa contribution avant que le serveur la reçoive et avant le crochet
`on_local_update`. Les autres cliniques continuent leur entraînement normal.

## Attaques disponibles

- `sign_inversion` : remplace le déplacement local `local - global` par son
  opposé, multiplié par la force `k`. Avec `k=1`, la clinique renvoie exactement
  le déplacement dans le sens opposé ; une valeur plus grande l'amplifie.
- `gaussian_noise` : ajoute un bruit normal aux poids locaux. Son écart-type est
  `force × RMS(déplacement local)` ; `1` correspond à l'échelle du déplacement.
- `label_flip` : avant l'entraînement local, choisit aléatoirement la proportion
  indiquée des étiquettes et décale chacune vers la classe suivante
  (`0 → 1`, ..., dernière classe `→ 0`). Le choix est reproductible avec la
  graine du run.

Par défaut, aucune attaque n'est appliquée. Lorsqu'une attaque est choisie, la
clinique ciblée est `clinic_3`, le premier round est `1` et la force vaut `10`.
Sans `--attack-end-round`, elle continue jusqu'au dernier round. Avec cette
option, elle s'arrête au round indiqué.
La force de `label_flip` doit être comprise entre `0` et `1` (par exemple `0.3`
pour 30 %). Pour comparer les résultats proprement, garder les mêmes paramètres
et la même graine entre le run de référence et les runs attaqués.

## Commandes PowerShell

Depuis la racine du projet, établir d'abord le résultat de référence FedAvg sans
attaque :

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --rounds 20 --epochs 5 --lr 0.1 --l2 0.1 --seed 42
```

Puis lancer un run FedAvg séparé pour chaque attaque :

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --rounds 20 --epochs 5 --lr 0.1 --l2 0.1 --seed 42 --attack-type sign_inversion --attack-clinic clinic_3 --attack-start-round 1 --attack-strength 1
```

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --rounds 20 --epochs 5 --lr 0.1 --l2 0.1 --seed 42 --attack-type gaussian_noise --attack-clinic clinic_3 --attack-start-round 1 --attack-strength 1
```

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --rounds 20 --epochs 5 --lr 0.1 --l2 0.1 --seed 42 --attack-type label_flip --attack-clinic clinic_3 --attack-start-round 1 --attack-strength 0.3
```

Pour étudier ensuite la défense robuste, reprendre les mêmes options d'attaque
et remplacer `--strategy fedavg` par `--strategy robust_median --tau 1.3`.
Cela compare la défense face à la même contribution empoisonnée.

Chaque exécution crée son propre dossier `offchain/runs/<run_id>/`, avec
`run_meta.json` (stratégie, graine, paramètres et attaque), les poids par round
et `history.json` (perte, accuracy, cliniques retenues/écartées). Le serveur
actualise également `measures/federation_history.json` au terme du run.

Les commandes ci-dessus lancent réellement l'entraînement. Les exécutions
doivent donc être faites séparément pour produire les courbes et mesurer l'effet
des attaques ; l'ajout du code seul ne constitue pas encore ces résultats.

## Brancher des expériences depuis le même checkpoint sain

Pour comparer des suites d'entraînement qui partent exactement du même modèle
sain, utiliser le fichier global du checkpoint comme point de départ de chaque
run. Par exemple, si le modèle sain après r2 est dans `r002\global_model.npz` :

```powershell
$checkpoint = "C:\Enis\3eme\fed_trace\offchain\runs\<run_sain>\r002\global_model.npz"
```

Les exemples suivants exécutent r3, r4 et r5. `--rounds 5` désigne le dernier
round, pas le nombre de rounds à ajouter. Le serveur copie le checkpoint dans
chaque nouveau dossier de run et ne modifie jamais le fichier source.

Branche témoin sans attaque :

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --initial-model $checkpoint --start-round 3 --rounds 5 --seed 42
```

Branche avec une inversion de signes uniquement à r3 :

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --initial-model $checkpoint --start-round 3 --rounds 5 --seed 42 --attack-type sign_inversion --attack-clinic clinic_3 --attack-start-round 3 --attack-end-round 3 --attack-strength 1
```

Branche avec bruit gaussien uniquement à r3 :

```powershell
.\.venv\Scripts\python.exe -m federation.server --strategy fedavg --initial-model $checkpoint --start-round 3 --rounds 5 --seed 42 --attack-type gaussian_noise --attack-clinic clinic_3 --attack-start-round 3 --attack-end-round 3 --attack-strength 1
```

À r4 et r5, la clinique ne répète alors pas l'attaque, mais les cliniques
repartent du modèle produit au round précédent. On peut donc observer si les
conséquences de l'attaque unique à r3 persistent. Chaque branche est isolée :
la branche attaquée ne change ni le checkpoint sain ni les autres branches.
