# data/make_split.py   (lancer depuis la racine : python data/make_split.py)
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED, TEST_SIZE, ALPHA = 42, 0.20, 2.0
N_CLINICS, MIN_PER_CLASS = 3, 149
TARGET, GROUP = "Diagnosis", "source_group_id"

df = pd.read_csv("data/raw/autoimmune_clean.csv")

def split_once(seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df[GROUP]))
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    # classe dominante de chaque groupe d'entraînement
    dom = train.groupby(GROUP)[TARGET].agg(lambda s: s.value_counts().idxmax())

    rng = np.random.default_rng(seed)
    assign = {}
    for c in sorted(dom.unique()):
        groups = dom[dom == c].index.to_numpy()
        rng.shuffle(groups)
        props = rng.dirichlet([ALPHA] * N_CLINICS)           # déséquilibre propre à la classe c
        cuts = (np.cumsum(props)[:-1] * len(groups)).astype(int)
        for k, chunk in enumerate(np.split(groups, cuts), start=1):
            for g in chunk:
                assign[g] = k

    train["clinic"] = train[GROUP].map(assign)
    test["clinic"] = "test"
    return train, test

def is_valid(train, test):
    counts = pd.crosstab(train["clinic"], train[TARGET])
    return (counts.shape[0] == N_CLINICS
            and (counts >= MIN_PER_CLASS).all().all()
            and set(test[TARGET]) == set(df[TARGET]))

for seed in range(SEED, SEED + 500):
    train, test = split_once(seed)
    if is_valid(train, test):
        break
else:
    raise SystemExit("Aucun découpage valide : augmentez ALPHA ou baissez MIN_PER_CLASS")

for k in range(1, N_CLINICS + 1):
    out = Path(f"data/clinic_{k}"); out.mkdir(parents=True, exist_ok=True)
    train[train["clinic"] == k].to_csv(out / f"clinic_{k}.csv", index=False)
Path("data/test").mkdir(parents=True, exist_ok=True)
test.to_csv("data/test/test.csv", index=False)

report = {
    "seed_utilisee": seed, "alpha": ALPHA, "test_size": TEST_SIZE,
    "lignes_par_clinique_et_classe": pd.crosstab(train["clinic"], train[TARGET]).to_dict(),
    "lignes_test": len(test),
    "groupes_par_ensemble": {
        **{f"clinic_{k}": int(train.loc[train["clinic"] == k, GROUP].nunique()) for k in range(1, 4)},
        "test": int(test[GROUP].nunique())},
}
Path("docs/split_report.json").write_text(json.dumps(report, indent=2, default=int))
print(pd.crosstab(train["clinic"], train[TARGET]))
print("graine retenue :", seed)