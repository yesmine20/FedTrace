import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", message="Unknown solver options")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federation.metrics import (TARGET, class_names, evaluate, feature_columns,  # noqa: E402
                                load_split, mean_std, standardizer, to_xy)

SEEDS = [0, 1, 2, 3, 4]


def fit_predict(train_df, test_df, classes, cols, seed):
    xtr, ytr = to_xy(train_df, classes, cols)
    xte, yte = to_xy(test_df, classes, cols)
    mu, sd = standardizer(xtr)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit((xtr - mu) / sd, ytr)
    return yte, model.predict((xte - mu) / sd)


def main():
    clinics, test = load_split()
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    central = pd.concat(clinics.values(), ignore_index=True)

    rows = []
    for s in SEEDS:
        sub = central.sample(frac=0.8, random_state=s)
        yte, yp = fit_predict(sub, test, classes, cols, s)
        rows.append(evaluate(yte, yp, classes))
    for metric, (m, sd) in mean_std(rows).items():
        print(f"centralisee  {metric}: {m:.3f} +/- {sd:.3f}")

    for k, d in clinics.items():
        rows = []
        for s in SEEDS:
            sub = d.sample(frac=0.8, random_state=s)
            yte, yp = fit_predict(sub, test, classes, cols, s)
            rows.append(evaluate(yte, yp, classes))
        for metric, (m, sd) in mean_std(rows).items():
            print(f"clinic_{k}  {metric}: {m:.3f} +/- {sd:.3f}")
    prop = pd.DataFrame({f"clinic_{k}": d[TARGET].value_counts(normalize=True).reindex(classes).fillna(0)
                         for k, d in clinics.items()})
    prop["moyenne"] = prop.mean(axis=1)
    print(prop.round(3))
    for c in ["clinic_1", "clinic_2", "clinic_3"]:
        tv = 0.5 * np.abs(prop[c] - prop["moyenne"]).sum()
        print(f"{c} : distance a la moyenne = {tv:.3f}")

if __name__ == "__main__":
    main()