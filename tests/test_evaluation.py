import numpy as np
import pytest

from federation.metrics import (accuracy_6, accuracy_normal, evaluate, feature_columns,
                                load_split, class_names, standardizer, to_xy)


def test_accuracy_6():
    assert accuracy_6([0, 1, 2, 2], [0, 1, 2, 0]) == 0.75


def test_accuracy_normal_ignores_confusion_between_diseases():
    # la classe 2 est Normal : confondre deux maladies (0 et 1) ne compte pas comme erreur
    assert accuracy_normal([0, 1, 2], [1, 0, 2], normal_index=2) == 1.0
    assert accuracy_normal([0, 1, 2], [2, 1, 2], normal_index=2) == pytest.approx(2 / 3)


def test_evaluate_keys():
    r = evaluate([0, 1], [0, 1], ["Normal", "X"])
    assert set(r) == {"accuracy_6", "accuracy_normal"}


def test_61_features_and_no_label_leak():
    clinics, test = load_split()
    cols = feature_columns(clinics[1])
    assert len(cols) == 61
    assert not {"Diagnosis", "source_group_id", "case_pseudo", "clinic"} & set(cols)


def test_standardizer_uses_train_only():
    clinics, test = load_split()
    classes = class_names(clinics)
    cols = feature_columns(clinics[1])
    x, _ = to_xy(clinics[1], classes, cols)
    mu, sd = standardizer(x)
    assert np.allclose(mu, x.mean(axis=0))
    assert (sd > 0).all()