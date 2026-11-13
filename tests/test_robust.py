import numpy as np
import pytest

from federation.robust import robust_aggregate

N = {"clinic_1": 4390, "clinic_2": 2991, "clinic_3": 3651}
TAU = 3.0


def honest(seed, scale=0.05):
    r = np.random.default_rng(seed)
    return {"W": 1.0 + scale * r.standard_normal((61, 6)), "b": scale * r.standard_normal(6)}


def updates():
    return {"clinic_1": honest(1), "clinic_2": honest(2), "clinic_3": honest(3)}


def flipped(p, k):
    return {"W": 1.0 - k * (p["W"] - 1.0), "b": -k * p["b"]}


def test_nobody_discarded_without_attack():
    r = robust_aggregate(updates(), N, TAU)
    assert r["discarded"] == [] and r["retained"] == ["clinic_1", "clinic_2", "clinic_3"]


def test_sign_flip_x10_is_discarded():
    u = updates()
    u["clinic_3"] = flipped(u["clinic_3"], 10)
    r = robust_aggregate(u, N, TAU)
    assert [d["clinic_id"] for d in r["discarded"]] == ["clinic_3"]


def test_huge_amplitude_gives_no_nan():
    u = updates()
    u["clinic_3"] = {"W": u["clinic_3"]["W"] * 1e6, "b": u["clinic_3"]["b"] * 1e6}
    r = robust_aggregate(u, N, TAU)
    assert np.isfinite(r["params"]["W"]).all()
    assert [d["clinic_id"] for d in r["discarded"]] == ["clinic_3"]


def test_nan_update_is_discarded_not_crashing():
    u = updates()
    u["clinic_2"]["W"][0, 0] = np.nan
    r = robust_aggregate(u, N, TAU)
    assert r["discarded"][0] == {"clinic_id": "clinic_2", "reason": "non_finite"}
    assert np.isfinite(r["params"]["W"]).all()


def test_inf_update_is_discarded():
    u = updates()
    u["clinic_1"]["b"][0] = np.inf
    r = robust_aggregate(u, N, TAU)
    assert r["discarded"][0] == {"clinic_id": "clinic_1", "reason": "non_finite"}


def test_order_of_arrival_does_not_change_result():
    u = updates()
    a = robust_aggregate(u, N, TAU)
    b = robust_aggregate({k: u[k] for k in ["clinic_3", "clinic_1", "clinic_2"]}, N, TAU)
    assert a["retained"] == b["retained"]
    assert np.array_equal(a["params"]["W"], b["params"]["W"])


def test_order_is_bit_identical():
    u = updates()
    a = robust_aggregate(u, N, TAU)
    b = robust_aggregate({k: u[k] for k in ["clinic_3", "clinic_1", "clinic_2"]}, N, TAU)
    assert a["discarded"] == b["discarded"]
    assert a["params"]["W"].tobytes() == b["params"]["W"].tobytes()
    assert a["params"]["b"].tobytes() == b["params"]["b"].tobytes()


def test_two_malicious_out_of_three_defeats_the_defense():
    u = updates()
    u["clinic_2"] = flipped(u["clinic_2"], 10)
    u["clinic_3"] = flipped(u["clinic_3"], 10)
    r = robust_aggregate(u, N, TAU)
    # limite connue : l'attaque reussit, personne n'est ecarte
    assert r["discarded"] == []
    assert r["retained"] == ["clinic_1", "clinic_2", "clinic_3"]


def test_weighted_average_of_retained():
    u = updates()
    r = robust_aggregate(u, N, tau=1e9)
    expected = sum(N[k] * u[k]["W"] for k in u) / sum(N.values())
    assert np.allclose(r["params"]["W"], expected)