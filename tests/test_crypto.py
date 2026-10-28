import pytest

from crypto.members import check_record
from crypto.pseudonyms import case_pseudo
from crypto.signing import canonical_bytes, generate_keypair, sign, verify

KEY = bytes(range(32))


@pytest.fixture(scope="module")
def world():
    keys, members = {}, {}
    roles = {
        "clinic_1": ("clinic", ["contribution"]),
        "clinic_2": ("clinic", ["contribution"]),
        "server": ("server", ["model", "decision"]),
    }
    for name, (role, types) in roles.items():
        priv, pub = generate_keypair()
        keys[name] = priv
        members[name] = {"public_key": pub, "role": role, "allowed_types": types}
    return keys, members


def contribution():
    return {"type": "contribution", "signer": "clinic_1", "n_samples": 4390, "round": 1}


def test_valide(world):
    keys, members = world
    r = contribution()
    assert check_record(r, sign(r, keys["clinic_1"]), members) == (True, "ok")


def test_un_caractere_modifie_est_rejete(world):
    keys, members = world
    r = contribution()
    s = sign(r, keys["clinic_1"])
    r["n_samples"] = 4391
    ok, raison = check_record(r, s, members)
    assert not ok and raison == "signature invalide"


def test_cle_dune_autre_clinique_est_rejetee(world):
    keys, members = world
    r = contribution()
    s = sign(r, keys["clinic_2"])  # clinic_2 se fait passer pour clinic_1
    assert check_record(r, s, members) == (False, "signature invalide")


def test_clinique_ne_peut_pas_signer_un_modele(world):
    keys, members = world
    r = {"type": "model", "signer": "clinic_1", "round": 1}
    s = sign(r, keys["clinic_1"])  # signature techniquement valide
    assert check_record(r, s, members) == (False, "type non autorise pour ce signataire")


def test_serveur_ne_peut_pas_signer_une_contribution(world):
    keys, members = world
    r = {"type": "contribution", "signer": "server", "n_samples": 1}
    assert not check_record(r, sign(r, keys["server"]), members)[0]


def test_signataire_inconnu(world):
    keys, members = world
    r = {"type": "contribution", "signer": "clinic_9"}
    assert check_record(r, sign(r, keys["clinic_1"]), members) == (False, "signataire inconnu")


def test_serialisation_independante_de_lordre():
    a = {"type": "model", "signer": "server", "round": 2}
    b = {"round": 2, "signer": "server", "type": "model"}
    assert canonical_bytes(a) == canonical_bytes(b)


def test_champ_signature_exclu_de_la_serialisation(world):
    keys, _ = world
    r = contribution()
    s = sign(r, keys["clinic_1"])
    assert canonical_bytes(dict(r, signature=s)) == canonical_bytes(r)


def test_signature_hex_corrompue_ne_plante_pas(world):
    keys, members = world
    r = contribution()
    assert not verify(r, "zz", members["clinic_1"]["public_key"])


def test_pseudo_stable_pour_une_cle():
    assert case_pseudo("patient_001", key=KEY) == case_pseudo("patient_001", key=KEY)


def test_pseudo_different_selon_le_cas():
    assert case_pseudo("patient_001", key=KEY) != case_pseudo("patient_002", key=KEY)


def test_pseudo_different_selon_la_cle():
    assert case_pseudo("patient_001", key=KEY) != case_pseudo("patient_001", key=bytes(32))