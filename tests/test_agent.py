from anchoring.agent import AnchorAgent
from anchoring.fake_registry import FakeRegistry
from anchoring.queue import AnchorQueue

FUTUR = 1e12      # bien apres l'heure reelle
PLUS_TARD = 2e12


def setup(tmp_path, **kw):
    reg = FakeRegistry()
    return AnchorAgent(AnchorQueue(str(tmp_path / "q.db"), **kw), reg), reg


def round_1(agent):
    for c in ("clinic_1", "clinic_2", "clinic_3"):
        agent.submit(f"r001:{c}", "contribution", 1, {"clinic_id": c})
    agent.submit("model:r001", "model", 1, {"round_id": "r001"})


def test_round_normal_3_contributions_avant_le_modele(tmp_path):
    agent, reg = setup(tmp_path)
    round_1(agent)
    assert [k for k, _ in reg.log] == ["contribution"] * 3 + ["model"]
    assert agent.queue.counts() == {"done": 4}


def test_registre_arrete_puis_redemarre_rien_perdu(tmp_path):
    agent, reg = setup(tmp_path)
    reg.mode = "down"
    round_1(agent)                           # l'entrainement continue
    assert reg.records == {} and agent.queue.counts() == {"pending": 4}
    reg.mode = "ok"                          # redemarrage
    agent.flush(now=FUTUR)
    assert len(reg.records) == 4             # rien perdu
    assert [k for k, _ in reg.log] == ["contribution"] * 3 + ["model"]
    assert agent.queue.counts() == {"done": 4}


def test_accuse_perdu_pas_de_doublon(tmp_path):
    agent, reg = setup(tmp_path)
    reg.lose_ack_next = True
    agent.submit("r001:clinic_1", "contribution", 1, {"clinic_id": "clinic_1"})
    assert agent.queue.counts() == {"pending": 1}   # l'agent croit que ca a echoue
    agent.flush(now=FUTUR)                          # rejeu : le registre repond 'duplicate'
    assert agent.queue.counts() == {"done": 1}
    assert len(reg.records) == 1 and len(reg.log) == 1


def test_panne_plus_longue_que_le_maximum_failed_visible(tmp_path):
    agent, reg = setup(tmp_path, max_attempts=3)
    reg.mode = "down"
    agent.submit("k", "contribution", 1, {"a": 1})  # essai 1 (heure reelle)
    agent.flush(now=FUTUR)                          # essai 2
    agent.flush(now=PLUS_TARD)                      # essai 3 -> failed
    assert agent.queue.counts() == {"failed": 1}
    row = agent.queue.db.execute("SELECT record FROM anchors WHERE idem_key='k'").fetchone()
    assert row == ('{"a": 1}',)                     # rien d'efface


def test_rejet_du_registre_est_definitif(tmp_path):
    agent, reg = setup(tmp_path)
    reg.reject_keys.add("k")
    agent.submit("k", "contribution", 1, {"a": 1})
    assert agent.queue.counts() == {"failed": 1}
    err = agent.queue.db.execute("SELECT last_error FROM anchors WHERE idem_key='k'").fetchone()[0]
    assert err == "rejected: rejected"