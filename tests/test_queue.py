from anchoring.queue import AnchorQueue


def q(tmp_path, **kw):
    return AnchorQueue(str(tmp_path / "q.db"), **kw)


def test_pas_de_doublon(tmp_path):
    f = q(tmp_path)
    assert f.enqueue("r001:clinic_1", "contribution", 1, {"a": 1}) is True
    assert f.enqueue("r001:clinic_1", "contribution", 1, {"a": 1}) is False
    assert f.counts() == {"pending": 1}


def test_survit_a_un_redemarrage(tmp_path):
    f = q(tmp_path)
    f.enqueue("r001:clinic_1", "contribution", 1, {"a": 1})
    f.close()  # simule l'arret de l'agent
    g = q(tmp_path)
    assert [d["idem_key"] for d in g.due()] == ["r001:clinic_1"]


def test_contributions_avant_modele(tmp_path):
    f = q(tmp_path)
    f.enqueue("model:r001", "model", 1, {"m": 1})  # enfile en premier
    f.enqueue("r001:clinic_2", "contribution", 1, {"c": 2})
    f.enqueue("r001:clinic_1", "contribution", 1, {"c": 1})
    assert [d["idem_key"] for d in f.due()] == ["r001:clinic_1", "r001:clinic_2"]  # modele bloque
    f.mark_done("r001:clinic_1")
    f.mark_done("r001:clinic_2")
    assert [d["idem_key"] for d in f.due()] == ["model:r001"]


def test_delai_croissant(tmp_path):
    f = q(tmp_path, base_delay=1.0)
    f.enqueue("k", "contribution", 1, {})
    f.mark_failure("k", "registre arrete", now=100.0)
    assert f.due(now=100.5) == []        # trop tot (delai 1 s)
    assert len(f.due(now=101.0)) == 1
    f.mark_failure("k", "registre arrete", now=101.0)
    assert f.due(now=102.5) == []        # delai passe a 2 s
    assert len(f.due(now=103.0)) == 1


def test_failed_apres_le_maximum_sans_rien_effacer(tmp_path):
    f = q(tmp_path, max_attempts=3)
    f.enqueue("k", "contribution", 1, {"a": 1})
    for _ in range(3):
        f.mark_failure("k", "panne", now=0.0)
    assert f.counts() == {"failed": 1}
    assert f.due(now=1e9) == []
    row = f.db.execute("SELECT record, last_error FROM anchors WHERE idem_key='k'").fetchone()
    assert row == ('{"a": 1}', "panne")


def test_ordre_entre_rounds(tmp_path):
    f = q(tmp_path)
    f.enqueue("r002:clinic_1", "contribution", 2, {})
    f.enqueue("r001:clinic_1", "contribution", 1, {})
    assert [d["round_index"] for d in f.due()] == [1, 2]