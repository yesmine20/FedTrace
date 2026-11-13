"""File de reprise des ancrages (B4) : ecrite AVANT l'appel au registre."""
import json
import sqlite3
import time

RANK = {"contribution": 0, "model": 1, "decision": 2}


class AnchorQueue:
    def __init__(self, path="anchor_queue.db", max_attempts=5, base_delay=1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.db = sqlite3.connect(path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS anchors (
                   idem_key TEXT PRIMARY KEY,
                   kind TEXT NOT NULL,
                   round_index INTEGER NOT NULL,
                   record TEXT NOT NULL,
                   status TEXT NOT NULL DEFAULT 'pending',
                   attempts INTEGER NOT NULL DEFAULT 0,
                   next_try REAL NOT NULL DEFAULT 0,
                   last_error TEXT
               )"""
        )
        self.db.commit()

    def enqueue(self, idem_key, kind, round_index, record):
        """Ajoute un ancrage. Retourne False si la cle existe deja (pas de doublon)."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO anchors (idem_key, kind, round_index, record) VALUES (?,?,?,?)",
            (idem_key, kind, round_index, json.dumps(record, sort_keys=True)),
        )
        self.db.commit()
        return cur.rowcount == 1

    def due(self, now=None):
        """Ancrages a (re)jouer, dans l'ordre : round, puis contribution < modele < decision.
        Un modele n'est propose que si toutes les contributions de son round sont 'done'."""
        now = time.time() if now is None else now
        rows = self.db.execute(
            """SELECT idem_key, kind, round_index, record, attempts FROM anchors a
               WHERE status='pending' AND next_try<=?
                 AND NOT (kind='model' AND EXISTS (
                       SELECT 1 FROM anchors c
                       WHERE c.kind='contribution' AND c.round_index=a.round_index
                         AND c.status!='done'))""",
            (now,),
        ).fetchall()
        rows.sort(key=lambda r: (r[2], RANK[r[1]], r[0]))
        return [
            {"idem_key": k, "kind": kd, "round_index": ri, "record": json.loads(rec), "attempts": at}
            for k, kd, ri, rec, at in rows
        ]

    def mark_done(self, idem_key):
        self.db.execute("UPDATE anchors SET status='done', last_error=NULL WHERE idem_key=?", (idem_key,))
        self.db.commit()

    def mark_failure(self, idem_key, error, now=None):
        """Un echec : delai croissant, puis 'failed' apres max_attempts (rien n'est efface)."""
        now = time.time() if now is None else now
        (attempts,) = self.db.execute("SELECT attempts FROM anchors WHERE idem_key=?", (idem_key,)).fetchone()
        attempts += 1
        if attempts >= self.max_attempts:
            self.db.execute(
                "UPDATE anchors SET status='failed', attempts=?, last_error=? WHERE idem_key=?",
                (attempts, str(error), idem_key),
            )
        else:
            delay = self.base_delay * 2 ** (attempts - 1)
            self.db.execute(
                "UPDATE anchors SET attempts=?, next_try=?, last_error=? WHERE idem_key=?",
                (attempts, now + delay, str(error), idem_key),
            )
        self.db.commit()

    def counts(self):
        return dict(self.db.execute("SELECT status, COUNT(*) FROM anchors GROUP BY status").fetchall())

    def close(self):
        self.db.close()

    def mark_rejected(self, idem_key, reason):
        """Refus definitif du registre : inutile de rejouer, mais rien n'est efface."""
        self.db.execute(
            "UPDATE anchors SET status='failed', last_error=? WHERE idem_key=?",
            ("rejected: " + str(reason), idem_key),
        )
        self.db.commit()