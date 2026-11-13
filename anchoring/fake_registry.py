"""Faux registre pour developper B4 sans plateforme (accepte, refuse, tombe en panne)."""


class RegistryUnavailable(Exception):
    pass


class FakeRegistry:
    def __init__(self):
        self.mode = "ok"              # "ok" ou "down"
        self.reject_keys = set()      # cles refusees (regle violee)
        self.lose_ack_next = False    # ecrit, puis perd l'accuse
        self.records = {}             # idem_key -> enregistrement
        self.log = []                 # ordre reel des ecritures : (kind, idem_key)

    def anchor(self, idem_key, kind, record):
        """Retourne 'ok', 'duplicate' ou 'rejected' (memes reponses que l'interface a 7 operations)."""
        if self.mode == "down":
            raise RegistryUnavailable("registre arrete")
        if idem_key in self.reject_keys:
            return "rejected"
        if idem_key in self.records:
            return "duplicate"
        self.records[idem_key] = record
        self.log.append((kind, idem_key))
        if self.lose_ack_next:
            self.lose_ack_next = False
            raise RegistryUnavailable("accuse perdu")
        return "ok"