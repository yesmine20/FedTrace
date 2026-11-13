"""Agent d'ancrage (B4) : file d'abord, registre ensuite, rejeu en cas d'echec."""
from anchoring.fake_registry import RegistryUnavailable


class AnchorAgent:
    def __init__(self, queue, registry):
        self.queue = queue
        self.registry = registry

    def submit(self, idem_key, kind, round_index, record):
        """Ecrit dans la file AVANT d'appeler le registre, puis tente d'ancrer."""
        self.queue.enqueue(idem_key, kind, round_index, record)
        self.flush()

    def flush(self, now=None):
        """Rejoue tout ce qui est du. S'arrete au premier registre indisponible
        (preserve l'ordre et ne brule qu'un essai par passage)."""
        progress = True
        while progress:
            progress = False
            for item in self.queue.due(now):
                try:
                    status = self.registry.anchor(item["idem_key"], item["kind"], item["record"])
                except RegistryUnavailable as e:
                    self.queue.mark_failure(item["idem_key"], e, now)
                    return
                if status in ("ok", "duplicate"):
                    self.queue.mark_done(item["idem_key"])
                    progress = True
                else:
                    self.queue.mark_rejected(item["idem_key"], status)