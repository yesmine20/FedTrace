"""Controle d'un enregistrement signe : identite, role, signature (B2)."""
import json


def load_members(path="members.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_record(record, signature_hex, members):
    """Retourne (ok, raison). L'enregistrement doit contenir 'signer' et 'type'."""
    from crypto.signing import verify

    signer = record.get("signer")
    if signer not in members:
        return False, "signataire inconnu"
    if record.get("record_type") not in members[signer]["allowed_types"]:
        return False, "type non autorise pour ce signataire"
    if record.get("record_type") == "contribution" and record.get("clinic_id") != signer:
        return False, "clinic_id different du signataire"
    if not verify(record, signature_hex, members[signer]["public_key"]):
        return False, "signature invalide"
    return True, "ok"