"""Signatures Ed25519 (B2). Cles echangees sous forme hexadecimale."""
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

# transforme le message en une suite d’octets toujours identique (champs triés, sans espaces)
def canonical_bytes(record):
    """Octets canoniques : JSON trie, sans espaces, UTF-8, champ 'signature' exclu."""
    r = {k: v for k, v in record.items() if k != "signature"}
    return json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

# crée une paire de clés. La privée reste secrète, la publique est partagée.
def generate_keypair():
    """Retourne (cle_privee_hex, cle_publique_hex)."""
    priv = Ed25519PrivateKey.generate()
    priv_hex = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return priv_hex, pub_hex

# signe le message avec la clé privée.
def sign(record, private_key_hex):
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    return priv.sign(canonical_bytes(record)).hex()

# renvoie True si la signature est bonne, False sinon (message modifié, mauvaise clé, signature invalide).
def verify(record, signature_hex, public_key_hex):
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pub.verify(bytes.fromhex(signature_hex), canonical_bytes(record))
        return True
    except (InvalidSignature, ValueError):
        return False