"""Pseudonymes de cas par HMAC-SHA256 (B2)."""
import hashlib
import hmac
import os
import unicodedata

KEY_FILE = os.path.join("secrets", "pseudo.key")


def generate_key(path=KEY_FILE):
    """Cree la cle secrete des pseudonymes (une seule fois)."""
    if os.path.exists(path):
        raise SystemExit("pseudo.key existe deja : cle non regeneree.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(os.urandom(32).hex())


def load_key(path=KEY_FILE):
    with open(path) as f:
        return bytes.fromhex(f.read().strip())


def case_pseudo(case_id, key=None, length=16):
    """Pseudonyme stable d'un identifiant de cas : 'case_' + debut du HMAC en hex."""
    if key is None:
        key = load_key()
    cid = unicodedata.normalize("NFC", str(case_id).strip())
    digest = hmac.new(key, cid.encode("utf-8"), hashlib.sha256).hexdigest()
    return "case_" + digest[:length]


if __name__ == "__main__":
    generate_key()
    print("Cle de pseudonymisation generee dans", KEY_FILE)