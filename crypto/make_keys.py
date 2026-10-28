"""Genere les cles Ed25519 des membres (B2). A lancer une seule fois."""
import json
import os

from crypto.signing import generate_keypair

MEMBERS = {
    "clinic_1": {"role": "clinic", "allowed_types": ["contribution"]},
    "clinic_2": {"role": "clinic", "allowed_types": ["contribution"]},
    "clinic_3": {"role": "clinic", "allowed_types": ["contribution"]},
    "server":   {"role": "server", "allowed_types": ["model", "decision"]},
}

PRIVATE_DIR = "secrets"      # hors depot (ajoute au .gitignore)
MEMBERS_FILE = "members.json"  # public, versionne


def main():
    if os.path.exists(MEMBERS_FILE):
        raise SystemExit("members.json existe deja : cles non regenerees.")
    os.makedirs(PRIVATE_DIR, exist_ok=True)
    public = {}
    for name, info in MEMBERS.items():
        priv, pub = generate_keypair()
        with open(os.path.join(PRIVATE_DIR, f"{name}.key"), "w") as f:
            f.write(priv)
        public[name] = {"public_key": pub, **info}
    with open(MEMBERS_FILE, "w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, sort_keys=True)
    print("Cles generees pour :", ", ".join(public))


if __name__ == "__main__":
    main()