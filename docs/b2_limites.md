# B2 - Limites

- Les cles privees (secrets/*.key) sont de simples fichiers lisibles sur une
  meme machine : n'importe quel processus ou utilisateur de la machine peut
  signer au nom d'une clinique. La federation etant simulee sur un seul poste,
  l'isolation entre cliniques n'est pas reelle. En production, chaque clinique
  garderait sa cle dans son propre HSM ou coffre, sur son propre site.
- La non-repudiation ne vaut que si la cle privee n'a pas fuite.
- Pas de revocation ni de rotation de cles dans ce prototype.
- La cle des pseudonymes (secrets/pseudo.key) doit rester secrete : qui la
  possede peut verifier si un identifiant correspond a un pseudonyme.