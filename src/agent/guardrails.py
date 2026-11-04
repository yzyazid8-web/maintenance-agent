"""Garde-fou d'entrée : pré-filtre HEURISTIQUE des injections évidentes, avant d'appeler le LLM.

Non exhaustif et ne prétend pas l'être."""

import re
import unicodedata
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

# Motifs comparés à un texte normalisé (minuscules, sans accents) -> les écrire sans accents.
_INJECTION_PATTERNS = [
    # ignore / disregard / forget + OBJET d'injection (pas « ignore the noise », « forget the machine »)
    r"\b(ignore[rz]?|disregard|forget)\b[^.\n]{0,20}\b(instructions?|consignes?|directives?|prompt|"
    r"system\s*prompt|guidelines?|everything\s+(above|before|prior))\b",
    # FR : oublie/ignore + possessif + instructions/consignes/directives
    r"\b(oublie[rz]?|ignore[rz]?)\b[^.\n]{0,12}\b(tes|les|vos)\b[^.\n]{0,8}"
    r"\b(instructions?|consignes?|directives?)\b",
    r"oublie[rz]?\s+tout\s+ce\s+qui\s+(precede|a\s+ete\s+dit)",
    r"ne\s+(tiens?|prends?)\s+plus[^.\n]{0,20}(instructions?|consignes?|directives?)",
    # Exfiltration du prompt système
    r"(reveal|revele[rz]?|montre[rz]?|affiche[rz]?|divulgue|print|repete)[^.\n]{0,40}"
    r"(system\s*prompt|prompt\s+systeme|ton\s+prompt)",
    # Redéfinition de rôle (objet de rôle requis -> évite « tu es maintenant sur quel modele »)
    r"tu es (maintenant|desormais)\s+(un|une|le|la|libre|autorise|capable|sans)\b",
    r"you are now\s+(a|an|the|free|able|allowed|no longer)\b",
    r"\bact as (a|an|if)\b",
    r"agis comme (un|une|si)\b",
    r"\bjailbreak\b",
    r"developer mode",
    r"override[^.\n]{0,30}(instructions?|rules?|guardrails?|consignes?)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

REFUSAL_MESSAGE = (
    "Désolé, je ne peux pas traiter cette demande. Je suis un assistant de maintenance "
    "prédictive : pose-moi une question sur le modèle, les risques de panne, leurs causes "
    "ou les actions correctives."
)


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def _normalize(text: str) -> str:
    """Minuscules, pleine chasse -> ASCII (NFKC), accents retirés."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def check_input(text: str) -> GuardResult:
    """Valide un message utilisateur avant de le transmettre à l'agent."""
    stripped = (text or "").strip()
    if not stripped:
        return GuardResult(False, "Message vide.")
    if len(stripped) > MAX_INPUT_CHARS:
        return GuardResult(False, f"Message trop long (> {MAX_INPUT_CHARS} caractères).")
    if _INJECTION_RE.search(_normalize(stripped)):
        return GuardResult(False, "Tentative de manipulation des instructions détectée.")
    return GuardResult(True)
