"""Garde-fous d'entrée : refus déterministe (injection, entrée invalide) avant d'appeler le LLM."""

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

# Motifs d'injection de prompt 
_INJECTION_PATTERNS = [
    r"ignore[rz]?\s+(tes|les|toutes?\s+tes|vos|previous|all|your)",
    r"oublie[rz]?\s+(tes|les|toute|vos)",
    r"disregard\s+(the|all|previous|your)",
    r"forget\s+(your|the|all|previous)",
    r"(reveal|montre|affiche|divulgue|donne)[^.]{0,30}(system[\s-]?prompt|instructions|consignes)",
    r"tu es (maintenant|désormais)",
    r"you are now",
    r"act as (a|an|if)",
    r"agis comme",
    r"jailbreak",
    r"developer mode",
    r"override[^.]{0,30}(instructions|rules|guardrails|consignes)",
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


def check_input(text: str) -> GuardResult:
    """Valide un message utilisateur avant de le transmettre à l'agent"""
    stripped = (text or "").strip()
    if not stripped:
        return GuardResult(False, "Message vide.")
    if len(stripped) > MAX_INPUT_CHARS:
        return GuardResult(False, f"Message trop long (> {MAX_INPUT_CHARS} caractères).")
    if _INJECTION_RE.search(stripped):
        return GuardResult(False, "Tentative de manipulation des instructions détectée.")
    return GuardResult(True)
