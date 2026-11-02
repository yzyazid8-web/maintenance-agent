"""Garde-fou d'entrée : pré-filtre LÉGER des injections évidentes, avant d'appeler le LLM."""

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore[rz]?\s+(the|all|your|previous|everything|tout\b|toutes?\s+(les|tes))",
    r"ignore[rz]?\s+(les|tes|vos)\s+(instructions|consignes|règles|directives)",
    r"oublie[rz]?\s+(tout\b|tes\s+(instructions|consignes|règles)|toutes?\s+tes)",
    r"disregard\s+(the|all|previous|your)",
    r"forget\s+(your|the|all|previous|everything)",
    r"(ne\s+tiens?\s+plus\s+compte|outrepasse|contourne|passe\s+outre)[^.]{0,20}"
    r"(instructions|consignes|règles|directives)",
    r"(reveal|révèle[rz]?|montre[rz]?|affiche[rz]?|divulgue|print|répète|donne)[^.]{0,40}"
    r"(system[\s-]?prompt|prompt\s+système|tes\s+(instructions|consignes|règles)|"
    r"ton\s+prompt|your\s+(instructions|prompt|rules|system))",
    r"tu es (maintenant|désormais)",
    r"désormais[,\s]+tu\s+(es|dois|vas|réponds)",
    r"you are now",
    r"act as (a|an|if)",
    r"agis comme (un|une|si)",
    r"fais comme si tu",
    r"aucune\s+(règle|consigne|limite|restriction)",
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
    """Valide un message utilisateur avant de le transmettre à l'agent."""
    stripped = (text or "").strip()
    if not stripped:
        return GuardResult(False, "Message vide.")
    if len(stripped) > MAX_INPUT_CHARS:
        return GuardResult(False, f"Message trop long (> {MAX_INPUT_CHARS} caractères).")
    if _INJECTION_RE.search(stripped):
        return GuardResult(False, "Tentative de manipulation des instructions détectée.")
    return GuardResult(True)
