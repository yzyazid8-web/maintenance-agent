"""Lecture de la trace d'exécution de l'agent (appels d'outils + résultats)."""

import json


def _is_call(part) -> bool:
    return getattr(part, "part_kind", None) == "tool-call" or type(part).__name__ == "ToolCallPart"


def extract_tool_calls(messages) -> list[str]:
    """Noms des outils effectivement appelés (parts 'tool-call'), lus dans l'historique."""
    return [p.tool_name for m in messages for p in getattr(m, "parts", []) if _is_call(p)]


def extract_tool_args(messages) -> list[tuple[str, dict]]:
    """Couples (nom d'outil, arguments) des appels. Tolère des args en dict ou en JSON."""
    out = []
    for m in messages:
        for p in getattr(m, "parts", []):
            if not _is_call(p):
                continue
            args = getattr(p, "args", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            out.append((p.tool_name, args or {}))
    return out


def extract_tool_trace(messages) -> list[dict]:
    """Trace (preuves) : appels d'outils et leurs résultats, pour que le juge puisse vérifier."""
    trace = []
    for message in messages:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", None)
            if kind == "tool-call" or type(part).__name__ == "ToolCallPart":
                trace.append({"appel": part.tool_name, "args": str(getattr(part, "args", ""))})
            elif kind == "tool-return" or type(part).__name__ == "ToolReturnPart":
                trace.append(
                    {"resultat_de": part.tool_name, "valeur": str(getattr(part, "content", ""))}
                )
    return trace
