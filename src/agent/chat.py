"""Pont entre mo.ui.chat et l'agent : conserve l'historique d'un tour à l'autre."""

import asyncio
import logging
import sys

from src.agent.agent import build_agent, trim_history
from src.agent.eval import extract_tool_trace
from src.agent.guardrails import REFUSAL_MESSAGE, check_input
from src.agent.tools import build_deps

TIMEOUT_S = 90
logger = logging.getLogger(__name__)


def _log_tools(result) -> None:
    """Affiche dans le terminal les outils appelés pendant CE tour (+ seeds du GraphRAG)"""
    trace = extract_tool_trace(result.new_messages())
    calls = [t["appel"] for t in trace if "appel" in t]
    if not calls:
        print("[agent] aucun outil appelé (réponse depuis la connaissance du LLM)", file=sys.stderr)
        return
    print(f"[agent] outils appelés : {calls}", file=sys.stderr)
    for t in trace:
        if t.get("resultat_de") == "search_knowledge":
            print(f"        search_knowledge (GraphRAG) → {t['valeur'][:220]}", file=sys.stderr)


def build_chat_model():
    """Retourne une fonction (messages, config) -> str, consommable par mo.ui.chat"""
    agent = build_agent()
    deps = build_deps()
    state = {"history": None}

    async def model(messages, config):
        prompt = messages[-1].content
        if not check_input(prompt).allowed:  # refus déterministe, sans appeler le LLM
            return REFUSAL_MESSAGE
        # On ne renvoie au LLM que les derniers tours -> borne la consommation de tokens
        history = trim_history(state["history"]) if state["history"] else None
        try:
            result = await asyncio.wait_for(
                agent.run(prompt, deps=deps, message_history=history),
                timeout=TIMEOUT_S,
            )
        except TimeoutError:
            return (
                "⏱️ L'agent met trop de temps à répondre — souvent une limite de débit Groq "
                "(8000 tokens/min). Patiente une minute puis réessaie."
            )
        except Exception as exc:
            logger.exception("Erreur pendant l'exécution de l'agent")  # trace complète en logs
            return f"⚠️ Erreur de l'agent ({type(exc).__name__}). Réessaie dans un instant."
        _log_tools(result)
        state["history"] = result.all_messages()
        return result.output

    return model
