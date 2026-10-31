"""Pont entre mo.ui.chat et l'agent : conserve l'historique d'un tour à l'autre"""

import asyncio

from src.agent.agent import build_agent
from src.agent.guardrails import REFUSAL_MESSAGE, check_input
from src.agent.tools import build_deps

TIMEOUT_S = 90


def build_chat_model():
    """Retourne une fonction (messages, config) -> str, consommable par mo.ui.chat"""
    agent = build_agent()
    deps = build_deps()
    state = {"history": None}

    async def model(messages, config):
        prompt = messages[-1].content
        if not check_input(prompt).allowed:  # refus déterministe, sans appeler le LLM
            return REFUSAL_MESSAGE
        try:
            result = await asyncio.wait_for(
                agent.run(prompt, deps=deps, message_history=state["history"]),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return (
                "⏱️ L'agent met trop de temps à répondre — souvent une limite de débit Groq "
                "(8000 tokens/min). Patiente une minute puis réessaie."
            )
        except Exception as exc:
            return f"⚠️ Erreur de l'agent ({type(exc).__name__}). Réessaie dans un instant."
        state["history"] = result.all_messages()
        return result.output

    return model
