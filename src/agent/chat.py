"""Pont entre mo.ui.chat et l'agent : conserve l'historique d'un tour à l'autre"""

from src.agent.agent import build_agent
from src.agent.guardrails import REFUSAL_MESSAGE, check_input
from src.agent.tools import build_deps


def build_chat_model():
    """Retourne une fonction (messages, config) -> str, consommable par mo.ui.chat"""
    agent = build_agent()
    deps = build_deps()
    state = {"history": None}

    async def model(messages, config):
        prompt = messages[-1].content
        if not check_input(prompt).allowed:  # refus déterministe, sans appeler le LLM
            return REFUSAL_MESSAGE
        result = await agent.run(prompt, deps=deps, message_history=state["history"])
        state["history"] = result.all_messages()
        return result.output

    return model
