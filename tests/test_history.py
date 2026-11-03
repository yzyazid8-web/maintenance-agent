"""Bornage de l'historique de conversation renvoyé au LLM (sans LLM)."""

from types import SimpleNamespace as S

from src.agent.agent import MAX_HISTORY_TURNS, trim_history


def _user():
    return S(parts=[S(part_kind="user-prompt")])


def _resp():
    return S(parts=[S(part_kind="text")])


def test_trim_keeps_last_n_turns_and_starts_on_user():
    msgs = []
    for _ in range(MAX_HISTORY_TURNS + 4):
        msgs += [_user(), _resp()]
    trimmed = trim_history(msgs)
    users = [m for m in trimmed if any(p.part_kind == "user-prompt" for p in m.parts)]
    assert len(users) == MAX_HISTORY_TURNS
    # démarre sur un message utilisateur -> pas de retour d'outil orphelin en tête
    assert trimmed[0].parts[0].part_kind == "user-prompt"


def test_trim_noop_on_short_history():
    msgs = [_user(), _resp(), _user(), _resp()]
    assert trim_history(msgs) == msgs


def test_instructions_survive_history_trim(monkeypatch):
    # P0-1 : les consignes (instructions=) doivent être présentes à CHAQUE requête, même quand
    # l'historique fourni ne contient aucun system prompt (cas d'un historique tronqué).
    monkeypatch.setenv("LLM_BACKEND", "ollama")  # build l'agent sans GROQ_API_KEY (modèle overridé)
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from src.agent.agent import SYSTEM_PROMPT, build_agent
    from src.agent.tools import build_deps

    seen = {}

    def capture(messages, info):
        seen["instr"] = getattr(messages[-1], "instructions", None)
        return ModelResponse(parts=[TextPart("ok")])

    agent = build_agent()
    deps = build_deps()
    hist = [ModelRequest.user_text_prompt("q1"), ModelResponse(parts=[TextPart("r1")])]
    with agent.override(model=FunctionModel(capture)):
        agent.run_sync("question tardive", deps=deps, message_history=hist)

    assert seen["instr"] and "MAINTENANCE PRÉDICTIVE" in seen["instr"]
    assert seen["instr"] == SYSTEM_PROMPT
