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
