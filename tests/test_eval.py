"""Tests de la partie déterministe de l'évaluation (sans LLM)."""

from types import SimpleNamespace

from src.agent.eval import extract_tool_calls, load_golden_set


def test_golden_set_loads():
    gs = load_golden_set()
    assert len(gs) >= 5
    assert all("question" in q and "expected_tool" in q for q in gs)


def test_extract_tool_calls_keeps_only_calls():
    msg = SimpleNamespace(
        parts=[
            SimpleNamespace(part_kind="text", content="bla"),
            SimpleNamespace(part_kind="tool-call", tool_name="predict_risk"),
            SimpleNamespace(part_kind="tool-return", tool_name="predict_risk"),
            SimpleNamespace(part_kind="tool-call", tool_name="explain_prediction"),
        ]
    )
    assert extract_tool_calls([msg]) == ["predict_risk", "explain_prediction"]
