"""Tests de la partie déterministe de l'évaluation (sans LLM)."""

from types import SimpleNamespace

from src.agent.eval import extract_tool_calls, extract_tool_trace, load_golden_set


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


def test_extract_tool_trace_includes_results():
    msg = SimpleNamespace(
        parts=[
            SimpleNamespace(part_kind="tool-call", tool_name="predict_risk", args={"machine_id": 7}),
            SimpleNamespace(part_kind="tool-return", tool_name="predict_risk", content=0.42),
        ]
    )
    trace = extract_tool_trace([msg])
    assert any(t.get("resultat_de") == "predict_risk" and "0.42" in t["valeur"] for t in trace)
