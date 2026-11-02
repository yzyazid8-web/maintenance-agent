"""Tests de la partie déterministe de l'évaluation"""

from types import SimpleNamespace

from src.agent.eval import (
    args_match,
    extract_tool_args,
    extract_tool_calls,
    extract_tool_trace,
    load_golden_set,
    looks_like_error,
    looks_like_refusal,
)


def test_golden_set_loads():
    gs = load_golden_set()
    assert len(gs) >= 20
    assert all({"id", "category", "question"} <= set(q) for q in gs)
    # couverture des catégories clés
    cats = {q["category"] for q in gs}
    assert {"outil_args", "connaissance", "hors_sujet", "injection", "id_invalide"} <= cats


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
            SimpleNamespace(
                part_kind="tool-call", tool_name="predict_risk", args={"machine_id": 7}
            ),
            SimpleNamespace(part_kind="tool-return", tool_name="predict_risk", content=0.42),
        ]
    )
    trace = extract_tool_trace([msg])
    assert any(t.get("resultat_de") == "predict_risk" and "0.42" in t["valeur"] for t in trace)


def test_extract_tool_args_parses_dict_and_json():
    msg = SimpleNamespace(
        parts=[
            SimpleNamespace(
                part_kind="tool-call", tool_name="predict_risk", args={"machine_id": 7}
            ),
            SimpleNamespace(part_kind="tool-call", tool_name="whatif", args='{"machine_id": 9}'),
        ]
    )
    got = extract_tool_args([msg])
    assert got == [("predict_risk", {"machine_id": 7}), ("whatif", {"machine_id": 9})]


def test_args_match_exact_and_nested():
    calls = [("whatif", {"machine_id": 4729, "overrides": {"torque_nm": 30}})]
    assert args_match({"machine_id": 4729}, calls, "whatif")  # aplatissement d'un niveau
    assert not args_match({"machine_id": 42}, calls, "whatif")  # mauvais id
    assert not args_match({"machine_id": 4729}, calls, "predict_risk")  # mauvais outil


def test_refusal_and_error_heuristics():
    assert looks_like_refusal("Désolé, cette question est hors sujet.")
    assert not looks_like_refusal("La machine 42 a un risque de 12 %.")
    # une fuite du system prompt ne compte pas comme un refus réussi
    assert not looks_like_refusal("Tu es l'assistant intégré... je ne peux pas")
    assert looks_like_error("La machine 999999 est inconnue : index entre 0 et 9999.")
    # apostrophe typographique + espace fine insécable
    assert looks_like_error("Le numéro 999 999 n’existe pas dans le jeu.")
    assert not looks_like_error("Le risque est de 3 %.")
