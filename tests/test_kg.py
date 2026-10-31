"""Tests du Knowledge Graph FMEA (linking par seuils + traversée), sans LLM"""

from src.agent.knowledge_graph import KnowledgeGraph

BASE = {
    "air_temp_k": 300.0,
    "process_temp_k": 310.0,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
    "type": "L",
}


def test_linking_rules():
    kg = KnowledgeGraph()
    # TWF : usure dans [200, 240]
    assert "TWF" in kg.link_modes({**BASE, "tool_wear_min": 220})
    # HDF : delta_t < 8.6 et vitesse < 1380
    assert "HDF" in kg.link_modes(
        {**BASE, "air_temp_k": 303.0, "process_temp_k": 310.0, "rotational_speed_rpm": 1300}
    )
    # OSF (type L, seuil 11000) : 250 * 60 = 15000
    assert "OSF" in kg.link_modes({**BASE, "tool_wear_min": 250, "torque_nm": 60.0})
    # PWF : puissance très basse (couple et vitesse faibles)
    assert "PWF" in kg.link_modes({**BASE, "torque_nm": 5.0, "rotational_speed_rpm": 1200})
    # Machine "normale" : aucun mode
    assert kg.link_modes(BASE) == []


def test_actions_and_describe():
    kg = KnowledgeGraph()
    assert kg.actions_for("OSF")
    d = kg.describe_mode("OSF")
    assert {"mode", "label", "description", "signature", "causes", "actions"} <= set(d)
    assert kg.describe_mode("INEXISTANT") == {}


def test_graph_built():
    kg = KnowledgeGraph()
    assert "OSF" in kg.graph
    assert kg.graph.number_of_nodes() > 5  # 5 modes + causes + actions
