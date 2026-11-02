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


# GraphRAG (embedder factice : vecteurs de comptage de mots-clés, sans réseau)

_VOCAB = ["usure", "outil", "couple", "temperature", "vitesse", "puissance", "refroidissement"]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


def test_semantic_search_returns_relevant_node():
    kg = KnowledgeGraph(embed_fn=_fake_embed)
    hits = kg.semantic_search("problème d'usure de l'outil de coupe", k=3)
    assert hits
    assert any("usure" in h.lower() or "outil" in h.lower() for h in hits)


def test_graphrag_returns_grounded_context():
    kg = KnowledgeGraph(embed_fn=_fake_embed)
    out = kg.graphrag("comment éviter une surcharge liée au couple et à l'usure ?")
    assert out["seeds"]
    assert isinstance(out["context"], str) and out["context"].strip()


def test_graphrag_serializes_graph_structure():
    # La structure (relations/triplets) doit être transmise, pas seulement du texte plat :
    kg = KnowledgeGraph(embed_fn=_fake_embed)
    out = kg.graphrag("usure de l'outil de coupe", k=3, hops=1)
    assert out["relations"], "le sous-graphe devrait exposer des triplets source--rel-->cible"
    assert any("-->" in t for t in out["relations"])
    assert "Relations (graphe)" in out["context"]


def test_expand_subgraph_connects_mode_to_its_nodes():
    kg = KnowledgeGraph(embed_fn=_fake_embed)
    sub = kg.expand_subgraph(["OSF"], hops=1)
    # un mode doit être relié à ses actions/causes
    assert sub.number_of_nodes() > 1
