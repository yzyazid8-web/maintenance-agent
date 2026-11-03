"""Tests de la mémoire épisodique (embedder factice, sans modèle ni réseau)"""

from src.agent.memory import EpisodicMemory

_VOCAB = ["osf", "usure", "couple", "hdf", "temperature", "machine"]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Embedding déterministe : comptage de mots-clés (suffit pour ranker en test)"""
    return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


def test_remember_and_recall(tmp_path):
    mem = EpisodicMemory(embed_fn=_fake_embed, path=tmp_path / "mem")
    mem.remember("Machine 7593 : surcharge OSF, usure et couple eleves")
    mem.remember("Machine 12 : dissipation thermique HDF, temperature basse")

    res = mem.recall("probleme d'usure et de couple (osf)", k=1)
    assert len(res) == 1
    assert "osf" in res[0]["text"].lower()


def test_recall_on_empty_memory(tmp_path):
    mem = EpisodicMemory(embed_fn=_fake_embed, path=tmp_path / "empty")
    assert mem.recall("quoi que ce soit") == []


def test_remember_stamps_timestamp(tmp_path):
    mem = EpisodicMemory(embed_fn=_fake_embed, path=tmp_path / "ts")
    mem.remember("Machine 7 : usure OSF")
    res = mem.recall("usure osf", k=1)
    assert res and "created_at" in res[0]  # horodatage automatique


def test_recall_filters_by_machine(tmp_path):
    mem = EpisodicMemory(embed_fn=_fake_embed, path=tmp_path / "filter")
    mem.remember("Machine 7 : usure OSF couple eleve", {"machine_id": 7})
    mem.remember("Machine 12 : HDF temperature", {"machine_id": 12})

    res = mem.recall("machine", k=5, where={"machine_id": 7})
    assert len(res) == 1
    assert res[0]["machine_id"] == 7


def test_recall_max_distance_filters_out_far_notes(tmp_path):
    mem = EpisodicMemory(embed_fn=_fake_embed, path=tmp_path / "dist")
    mem.remember("Machine 7 : usure OSF couple eleve", {"machine_id": 7})
    # un seuil de distance nul n'accepte qu'une correspondance quasi parfaite -> ici rien
    assert mem.recall("temperature hdf", k=5, max_distance=0.0) == []
