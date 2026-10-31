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
