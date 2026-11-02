"""Embedder partagé par le knowledge graph et la mémoire."""

from collections.abc import Callable
from functools import lru_cache

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  

Embedder = Callable[[list[str]], list[list[float]]]


@lru_cache(maxsize=2)
def _load(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def default_embedder(name: str = EMBED_MODEL) -> Embedder:
    """Fonction d'embedding (textes -> vecteurs normalisés), modèle mis en cache."""

    def embed(texts: list[str]) -> list[list[float]]:
        return _load(name).encode(list(texts), normalize_embeddings=True).tolist()

    return embed
