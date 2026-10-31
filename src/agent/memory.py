"""Mémoire persistante : l'agent stocke et retrouve des diagnostics passés"""

import uuid
from collections.abc import Callable

import chromadb

from src import config

MEMORY_DIR = config.ROOT / "agent_memory"
COLLECTION = "episodes"
EMBED_MODEL = "all-MiniLM-L6-v2"

Embedder = Callable[[list[str]], list[list[float]]]


def _default_embedder() -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed


class EpisodicMemory:
    """Stocke des épisodes (diagnostics, observations) et les retrouve par similarité"""

    def __init__(self, embed_fn: Embedder | None = None, path=None):
        self._embed_fn = embed_fn  # chargé paresseusement si None
        client = chromadb.PersistentClient(path=str(path or MEMORY_DIR))
        self.collection = client.get_or_create_collection(COLLECTION)

    @property
    def _embed(self) -> Embedder:
        if self._embed_fn is None:
            self._embed_fn = _default_embedder()
        return self._embed_fn

    def remember(self, text: str, metadata: dict | None = None) -> str:
        episode_id = uuid.uuid4().hex
        kwargs = {"ids": [episode_id], "documents": [text], "embeddings": self._embed([text])}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self.collection.add(**kwargs)
        return episode_id

    def recall(self, query: str, k: int = 3) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(
            query_embeddings=self._embed([query]), n_results=min(k, count)
        )
        docs = res["documents"][0]
        dists = res["distances"][0]
        return [{"text": d, "distance": round(float(dist), 4)} for d, dist in zip(docs, dists)]
