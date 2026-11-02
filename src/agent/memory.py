"""Mémoire persistante : l'agent stocke et retrouve des diagnostics passés"""

import uuid
from datetime import UTC, datetime

import chromadb

from src import config
from src.agent.embeddings import Embedder, default_embedder

MEMORY_DIR = config.ROOT / "agent_memory"
COLLECTION = "episodes"


class EpisodicMemory:
    """Stocke des épisodes (diagnostics, observations) et les retrouve par similarité"""

    def __init__(self, embed_fn: Embedder | None = None, path=None):
        self._embed_fn = embed_fn  # chargé paresseusement si None
        client = chromadb.PersistentClient(path=str(path or MEMORY_DIR))
        self.collection = client.get_or_create_collection(COLLECTION)

    @property
    def _embed(self) -> Embedder:
        if self._embed_fn is None:
            self._embed_fn = default_embedder()
        return self._embed_fn

    def remember(self, text: str, metadata: dict | None = None) -> str:
        """Stocke un épisode HORODATÉ. metadata : ex. {'machine_id': 42, 'modes': 'OSF,TWF'}."""
        episode_id = uuid.uuid4().hex
        meta = {"created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        if metadata:
            meta.update({k: v for k, v in metadata.items() if v is not None})
        self.collection.add(
            ids=[episode_id],
            documents=[text],
            embeddings=self._embed([text]),
            metadatas=[meta],
        )
        return episode_id

    def recall(self, query: str, k: int = 3, where: dict | None = None) -> list[dict]:
        """Retrouve des épisodes par similarité. `where` filtre par métadonnée (ex. {'machine_id': 42})."""
        count = self.collection.count()
        if count == 0:
            return []
        kwargs = {"query_embeddings": self._embed([query]), "n_results": min(k, count)}
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        docs = res["documents"][0]
        dists = res["distances"][0]
        metas = (res.get("metadatas") or [[]])[0] or [{}] * len(docs)
        return [
            {"text": d, "distance": round(float(dist), 4), **(m or {})}
            for d, dist, m in zip(docs, dists, metas, strict=False)
        ]
