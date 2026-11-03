"""Knowledge graph FMEA : diagnostic déterministe (seuils) + GraphRAG (sémantique + multi-sauts)."""

from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from src import config
from src.agent.embeddings import Embedder, default_embedder
from src.features import delta_t_k, power_w, wear_x_torque  # formules physiques partagées

ONTOLOGY_PATH = config.ROOT / "knowledge" / "failure_modes.yaml"

# Champ YAML -> (relation d'arête, type de noeud)
_RELATIONS = {
    "causes": ("a_pour_cause", "cause"),
    "symptoms": ("a_pour_symptome", "symptôme"),
    "components": ("affecte", "composant"),
    "actions": ("corrige_par", "action"),
    "sensors": ("mesure_par", "capteur"),
}


class KnowledgeGraph:
    """Ontologie FMEA en graphe NetworkX : diagnostic par seuils + récupération GraphRAG."""

    def __init__(self, path: Path = ONTOLOGY_PATH, embed_fn: Embedder | None = None):
        self.ontology = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.graph = self._build_graph()
        self._embed_fn = embed_fn
        # Index vectoriel en numpy.
        self._node_ids: list[str] | None = None
        self._node_vecs: np.ndarray | None = None

    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for mode, info in self.ontology.items():
            g.add_node(mode, kind="mode", text=f"{info['label']} : {info['description'].strip()}")
            for field, (rel, kind) in _RELATIONS.items():
                for name in info.get(field) or []:
                    if name not in g:
                        g.add_node(name, kind=kind, text=f"{kind} : {name}")
                    g.add_edge(mode, name, rel=rel)
        return g

    # Diagnostic déterministe (seuils physiques)

    def link_modes(self, row) -> list[str]:
        th = self.ontology
        modes = []
        twf = th["TWF"]["thresholds"]
        if twf["tool_wear_low"] <= row["tool_wear_min"] <= twf["tool_wear_high"]:
            modes.append("TWF")
        hdf = th["HDF"]["thresholds"]
        if delta_t_k(row) < hdf["delta_t_max"] and row["rotational_speed_rpm"] < hdf["speed_max"]:
            modes.append("HDF")
        pwf = th["PWF"]["thresholds"]
        power = power_w(row)
        if power < pwf["power_min"] or power > pwf["power_max"]:
            modes.append("PWF")
        osf = th["OSF"]["thresholds"]["osf_by_type"].get(row["type"])
        if osf is not None and wear_x_torque(row) > osf:
            modes.append("OSF")
        return modes

    def actions_for(self, mode: str) -> list[str]:
        if mode not in self.graph:
            return []
        return [n for n in self.graph.successors(mode) if self.graph.nodes[n]["kind"] == "action"]

    def describe_mode(self, mode: str) -> dict:
        info = self.ontology.get(mode)
        if info is None:
            return {}
        return {
            "mode": mode,
            "label": info["label"],
            "description": info["description"].strip(),
            "signature": info["signature"],
            "causes": info.get("causes", []),
            "symptoms": info.get("symptoms", []),
            "components": info.get("components", []),
            "actions": info.get("actions", []),
            "sensors": info.get("sensors", []),
        }

    def diagnose(self, row) -> dict:
        modes = self.link_modes(row)
        if not modes:
            return {
                "modes": [],
                "message": "Aucun mode physique déclenché (risque diffus ou aléatoire).",
            }
        return {"modes": modes, "details": [self.describe_mode(m) for m in modes]}

    # GraphRAG (récupération sémantique + parcours multi-sauts)

    @property
    def _embed(self) -> Embedder:
        if self._embed_fn is None:
            self._embed_fn = default_embedder()
        return self._embed_fn

    def _ensure_index(self) -> None:
        if self._node_vecs is not None:
            return
        self._node_ids = list(self.graph.nodes)
        texts = [self.graph.nodes[n]["text"] for n in self._node_ids]
        self._node_vecs = np.asarray(self._embed(texts), dtype=float)

    def semantic_search(self, query: str, k: int = 3) -> list[str]:
        """Noeuds du graphe les plus proches sémantiquement (similarité cosinus, embeddings normalisés)."""
        self._ensure_index()
        assert self._node_ids is not None and self._node_vecs is not None
        q = np.asarray(self._embed([query])[0], dtype=float)
        sims = self._node_vecs @ q
        top = np.argsort(sims)[::-1][: min(k, len(self._node_ids))]
        return [self._node_ids[i] for i in top]

    def expand_subgraph(self, seeds, hops: int = 2) -> nx.Graph:
        """Étend les nœuds de départ par `hops` sauts (voisins entrants et sortants)."""
        nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            neighbours = set()
            for n in frontier:
                if n in self.graph:
                    neighbours |= set(self.graph.successors(n)) | set(self.graph.predecessors(n))
            frontier = neighbours - nodes
            nodes |= frontier
        return self.graph.subgraph(nodes)

    def graphrag(self, query: str, k: int = 3, hops: int = 1) -> dict:
        """Récupère un sous-graphe pertinent et le sérialise pour le LLM, avec sa structure"""
        seeds = self.semantic_search(query, k)
        sub = self.expand_subgraph(seeds, hops)
        modes = [n for n in sub.nodes if self.graph.nodes[n]["kind"] == "mode"]

        lines = [
            f"### {self.describe_mode(m)['label']}\n{self.describe_mode(m)['description']}"
            for m in modes
        ]

        # Structure du sous-graphe : triplets.
        triplets = [f"{s} --{self.graph.edges[s, t]['rel']}--> {t}" for s, t in sub.edges]
        if triplets:
            lines.append("#### Relations (triplets)")
            lines.extend(f"- {t}" for t in triplets)

        # Éléments partages entre modes.
        bridges = []
        for n in sub.nodes:
            if self.graph.nodes[n]["kind"] == "mode":
                continue
            linked = sorted(m for m in self.graph.predecessors(n) if m in modes)
            if len(linked) >= 2:
                bridges.append(f"{n} relie {', '.join(linked)}")
        if bridges:
            lines.append("#### Éléments partagés entre modes")
            lines.extend(f"- {b}" for b in bridges)

        if not modes and not triplets:
            lines = [f"- {self.graph.nodes[n]['text']}" for n in sub.nodes]

        return {
            "query": query,
            "seeds": list(seeds),
            "modes": modes,
            "relations": triplets,
            "bridges": bridges,
            "context": "\n".join(lines) if lines else "Aucune connaissance pertinente trouvée.",
        }
