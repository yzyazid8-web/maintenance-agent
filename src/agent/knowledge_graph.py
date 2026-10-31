"""Knowledge Graph des modes de panne (FMEA) + GraphRAG déterministe"""

import math
from pathlib import Path

import networkx as nx
import yaml

from src import config

ONTOLOGY_PATH = config.ROOT / "knowledge" / "failure_modes.yaml"


def _power_w(row) -> float:
    return row["torque_nm"] * row["rotational_speed_rpm"] * 2 * math.pi / 60


def _delta_t(row) -> float:
    return row["process_temp_k"] - row["air_temp_k"]


def _wear_x_torque(row) -> float:
    return row["tool_wear_min"] * row["torque_nm"]


class KnowledgeGraph:
    """Ontologie FMEA chargée en graphe NetworkX, avec linking par seuils"""

    def __init__(self, path: Path = ONTOLOGY_PATH):
        self.ontology = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for mode, info in self.ontology.items():
            g.add_node(mode, kind="mode", label=info["label"], description=info["description"])
            for cause in info.get("causes", []):
                g.add_node(cause, kind="cause")
                g.add_edge(mode, cause, rel="a_pour_cause")
            for action in info.get("actions", []):
                g.add_node(action, kind="action")
                g.add_edge(mode, action, rel="action_corrective")
        return g

    def link_modes(self, row) -> list[str]:
        """Modes déclenchés par les seuils, à partir des variables brutes d'une machine."""
        th = self.ontology
        modes = []

        twf = th["TWF"]["thresholds"]
        if twf["tool_wear_low"] <= row["tool_wear_min"] <= twf["tool_wear_high"]:
            modes.append("TWF")

        hdf = th["HDF"]["thresholds"]
        if _delta_t(row) < hdf["delta_t_max"] and row["rotational_speed_rpm"] < hdf["speed_max"]:
            modes.append("HDF")

        pwf = th["PWF"]["thresholds"]
        power = _power_w(row)
        if power < pwf["power_min"] or power > pwf["power_max"]:
            modes.append("PWF")

        osf_threshold = th["OSF"]["thresholds"]["osf_by_type"].get(row["type"])
        if osf_threshold is not None and _wear_x_torque(row) > osf_threshold:
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
            "description": info["description"],
            "signature": info["signature"],
            "causes": info.get("causes", []),
            "actions": info.get("actions", []),
        }

    def diagnose(self, row) -> dict:
        """Mode(s) probable(s) pour une machine + leur description"""
        modes = self.link_modes(row)
        if not modes:
            return {"modes": [], "message": "Aucun mode physique déclenché (risque diffus ou aléatoire)."}
        return {"modes": modes, "details": [self.describe_mode(m) for m in modes]}
