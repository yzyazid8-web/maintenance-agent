"""Agent ReAct (Pydantic AI) : enregistre les outils et pilote le raisonnement"""

from pydantic_ai import Agent, ModelRetry, RunContext

from src.agent import tools
from src.agent.llm import build_model
from src.agent.tools import Deps, build_deps

SYSTEM_PROMPT = (
    "Tu es un assistant de maintenance prédictive industrielle (dataset AI4I 2020). "
    "Tu réponds en français, de façon concise et factuelle. "
    "N'avance AUCUN chiffre de toi-même : utilise les outils pour les métriques, les risques, "
    "les explications SHAP, les scénarios what-if et les statistiques. "
    "Une machine est identifiée par son index entier. "
    "Pour la CAUSE physique d'une panne et les ACTIONS correctives, appuie-toi sur "
    "diagnose_mode, recommend_actions et explain_mode (ancrés dans un graphe FMEA) : "
    "ne propose jamais une action qui ne provient pas du graphe. "
    "Si la demande sort du périmètre maintenance / modèle, dis-le simplement."
)


def _require_machine(deps: Deps, machine_id: int) -> None:
    if machine_id not in deps.scored.index:
        raise ModelRetry(
            f"machine_id {machine_id} inconnu. Index valides : 0 à {len(deps.scored) - 1}."
        )


def build_agent() -> Agent:
    agent = Agent(build_model(), deps_type=Deps, system_prompt=SYSTEM_PROMPT, retries=2)

    @agent.tool
    def model_metrics(ctx: RunContext[Deps]) -> dict:
        """Performances du modèle (PR-AUC, ROC-AUC)"""
        return tools.get_metrics(ctx.deps)

    @agent.tool
    def feature_importance(ctx: RunContext[Deps], top_n: int = 6) -> dict:
        """Facteurs les plus influents globalement (importance SHAP)"""
        return tools.feature_importance(ctx.deps, top_n)

    @agent.tool
    def predict_risk(ctx: RunContext[Deps], machine_id: int) -> float:
        """Probabilité de panne d'une machine donnée"""
        _require_machine(ctx.deps, machine_id)
        return tools.predict_risk(ctx.deps, machine_id)

    @agent.tool
    def explain_prediction(ctx: RunContext[Deps], machine_id: int, top_n: int = 5) -> dict:
        """Pourquoi une machine est à risque : contributions SHAP locales"""
        _require_machine(ctx.deps, machine_id)
        return tools.explain_prediction(ctx.deps, machine_id, top_n)

    @agent.tool
    def whatif(ctx: RunContext[Deps], machine_id: int, overrides: dict[str, float]) -> dict:
        """Re-score une machine en modifiant des variables BRUTES. Modifiables :
        air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min.
        Exemple d'overrides : {'torque_nm': 30}"""
        _require_machine(ctx.deps, machine_id)
        valid = set(ctx.deps.X.columns)
        unknown = [k for k in overrides if k.split("__", 1)[-1] not in valid]
        if unknown:
            raise ModelRetry(f"Variables inconnues {unknown}. Modifiables : {sorted(valid)}.")
        return tools.whatif(ctx.deps, machine_id, overrides)

    @agent.tool
    def top_risk_machines(ctx: RunContext[Deps], n: int = 5) -> list[dict]:
        """Les n machines les plus à risque"""
        return tools.top_risk_machines(ctx.deps, n)

    @agent.tool
    def dataset_summary(ctx: RunContext[Deps]) -> dict:
        """Volumétrie et taux de panne du parc"""
        return tools.dataset_summary(ctx.deps)

    @agent.tool
    def diagnose_mode(ctx: RunContext[Deps], machine_id: int) -> dict:
        """Mode de panne physique probable d'une machine (graphe FMEA), via ses variables"""
        _require_machine(ctx.deps, machine_id)
        return ctx.deps.kg.diagnose(ctx.deps.X.loc[machine_id].to_dict())

    @agent.tool
    def recommend_actions(ctx: RunContext[Deps], machine_id: int) -> dict:
        """Actions correctives ANCRÉES dans le graphe FMEA pour une machine."""
        _require_machine(ctx.deps, machine_id)
        modes = ctx.deps.kg.link_modes(ctx.deps.X.loc[machine_id].to_dict())
        if not modes:
            return {"modes": [], "actions": {}, "message": "Aucun mode déclenché : pas d'action ciblée."}
        return {"modes": modes, "actions": {m: ctx.deps.kg.actions_for(m) for m in modes}}

    @agent.tool
    def explain_mode(ctx: RunContext[Deps], mode: str) -> dict:
        """Décrit un mode de panne (TWF, HDF, PWF, OSF, RNF) : cause, signature, actions"""
        return ctx.deps.kg.describe_mode(mode.upper())

    return agent


def ask(question: str, agent=None, deps=None, message_history=None):
    """Pose une question à l'agent. Retourne (réponse, messages) pour la continuité multi-tours"""
    agent = agent or build_agent()
    deps = deps if deps is not None else build_deps()
    result = agent.run_sync(question, deps=deps, message_history=message_history)
    return result.output, result.all_messages()
