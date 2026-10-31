"""Évaluation de l'agent : précision d'appel d'outil (déterministe) + groundedness (LLM-as-judge).

Les helpers de chargement/extraction sont légers (testables sans LLM). L'évaluation complète
appelle le LLM -> script manuel :  python -m src.agent.eval
"""

import yaml
from pydantic import BaseModel, Field

from src import config

GOLDEN_PATH = config.ROOT / "eval" / "golden_set.yaml"


def load_golden_set(path=GOLDEN_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def extract_tool_calls(messages) -> list[str]:
    """Noms des outils effectivement APPELÉS (parts 'tool-call'), lus dans l'historique."""
    calls = []
    for message in messages:
        for part in getattr(message, "parts", []):
            is_call = (
                getattr(part, "part_kind", None) == "tool-call"
                or type(part).__name__ == "ToolCallPart"
            )
            if is_call:
                calls.append(part.tool_name)
    return calls


def extract_tool_trace(messages) -> list[dict]:
    """Trace (preuves) : appels d'outils ET leurs résultats, pour que le juge puisse vérifier."""
    trace = []
    for message in messages:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", None)
            if kind == "tool-call" or type(part).__name__ == "ToolCallPart":
                trace.append({"appel": part.tool_name, "args": str(getattr(part, "args", ""))})
            elif kind == "tool-return" or type(part).__name__ == "ToolReturnPart":
                trace.append(
                    {"resultat_de": part.tool_name, "valeur": str(getattr(part, "content", ""))}
                )
    return trace


class GroundednessScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = affirmations inventées, 5 = entièrement étayé")
    justification: str


JUDGE_PROMPT = (
    "Tu es un évaluateur rigoureux. On te donne une QUESTION, la TRACE des outils appelés par "
    "un assistant AVEC leurs résultats (les preuves), et sa RÉPONSE. Note de 1 à 5 si la réponse "
    "est ANCRÉE : chaque affirmation factuelle ou chiffrée doit correspondre aux résultats des "
    "outils dans la trace, sans rien inventer. 5 = tout est étayé par la trace ; 1 = affirmations "
    "non étayées ou contradictoires avec la trace."
)


def build_judge():
    from pydantic_ai import Agent

    from src.agent.llm import build_model

    return Agent(build_model(), output_type=GroundednessScore, system_prompt=JUDGE_PROMPT)


def _judge(judge_agent, question: str, answer: str, trace: list[dict]) -> GroundednessScore:
    prompt = f"QUESTION:\n{question}\n\nTRACE DES OUTILS (preuves):\n{trace}\n\nRÉPONSE:\n{answer}"
    return judge_agent.run_sync(prompt).output


def _retry(call, retries: int = 6, base: float = 3.0):
    """Réessaie en cas de rate limit (429) avec backoff exponentiel — utile sur le free tier."""
    import time

    from pydantic_ai.exceptions import ModelRateLimitError

    for attempt in range(retries):
        try:
            return call()
        except ModelRateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(min(base * 2**attempt, 30))


def evaluate(pause: float = 2.0) -> list[dict]:
    """Exécute l'agent sur le golden set et note chaque réponse (nécessite un backend LLM)."""
    import time

    from src.agent.agent import build_agent
    from src.agent.tools import build_deps

    agent = build_agent()
    deps = build_deps()
    judge_agent = build_judge()

    rows = []
    for i, item in enumerate(load_golden_set(), 1):
        question = item["question"]
        print(f"[{i}] {question[:55]}...")
        try:
            res = _retry(lambda q=question: agent.run_sync(q, deps=deps))
            msgs = res.all_messages()
            calls = extract_tool_calls(msgs)
            trace = extract_tool_trace(msgs)
            score = _retry(
                lambda q=question, r=res, t=trace: _judge(judge_agent, q, r.output, t)
            ).score
            tool_ok = item["expected_tool"] in calls
        except Exception as exc:  # une question qui échoue ne casse pas tout le run
            print(f"    erreur : {type(exc).__name__}")
            calls, score, tool_ok = [], 0, False
        rows.append(
            {
                "question": question,
                "expected_tool": item["expected_tool"],
                "tool_ok": tool_ok,
                "groundedness": score,
            }
        )
        time.sleep(pause)  # espacer pour rester sous la limite tokens/minute
    return rows


def to_markdown(rows: list[dict]) -> str:
    n = len(rows)
    tool_ok = sum(r["tool_ok"] for r in rows)
    avg_g = sum(r["groundedness"] for r in rows) / n
    lines = [
        "# Évaluation de l'agent",
        "",
        f"- Précision d'appel d'outil : **{tool_ok / n:.0%}** ({tool_ok}/{n})",
        f"- Groundedness moyenne (1-5) : **{avg_g:.2f}**",
        "",
        "| Question | Outil attendu | Outil OK | Groundedness |",
        "|---|---|:--:|:--:|",
    ]
    for r in rows:
        mark = "✅" if r["tool_ok"] else "❌"
        lines.append(
            f"| {r['question'][:60]} | `{r['expected_tool']}` | {mark} | {r['groundedness']}/5 |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = to_markdown(evaluate())
    print(report)
    (config.ROOT / "eval" / "report.md").write_text(report, encoding="utf-8")
    print("\nRapport écrit : eval/report.md")
