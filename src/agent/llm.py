"""LLM hybride : client OpenAI-compatible, routé par la variable d'env LLM_BACKEND.

- groq   : gpt-oss-120b via l'API Groq (GROQ_API_KEY) — prod / démo.
- ollama : modèle local (qwen2.5:3b) via le endpoint OpenAI-compatible d'Ollama — offline.
"""

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src import config

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _load_env_file() -> None:
    """Charge le .env du projet dans os.environ (sans écraser les variables déjà définies)"""
    env_path = config.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build(model: str, backend: str) -> OpenAIChatModel:
    """Client OpenAI-compatible (Groq ou Ollama) pour un modèle donné."""
    if backend == "ollama":
        provider = OpenAIProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        return OpenAIChatModel(model, provider=provider)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante. Renseigne-la dans .env, ou mets le backend sur ollama"
        )
    provider = OpenAIProvider(base_url=GROQ_BASE_URL, api_key=api_key)
    return OpenAIChatModel(model, provider=provider)


def resolve_agent() -> tuple[str, str]:
    """(modèle, backend) de l'agent, routé par LLM_BACKEND (groq par défaut)."""
    _load_env_file()
    backend = os.environ.get("LLM_BACKEND", "groq").lower()
    default = "qwen2.5:3b" if backend == "ollama" else "openai/gpt-oss-120b"
    env_var = "OLLAMA_MODEL" if backend == "ollama" else "GROQ_MODEL"
    return os.environ.get(env_var, default), backend


def build_model() -> OpenAIChatModel:
    """Modèle de l'AGENT, routé par LLM_BACKEND (groq par défaut)"""
    return _build(*resolve_agent())


def resolve_judge() -> tuple[str, str]:
    """(modèle, backend) du JUGE d'évaluation, volontairement distinct de l'agent."""
    _load_env_file()
    backend = os.environ.get("JUDGE_BACKEND", os.environ.get("LLM_BACKEND", "groq")).lower()
    default = "llama3.1:8b" if backend == "ollama" else "qwen/qwen3.8-27b"
    return os.environ.get("JUDGE_MODEL", default), backend


def build_judge_model() -> OpenAIChatModel:
    """Modèle JUGE de l'éval : famille différente de l'agent par défaut"""
    return _build(*resolve_judge())
