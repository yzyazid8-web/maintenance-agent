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


def build_model() -> OpenAIChatModel:
    _load_env_file()
    backend = os.environ.get("LLM_BACKEND", "groq").lower()

    if backend == "ollama":
        provider = OpenAIProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama ignore la clé mais le client en exige une
        )
        return OpenAIChatModel(os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"), provider=provider)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante. Renseigne-la dans .env, ou mets LLM_BACKEND=ollama"
        )
    provider = OpenAIProvider(base_url=GROQ_BASE_URL, api_key=api_key)
    return OpenAIChatModel(os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), provider=provider)
